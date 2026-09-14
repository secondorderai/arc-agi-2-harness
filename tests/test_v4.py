from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from arc_agent.models import ArcTask
from arc_agent.v4_cli import gate_decision
from arc_agent.v4_config import V4Config, content_hash, file_hash, verify_manifest
from arc_agent.v4_experiment import (
    decode_grid,
    initial_messages,
    predictions_from_json,
    render_tool_result,
    solve_one,
    write_outputs,
)
from arc_agent.v4_grammar import final_grammar
from arc_agent.v4_reasoner import (
    ContextOverflow,
    LocalReasoner,
    NativeTemplate,
    final_json,
    parse_tool_calls,
    separate_reasoning,
    without_reasoning,
)
from arc_agent.v4_runtime import LocalServer
from arc_agent.v4_state import (
    ArtifactStore,
    ExperimentRun,
    SymbolicTaskState,
    atomic_json,
    blind_task,
    freeze_split,
    task_hash,
)
from arc_agent.v4_tools import (
    execute_program,
    invoke_tool,
    run_program,
    tool_schema,
    validate_program,
)

TOKENIZER = Path(".runtime/nanbeige/model-hf/tokenizer_config.json")


@pytest.fixture
def task():
    return ArcTask.model_validate(
        {
            "task_id": "fixture",
            "train": [{"input": [[1, 0], [0, 2]], "output": [[1, 0], [0, 2]]}],
            "test": [{"input": [[3, 4]], "output": [[8, 9]]}, {"input": [[5]], "output": [[7]]}],
        }
    )


@pytest.fixture
def run(tmp_path):
    instance = ExperimentRun(tmp_path / "run", {"model": "Nanbeige"})
    yield instance
    instance.close()


@pytest.mark.parametrize(
    "setting",
    [
        {"model_id": "Qwen/Qwen3-4B"},
        {"model_id": "Bonsai"},
        {"model_id": "gpt-6-astra"},
        {"model_revision": "main"},
        {"runtime_revision": "main"},
        {"base_url": "https://api.openai.com"},
        {"base_url": "http://localhost:8094"},
        {"base_url": "http://127.0.0.1.evil.test"},
        {"base_url": "http://u:p@127.0.0.1"},
        {"base_url": "http://127.0.0.1/x"},
        {"budget_usd": 20},
        {"adapter": "legacy"},
        {"max_new_tokens": 4097},
        {"context_tokens": 2048},
        {"min_free_disk_gib": 9},
    ],
)
def test_configuration_rejects_incompatible_or_unsafe(setting):
    with pytest.raises(ValidationError):
        V4Config(**setting)


def test_default_is_local_nanbeige_without_money_controls():
    config = V4Config()
    assert config.model_id == "Nanbeige/Nanbeige4.2-3B"
    assert config.base_url == "http://127.0.0.1:8094"
    assert not any("budget" in field or "cost" in field for field in type(config).model_fields)


def test_manifest_rejects_adapter_or_wrong_model(tmp_path):
    config = V4Config()
    path = tmp_path / "manifest.json"
    payload = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "runtime_revision": config.runtime_revision,
        "quantization": "Q4_K_M",
        "adapter": "qwen-lora",
    }
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="adapters"):
        verify_manifest(path, config)
    payload["model_id"] = "Qwen"
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="lineage"):
        verify_manifest(path, config)


@pytest.mark.parametrize(
    "value", [[], [[True]], [[1.0]], [[10]], [[1], [1, 2]], ["1a"], [""], ["0" * 31], [[0]] * 31]
)
def test_grid_rejects_malformed(value):
    with pytest.raises((ValueError, TypeError)):
        decode_grid(value)


def test_compact_grid_roundtrip():
    assert decode_grid(["019", "283"]) == [[0, 1, 9], [2, 8, 3]]
    assert predictions_from_json({"predictions": [{"attempt_1": ["3"], "attempt_2": [[4]]}]}, 1)[0][
        "attempt_1"
    ] == [[3]]
    with pytest.raises(ValueError):
        predictions_from_json({"predictions": []}, 1)


def test_frozen_split_task_hash_order_and_no_overlap(tmp_path, task):
    tasks = [task.model_copy(update={"task_id": str(i)}) for i in range(1000)]
    path = tmp_path / "split.json"
    split = freeze_split(tasks, path)
    groups = split["groups"]
    assert all(ids == sorted(ids, key=task_hash) for ids in groups.values())
    assert len(set().union(*(set(ids) for ids in groups.values()))) == 1000
    assert 750 < len(groups["training"]) < 850
    assert 60 < len(groups["development"]) < 140
    assert freeze_split(list(reversed(tasks)), path) == split
    with pytest.raises(ValueError, match="changed"):
        freeze_split(tasks[:-1], path)


def test_labels_never_enter_prompts_or_tools(task, run):
    masked = blind_task(task)
    assert all(pair.output is None for pair in masked.test)
    messages = initial_messages(masked, "program", run)
    payload = json.loads(messages[-1]["content"])
    assert set(payload["grids"]) == {
        "train_0_input",
        "train_0_output",
        "test_0_input",
        "test_1_input",
    }
    assert "test_0_output" not in json.dumps(payload)
    with pytest.raises(KeyError):
        invoke_tool(masked, {"name": "read_grid", "arguments": {"grid_id": "test_0_output"}})


def test_artifacts_detect_missing_and_corrupt(tmp_path):
    store = ArtifactStore(tmp_path)
    key = store.put([[1]])
    assert store.get(key) == [[1]]
    with pytest.raises(FileNotFoundError):
        store.get("f" * 64)
    atomic_json(tmp_path / f"{key}.json", [[2]])
    with pytest.raises(ValueError, match="corrupt"):
        store.get(key)


def test_state_rejects_undefined_and_cyclic_definitions():
    with pytest.raises(ValidationError, match="undefined"):
        SymbolicTaskState(task_id="x", definitions={"a": {"description": "A", "depends_on": ["b"]}})
    with pytest.raises(ValidationError, match="cyclic"):
        SymbolicTaskState(task_id="x", definitions={"a": {"description": "A", "depends_on": ["a"]}})


def test_grounding_and_five_checkpoint_roundtrips(tmp_path):
    store = ArtifactStore(tmp_path)
    key = store.put([[1, 2]])
    state = SymbolicTaskState(
        task_id="x",
        artifact_refs=[key],
        observations=[{"artifact": key, "row": 0, "column": 0, "color": 1}],
        hypotheses=[
            {
                "description": "Possibly translate",
                "evidence": [key],
                "counterexamples": [key],
                "status": "possible",
            }
        ],
        unresolved=["Which object segmentation?"],
    )
    original = state.model_dump(mode="json")
    for _ in range(5):
        state = SymbolicTaskState.model_validate(
            store.get(store.put(state.model_dump(mode="json")))
        )
        state.verify_grounding(store)
        assert state.model_dump(mode="json") == original
    state.observations[0].color = 3
    with pytest.raises(ValueError, match="contradicts"):
        state.verify_grounding(store)


@pytest.mark.parametrize(
    "program",
    [
        {"op": "shell"},
        {"op": "identity", "args": {"x": 1}},
        {"op": "scale", "args": {"rows": 100000}},
        {"op": "compose"},
        {"op": "identity", "steps": [{"op": "identity"}]},
        {"op": "recolor", "args": {"mapping": {"1": 20}}},
        {"op": "rotate", "args": {"turns": True}},
    ],
)
def test_bounded_dsl_rejects_unapproved_programs(program):
    with pytest.raises(ValueError):
        validate_program(program)


def test_program_execution_and_real_subprocess(task):
    masked = blind_task(task)
    result = execute_program(masked, {"op": "identity"})
    assert result["verified"] and result["test_predictions"] == [[[3, 4]], [[5]]]
    assert run_program(masked, {"op": "identity"}) == result
    rendered = json.loads(render_tool_result(result))
    assert [decode_grid(g) for g in rendered["train_predictions"]] == result["train_predictions"]
    assert [decode_grid(g) for g in rendered["test_predictions"]] == result["test_predictions"]
    with pytest.raises(ValueError):
        invoke_tool(masked, {"name": "terminal", "arguments": {}})


def test_native_template_matches_huggingface_and_retention_boundaries():
    if not TOKENIZER.exists():
        pytest.skip("download the pinned tokenizer before certifying Phase 0")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER.parent, local_files_only=True, trust_remote_code=False
    )
    renderer = NativeTemplate(TOKENIZER)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "query"},
        {
            "role": "assistant",
            "content": (
                "<tool_call><function=read_grid><parameter=grid_id>"
                "train_0_input</parameter></function></tool_call>"
            ),
            "reasoning_content": "REASONING_SENTINEL",
        },
        {"role": "tool", "content": '{"rows":["12"]}'},
    ]
    rendered = renderer.render(messages, tools=tool_schema())
    official = tokenizer.apply_chat_template(
        messages,
        tools=tool_schema(),
        tokenize=False,
        preserve_thinking=True,
        enable_thinking=True,
        tool_call_format="xml",
        add_generation_prompt=True,
    )
    assert rendered == official
    stripped = renderer.render(messages, tools=tool_schema(), retain=False)
    assert "REASONING_SENTINEL" in rendered and "REASONING_SENTINEL" not in stripped
    assert "<tool_call>" in stripped and '"12"' in stripped
    assert "REASONING_SENTINEL" in messages[2]["reasoning_content"]
    assert rendered.endswith("<think>\n") or rendered.endswith("<think>")


def test_inline_reasoning_removed_without_mutating_history():
    messages = [{"role": "assistant", "content": "<think>secret</think>answer"}]
    assert without_reasoning(messages)[0]["content"] == "answer"
    assert "secret" in messages[0]["content"]
    assert separate_reasoning("reason</think>answer<|im_end|>") == ("reason", "answer")
    assert separate_reasoning("unfinished reasoning") == ("unfinished reasoning", "")


def test_native_tool_parser_and_strict_json():
    calls = parse_tool_calls(
        "<tool_call><function=run_program><parameter=program>"
        '{"op":"identity"}</parameter></function></tool_call>'
    )
    assert calls == [{"name": "run_program", "arguments": {"program": {"op": "identity"}}}]
    assert final_json('```json\n{"a":1}\n```') == {"a": 1}
    for bad in ('{"a":NaN}', "[]", 'prefix {"a":1}'):
        with pytest.raises(ValueError):
            final_json(bad)
    with pytest.raises(ValueError):
        parse_tool_calls('<tool_call>{"name":"x","arguments":{}}</tool_call> extra')


def test_context_overflow_is_not_truncated(task):
    reasoner = object.__new__(LocalReasoner)
    reasoner.config = V4Config()
    reasoner.template = SimpleNamespace(render=lambda *a, **k: "long")
    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1:8094",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"tokens": [1] * 4050})
        ),
    )
    try:
        with pytest.raises(ContextOverflow):
            reasoner.prepare([], [], seed=42, task=task)
    finally:
        reasoner.close()


def test_reasoning_reserves_final_tokens_and_initializes_native_sampler(task):
    reasoner = object.__new__(LocalReasoner)
    reasoner.config = V4Config()
    reasoner.template = SimpleNamespace(render=lambda *a, **k: "<think>\n")
    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1:8094",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"tokens": [1] * 200})
        ),
    )
    try:
        prepared = reasoner.prepare([], [], seed=42, task=task)
        assert prepared["n_predict"] == 2048 and prepared["reasoning_budget_tokens"] == 1024
        assert prepared["generation_prompt"] == "<think>\n"
        assert prepared["reasoning_budget_end_tags"] == ["</think>"]
        assert prepared["grammar_lazy"] and 'root ::= "</think>"' in prepared["grammar"]
    finally:
        reasoner.close()


def test_repeat_attempt_shorthand_expands_to_complete_submission():
    pair = predictions_from_json(
        {"predictions": [{"attempt_1": ["12", "34"], "attempt_2": "same_as_1"}]}, 1
    )[0]
    assert pair["attempt_1"] == pair["attempt_2"] == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "mode,answer,valid",
    [
        ("direct", '{"predictions":[{"attempt_1":["12"],"attempt_2":"same_as_1"}]}', True),
        ("direct", "", False),
        ("direct", '{"predictions":[]}', False),
        ("direct", '{"predictions":[{"attempt_1":["1x"],"attempt_2":"same_as_1"}]}', False),
        ("program", '{"program":{"op":"identity"}}', True),
        (
            "program",
            '<tool_call><function=run_program><parameter=program>{"op":"identity"}'
            "</parameter></function></tool_call>",
            True,
        ),
        ("program", '{"program":{"op":"shell"}}', False),
        ("program", '{"program":{"op":"rotate","args":{"turns":100}}}', False),
    ],
)
def test_native_final_grammar_compiles_and_rejects_invalid(mode, answer, valid, tmp_path):
    validator = Path(".runtime/nanbeige/gbnf-validator")
    if not validator.exists():
        pytest.skip("build the pinned native grammar validator before local certification")
    grammar_file, answer_file = tmp_path / "grammar.gbnf", tmp_path / "answer.txt"
    grammar_file.write_text(final_grammar(mode, 1, ["train_0_input", "test_0_input"]))
    answer_file.write_text("</think>\n" + answer)
    result = subprocess.run(
        [str(validator), str(grammar_file), str(answer_file)],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert ("Input string is valid according to the grammar." in result.stdout) == valid


class FakeReasoner:
    def __init__(self, final=None):
        self.calls = 0
        self.final = final or json.dumps(
            {
                "predictions": [
                    {"attempt_1": ["89"], "attempt_2": ["34"]},
                    {"attempt_1": ["7"], "attempt_2": ["5"]},
                ]
            }
        )

    def prepare(self, messages, tools, *, seed, task):
        return {"messages": messages.copy(), "tools": tools, "seed": seed}

    def complete(self, prepared, *, timeout):
        self.calls += 1
        return {
            "raw": {},
            "reasoning": "Consider the rule",
            "final": self.final,
            "prompt_tokens": 10,
            "generated_tokens": 20,
            "elapsed_seconds": 0.001,
        }


def test_completed_work_is_not_repeated_on_resume(tmp_path, task):
    reasoner = FakeReasoner()
    root = tmp_path / "run"
    run = ExperimentRun(root, {"v": 1})
    result = solve_one(task, "direct", V4Config(), run, reasoner)
    run.close()
    run = ExperimentRun(root, {"v": 1}, resume=True)
    try:
        assert solve_one(task, "direct", V4Config(), run, reasoner) == result
        assert reasoner.calls == 1
    finally:
        run.close()


def test_crash_after_response_replays_without_model_call(tmp_path, task):
    reasoner = FakeReasoner()
    root = tmp_path / "run"
    run = ExperimentRun(root, {"v": 1})
    original = run.response

    def interrupted(key, response):
        original(key, response)
        raise KeyboardInterrupt()

    run.response = interrupted
    with pytest.raises(KeyboardInterrupt):
        solve_one(task, "direct", V4Config(), run, reasoner)
    run.close()
    run = ExperimentRun(root, {"v": 1}, resume=True)
    try:
        result = solve_one(task, "direct", V4Config(), run, reasoner)
        assert result["status"] == "complete" and reasoner.calls == 1
        assert result["request_count"] == 1
    finally:
        run.close()


def test_replay_uses_saved_request_without_preparing_a_new_context(tmp_path, task):
    root = tmp_path / "adaptive-replay"
    journal = ExperimentRun(root, {"adaptive": True})
    reasoner = FakeReasoner()
    save_response = journal.response

    def interrupted(key, response):
        save_response(key, response)
        raise KeyboardInterrupt()

    journal.response = interrupted
    with pytest.raises(KeyboardInterrupt):
        solve_one(task, "direct", V4Config(), journal, reasoner)
    journal.close()
    journal = ExperimentRun(root, {"adaptive": True}, resume=True)
    reasoner.prepare = lambda *a, **k: pytest.fail("completed response must not be re-prepared")
    try:
        state = solve_one(task, "direct", V4Config(), journal, reasoner)
        assert state["status"] == "complete" and reasoner.calls == 1
    finally:
        journal.close()


@pytest.fixture
def adaptive_runtime(monkeypatch, tmp_path):
    import arc_agent.v4_adaptive as adaptive

    model_path = tmp_path / "config.json"
    atomic_json(model_path, {"max_position_embeddings": 262144})
    manifest = {"files": {"model_config": {"path": str(model_path)}}}
    tracker = SimpleNamespace(active=0, contexts=[], fail_context=None, calls=0)

    class Server:
        def __init__(self, config, manifest, path, **kwargs):
            self.config, self.path, self.kwargs = config, path, kwargs
            self.live = False
            self.baseline = {"swap_used_bytes": 50}
            self.guard_error = None

        def __enter__(self):
            assert tracker.active == 0
            self.path.mkdir(parents=True)
            tracker.contexts.append(self.config.context_tokens)
            if self.config.context_tokens == tracker.fail_context:
                self.guard_error = "memory safety stop"
                raise RuntimeError(self.guard_error)
            tracker.active += 1
            self.live = True
            return self

        def check(self):
            assert self.live

        def __exit__(self, *_):
            if self.live:
                tracker.active -= 1
                self.live = False

    class Reasoner:
        def __init__(self, config, manifest):
            self.config = config

        def prepare(self, messages, tools, *, seed, task):
            count = messages[0]["test_token_count"]
            available = self.config.context_tokens - count - 8
            if available < 128:
                raise ContextOverflow("prompt does not fit")
            return {
                "_prompt_tokens": count,
                "n_predict": min(available, self.config.max_new_tokens),
                "seed": seed,
            }

        def complete(self, prepared, *, timeout):
            tracker.calls += 1
            return {"final": "verified fixture", "timeout": timeout}

        def close(self):
            pass

    def evidence(server, path):
        return {
            "offline_enforced": True,
            "metal_verified": True,
            "memory_safe": not server.guard_error,
            "guard_error": server.guard_error,
            "peak_server_rss_bytes": 100,
            "swap_growth_bytes": 0,
            "peak_pressure_level": 1,
            "sample_count": 1,
        }

    monkeypatch.setattr(adaptive, "LocalServer", Server)
    monkeypatch.setattr(adaptive, "LocalReasoner", Reasoner)
    config = V4Config(context_tokens=8192)
    return adaptive, config, manifest, tracker, evidence


def test_adaptive_expands_twice_with_same_history_deadline_and_single_process(
    adaptive_runtime, tmp_path, task
):
    import time

    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    messages = [{"test_token_count": 14000, "content": "preserve all evidence"}]
    deadline = time.monotonic() + 300
    with adaptive.AdaptiveSession(config, manifest, tmp_path / "run", evidence_fn=evidence) as run:
        run.set_deadline(deadline)
        prepared = run.prepare(messages, [], seed=42, task=task)
        assert tracker.contexts == [8192, 12288, 16384]
        assert tracker.calls == 0 and tracker.active == 1
        assert run.deadline == deadline and run.server.kwargs["deadline"] == deadline
        assert run.server.kwargs["swap_baseline_bytes"] == 50
        assert prepared["n_predict"] == 2048 and prepared["_context_tokens"] == 16384
        assert messages == [{"test_token_count": 14000, "content": "preserve all evidence"}]
        assert all(e["status"] == "ready" for e in run.state["events"])
    assert tracker.active == 0


def test_adaptive_reserves_full_answer_space(adaptive_runtime, tmp_path, task):
    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    with adaptive.AdaptiveSession(config, manifest, tmp_path / "run", evidence_fn=evidence) as run:
        result = run.prepare([{"test_token_count": 7000}], [], seed=42, task=task)
        assert result["n_predict"] == 2048 and tracker.contexts == [8192, 12288]


def test_adaptive_memory_failure_stops_escalation(adaptive_runtime, tmp_path, task):
    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    tracker.fail_context = 12288
    with adaptive.AdaptiveSession(config, manifest, tmp_path / "run", evidence_fn=evidence) as run:
        with pytest.raises(RuntimeError, match="memory safety"):
            run.prepare([{"test_token_count": 9000}], [], seed=42, task=task)
        assert tracker.contexts == [8192, 12288] and tracker.active == 0
        assert not run.evidence()["memory_safe"]
        with pytest.raises(RuntimeError, match="runtime stopped"):
            run.check()


def test_adaptive_expansion_never_resets_task_deadline(adaptive_runtime, tmp_path, task):
    import time

    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    with adaptive.AdaptiveSession(config, manifest, tmp_path / "run", evidence_fn=evidence) as run:
        deadline = time.monotonic() + 10
        run.set_deadline(deadline)
        with pytest.raises(TimeoutError, match="insufficient task time"):
            run.prepare([{"test_token_count": 9000}], [], seed=42, task=task)
        assert tracker.contexts == [8192] and run.deadline == deadline


def test_adaptive_ceiling_and_resume_context_are_preserved(adaptive_runtime, tmp_path, task):
    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    path = tmp_path / "run"
    with adaptive.AdaptiveSession(config, manifest, path, evidence_fn=evidence) as run:
        run.prepare([{"test_token_count": 9000}], [], seed=42, task=task)
    with adaptive.AdaptiveSession(config, manifest, path, evidence_fn=evidence) as resumed:
        assert resumed.state["context_tokens"] == 12288
        assert len(resumed.state["events"]) == 1
        assert resumed.server.kwargs["swap_baseline_bytes"] == 50
    assert tracker.contexts == [8192, 12288, 12288]
    with pytest.raises(ContextOverflow, match="ceiling"):
        adaptive.next_context(262144, 262144)
    with pytest.raises(ValidationError):
        V4Config(context_tokens=262145)


def test_adaptive_resume_preserves_unfinalized_segment_as_unverified(adaptive_runtime, tmp_path):
    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    root = tmp_path / "run"
    orphan = root / "runtime-segments" / "000-context-8192"
    orphan.mkdir(parents=True)
    with adaptive.AdaptiveSession(config, manifest, root, evidence_fn=evidence) as run:
        assert run.server.path.name == "001-context-8192"
        assert run.state["segments"][0]["path"] == str(orphan)
        assert not run.evidence()["memory_safe"]
    assert tracker.active == 0


def test_runtime_cumulative_swap_guard_does_not_reset_on_context_reload(monkeypatch, tmp_path):
    import arc_agent.v4_runtime as runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime, "external_network_denied", lambda: True)
    monkeypatch.setattr(runtime.LocalServer, "_claim_runtime", lambda self: None)
    monkeypatch.setattr(
        runtime,
        "memory_sample",
        lambda: {
            "time": 0,
            "pressure_level": 1,
            "swap_used_bytes": 2 * 1024**3,
            "server_rss_bytes": 100,
        },
    )
    server = runtime.LocalServer(V4Config(), {}, tmp_path / "runtime", swap_baseline_bytes=0)
    with pytest.raises(ValueError, match="cumulative swap growth"), server:
        pytest.fail("model must not start above the experiment-wide swap allowance")
    assert server.process is None


def test_inflight_calls_never_silently_repeat(run):
    assert run.request("x", {"a": 1}) is None
    with pytest.raises(ValueError, match="interrupted in-flight"):
        run.request("x", {"a": 1})
    with pytest.raises(ValueError, match="changed"):
        run.request("x", {"a": 2})


def test_inflight_usage_is_unknown_even_before_task_checkpoint(task, run):
    run.request(f"{task.task_id}:direct:0", {"request": "pending"})
    report = write_outputs([task], run, V4Config())["modes"]["direct"]
    assert report["unknown_usage_calls"] == 1 and report["token_counts_are_lower_bounds"]
    assert report["response_attempts"] == 1 and report["completed_tasks"] == 0


def test_exclusive_run_and_identity(run):
    with pytest.raises(ValueError, match="another process"):
        ExperimentRun(run.root, {"model": "Nanbeige"}, resume=True)


def test_malformed_or_timeout_still_produces_all_fallbacks(task, run):
    result = solve_one(task, "direct", V4Config(), run, FakeReasoner(final="broken"))
    assert result["used_fallback"] and result["errors"] and result["valid_responses"] == 0
    report = write_outputs([task], run, V4Config())
    for mode in ("direct", "program"):
        payload = json.loads((run.root / f"submission-{mode}.json").read_text())
        assert len(payload[task.task_id]) == 2
        assert all(set(item) == {"attempt_1", "attempt_2"} for item in payload[task.task_id])
    assert report["modes"]["direct"]["valid_response_rate"] == 0


@pytest.mark.parametrize("error_type", [TimeoutError, MemoryError])
def test_inference_failure_is_counted(task, run, error_type):
    class TimeoutReasoner(FakeReasoner):
        def complete(self, prepared, *, timeout):
            raise error_type("inference failure")

    result = solve_one(task, "direct", V4Config(), run, TimeoutReasoner())
    assert result["request_count"] == 1 and result["used_fallback"]
    assert error_type.__name__ in result["errors"][0]
    assert result["unknown_usage_calls"] == 1


@pytest.mark.parametrize("pressure,exit_code", [(1, 0), (2, 1)])
def test_preflight_never_loads_a_model(pressure, exit_code, monkeypatch, tmp_path):
    import arc_agent.v4_cli as cli

    monkeypatch.setattr(
        cli,
        "memory_sample",
        lambda: {"pressure_level": pressure, "swap_used_bytes": 0, "server_rss_bytes": 100},
    )
    monkeypatch.setattr(cli, "external_network_denied", lambda: True)
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not load a model")

    monkeypatch.setattr(cli, "LocalServer", forbidden)
    result = CliRunner().invoke(cli.app, ["preflight", "--output", str(tmp_path / "check.json")])
    assert result.exit_code == exit_code, result.output
    assert json.loads((tmp_path / "check.json").read_text())["phase0_passed"] is False


def test_atomic_failure_preserves_prior_submission(tmp_path):
    path = tmp_path / "submission.json"
    atomic_json(path, {"old": 1})
    previous = file_hash(path)
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert file_hash(path) == previous
    assert list(tmp_path.iterdir()) == [path]


def test_sustained_pressure_stops_owned_server(monkeypatch, tmp_path):
    import arc_agent.v4_runtime as runtime

    server = LocalServer(V4Config(), {}, tmp_path)
    server.baseline = {"swap_used_bytes": 0}
    terminated = []
    server.process = SimpleNamespace(
        pid=123,
        terminate=lambda: terminated.append(True),
        poll=lambda: None,
        wait=lambda **kwargs: 0,
    )
    server.stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: None)
    monkeypatch.setattr(
        runtime,
        "memory_sample",
        lambda _: {"pressure_level": 2, "swap_used_bytes": 0, "server_rss_bytes": 100},
    )
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    server._watch()
    assert len(server.samples) == 3 and terminated == [True]
    assert "memory safety" in server.guard_error


@pytest.mark.parametrize("policy", ["stop", "record"])
@pytest.mark.parametrize("level", [None, 0, 1, 2, 3, 4])
def test_warning_tolerance_is_explicit_and_never_allows_critical(policy, level):
    from arc_agent.v4_runtime import pressure_allows_start

    config = V4Config(warning_pressure_policy=policy)
    assert pressure_allows_start(level, config) == (
        level == 1 or (level == 2 and policy == "record")
    )
    assert V4Config().warning_pressure_policy == "stop"


@pytest.mark.parametrize("failure", [None, "critical", "swap", "telemetry", "disk"])
def test_advisory_warnings_retain_all_other_guards(monkeypatch, tmp_path, failure):
    import arc_agent.v4_runtime as runtime

    server = LocalServer(V4Config(warning_pressure_policy="record"), {}, tmp_path)
    server.baseline = {"swap_used_bytes": 0}
    terminated = []
    server.process = SimpleNamespace(
        pid=123,
        terminate=lambda: terminated.append(True),
        poll=lambda: None,
        wait=lambda **kwargs: 0,
    )
    samples = [dict(pressure_level=2, swap_used_bytes=0, server_rss_bytes=100) for _ in range(6)]
    if failure == "critical":
        samples[-1]["pressure_level"] = 4
    elif failure == "swap":
        samples[-1]["swap_used_bytes"] = 2 * 1024**3
    elif failure == "telemetry":
        samples[-1]["server_rss_bytes"] = None
    stream = iter(samples)
    server.stop = SimpleNamespace(is_set=lambda: len(server.samples) == 6, wait=lambda _: None)
    monkeypatch.setattr(runtime, "memory_sample", lambda _: next(stream))
    monkeypatch.setattr(
        runtime.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            free=(5 if failure == "disk" and len(server.samples) == 6 else 20) * 1024**3
        ),
    )
    server._watch()
    assert len(server.samples) == 6
    assert terminated == ([True] if failure else [])
    assert bool(server.guard_error) == bool(failure)
    assert sum(s["pressure_level"] == 2 for s in server.samples) >= 5


def test_memory_shutdown_has_a_firm_timeout(tmp_path):
    server = LocalServer(V4Config(), {}, tmp_path)
    events = []

    def wait(*, timeout):
        events.append(("wait", timeout))
        if len(events) == 2:
            raise subprocess.TimeoutExpired("owned-server", timeout)
        return 0

    server.process = SimpleNamespace(
        poll=lambda: None,
        wait=wait,
        terminate=lambda: events.append("terminate"),
        kill=lambda: events.append("kill"),
    )
    server._stop_owned_process()
    assert events == ["terminate", ("wait", 2), "kill", ("wait", 2)]


def test_different_runs_cannot_start_two_local_model_processes(tmp_path):
    config = V4Config(manifest=tmp_path / "runtime/manifest.json")
    first = LocalServer(config, {}, tmp_path / "one")
    second = LocalServer(config, {}, tmp_path / "two")
    first._claim_runtime()
    with pytest.raises(ValueError, match="already owns"):
        second._claim_runtime()
    first.__exit__()
    second._claim_runtime()
    second.__exit__()


def test_phase_gate_fails_closed_and_does_not_require_accuracy():
    result = {
        "completed_tasks": 20,
        "total_tasks": 20,
        "valid_response_rate": 0.95,
        "exact_match": 0.0,
    }
    report = {"modes": {"direct": result.copy(), "program": result.copy()}}
    evidence = dict.fromkeys(
        ["offline_enforced", "metal_verified", "memory_safe", "smoke_passed", "resume_verified"],
        True,
    )
    assert gate_decision(report, evidence, {"passed": True})["passed"]
    report["modes"]["direct"]["completed_tasks"] = 19
    assert not gate_decision(report, evidence, {"passed": True})["passed"]
    assert not gate_decision(report, {}, {"passed": True})["passed"]
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})


def diagnostic_config(**settings):
    values = {
        "timing_mode": "diagnostic",
        "task_seconds": 10,
        "deadline_schedule_seconds": (10, 20, 40),
        "max_rounds": 128,
    }
    return V4Config(**{**values, **settings})


@pytest.mark.parametrize(
    "settings",
    [
        {"task_seconds": 900},
        {"max_rounds": 128},
        {"deadline_schedule_seconds": (300, 900)},
        {"timing_mode": "diagnostic"},
        {"timing_mode": "diagnostic", "deadline_schedule_seconds": (300, 200)},
        {"timing_mode": "diagnostic", "deadline_schedule_seconds": (300, 3601)},
        {"timing_mode": "diagnostic", "deadline_schedule_seconds": (300, float("nan"))},
    ],
)
def test_long_deadlines_require_explicit_valid_diagnostic_policy(settings):
    with pytest.raises(ValidationError):
        V4Config(**settings)


def test_diagnostic_extends_across_milestones_without_resetting_time(monkeypatch, task, run):
    import arc_agent.v4_experiment as experiment

    clock = [0.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])

    class TimedReasoner(FakeReasoner):
        def __init__(self):
            super().__init__()
            self.timeouts, self.deadlines = [], []

        def set_deadline(self, deadline):
            self.deadlines.append(deadline)

        def complete(self, prepared, *, timeout):
            self.timeouts.append(timeout)
            response = super().complete(prepared, timeout=timeout)
            clock[0] += 12
            if self.calls == 1:
                response["final"] = ""
            return response

    reasoner = TimedReasoner()
    checkpoints = []
    state = solve_one(
        task,
        "direct",
        diagnostic_config(),
        run,
        reasoner,
        on_checkpoint=lambda: checkpoints.append(True),
    )
    assert state["elapsed_seconds"] == 24 and reasoner.calls == 2
    assert reasoner.timeouts == [40, 28] and reasoner.deadlines == [40, 40]
    assert state["timing"]["allowance_seconds"] == 40
    assert [e["to_seconds"] for e in state["timing"]["deadline_extensions"]] == [20, 40]
    assert state["timing"]["first_valid_response_seconds"] == 24
    assert state["timing"]["stop_reason"] == "prediction_produced"
    assert len(checkpoints) >= 2
    report = write_outputs([task], run, diagnostic_config())
    assert report["timing_mode"] == "diagnostic"
    timing = json.loads((run.root / "timing.json").read_text())["tasks"][0]
    assert timing["first_correct_predictions_seconds"] == 24 and not timing["right_censored"]


def test_diagnostic_time_limit_reports_unresolved_not_solved(monkeypatch, task, run):
    import arc_agent.v4_experiment as experiment

    clock = [0.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])

    class SlowReasoner(FakeReasoner):
        def complete(self, prepared, *, timeout):
            response = super().complete(prepared, timeout=timeout)
            clock[0] += 20
            return response

    reasoner = SlowReasoner(final="unusable")
    state = solve_one(task, "direct", diagnostic_config(), run, reasoner)
    assert state["elapsed_seconds"] == 40 and state["round"] == 2
    assert state["timing"]["stop_reason"] == "time_limit_unresolved"
    write_outputs([task], run, diagnostic_config())
    timing = json.loads((run.root / "timing.json").read_text())["tasks"][0]
    assert timing["first_correct_predictions_seconds"] is None and timing["right_censored"]


def test_diagnostic_stall_limit_preserves_failed_rounds(task, run):
    reasoner = FakeReasoner(final="unusable")
    state = solve_one(task, "direct", diagnostic_config(), run, reasoner)
    assert reasoner.calls == 5 and len(state["errors"]) == 5
    assert state["timing"]["stop_reason"] == "stalled_unresolved"
    assert state["used_fallback"] and len(state["responses"]) == 5


def test_diagnostic_transport_timeout_can_continue_as_new_attempt(task, run):
    class TimeoutThenAnswer(FakeReasoner):
        def complete(self, prepared, *, timeout):
            if self.calls == 0:
                self.calls += 1
                raise httpx.ReadTimeout("timed out")
            return super().complete(prepared, timeout=timeout)

    reasoner = TimeoutThenAnswer()
    state = solve_one(task, "direct", diagnostic_config(), run, reasoner)
    assert reasoner.calls == 2 and not state["used_fallback"]
    assert state["unknown_usage_calls"] == 1
    assert run.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 2


def test_diagnostic_progress_remains_label_blind_and_demo_fit_is_not_test_correct(task, run):
    reasoner = FakeReasoner(final='{"program":{"op":"identity"}}')
    captured = []
    original = reasoner.prepare

    def prepare(messages, tools, *, seed, task):
        captured.append(task)
        return original(messages, tools, seed=seed, task=task)

    reasoner.prepare = prepare
    state = solve_one(task, "program", diagnostic_config(), run, reasoner)
    assert all(pair.output is None for pair in captured[0].test)
    assert state["timing"]["stop_reason"] == "demonstrations_verified"
    write_outputs([task], run, diagnostic_config())
    timing = json.loads((run.root / "timing.json").read_text())["tasks"][1]
    assert timing["first_demo_verified_program_seconds"] is not None
    assert timing["first_correct_predictions_seconds"] is None
    assert timing["right_censored"]


def test_diagnostic_can_never_unlock_phase_one():
    result = {"completed_tasks": 20, "total_tasks": 20, "valid_response_rate": 1.0}
    report = {"timing_mode": "diagnostic", "modes": {"direct": result, "program": result}}
    evidence = dict.fromkeys(
        ["offline_enforced", "metal_verified", "memory_safe", "smoke_passed", "resume_verified"],
        True,
    )
    gate = gate_decision(report, evidence, {"passed": True})
    assert not gate["passed"] and gate["unlocks"] == []
    assert "diagnostic" in gate["failures"][0]


def test_diagnostic_replay_keeps_prior_time_and_extensions(monkeypatch, tmp_path, task):
    import arc_agent.v4_experiment as experiment

    clock = [0.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])

    class TimedReasoner(FakeReasoner):
        def complete(self, prepared, *, timeout):
            response = super().complete(prepared, timeout=timeout)
            clock[0] += 12
            if self.calls == 1:
                response["final"] = "unusable"
            return response

    reasoner = TimedReasoner()
    journal = ExperimentRun(tmp_path / "resume", {"mode": "diagnostic"})
    original = journal.response

    def interrupt_second(key, response):
        original(key, response)
        if key.endswith(":1"):
            raise KeyboardInterrupt()

    journal.response = interrupt_second
    with pytest.raises(KeyboardInterrupt):
        solve_one(task, "direct", diagnostic_config(), journal, reasoner)
    journal.close()
    journal = ExperimentRun(tmp_path / "resume", {"mode": "diagnostic"}, resume=True)
    try:
        state = solve_one(task, "direct", diagnostic_config(), journal, reasoner)
        assert reasoner.calls == 2 and state["elapsed_seconds"] == 24
        assert state["timing"]["allowance_seconds"] == 40
        assert len(state["timing"]["deadline_extensions"]) == 2
    finally:
        journal.close()


def test_diagnostic_distinguishes_token_limits_from_wall_clock_limits(task, run):
    class TokenLimitedReasoner(FakeReasoner):
        def complete(self, prepared, *, timeout):
            response = super().complete(prepared, timeout=timeout)
            response.update({"final": "", "generated_tokens": 2048, "raw": {"stop_type": "limit"}})
            return response

    reasoner = TokenLimitedReasoner()
    state = solve_one(task, "direct", diagnostic_config(), run, reasoner)
    assert state["timing"]["stop_reason"] == "repeated_generation_limit_unresolved"
    assert len(state["timing"]["rounds"]) == 5
    write_outputs([task], run, diagnostic_config())
    summary = json.loads((run.root / "timing.json").read_text())["summary"]["direct"]
    assert summary["generation_limit_stops"] == 5
    assert summary["completed_without_correct_solution"] == 1
    assert summary["median_seconds_among_correct_tasks_only"] is None


def test_correct_timing_uses_both_selected_predictions(task, run):
    answer = {
        "predictions": [
            {"attempt_1": [[8, 9]], "attempt_2": [[0, 0]]},
            {"attempt_1": [[0]], "attempt_2": [[7]]},
        ]
    }
    state = solve_one(
        task, "direct", diagnostic_config(), run, FakeReasoner(final=json.dumps(answer))
    )
    write_outputs([task], run, diagnostic_config())
    timing = json.loads((run.root / "timing.json").read_text())["tasks"][0]
    assert timing["first_correct_predictions_seconds"] == state["elapsed_seconds"]
    assert not timing["right_censored"]
