"""User-approved 4K output protocol cannot inherit earlier 2K fixture evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_v4 import adaptive_runtime as adaptive_runtime
from test_v7 import task as task

from arc_agent.v4_config import V4Config, load_v4_config
from arc_agent.v8_cli import (
    experiment_binding,
    long_context_fixture,
    matching_smoke,
    require_pilot_smoke,
    smoke_contexts,
)
from arc_agent.v8_config import V8Config, load_config


def test_new_output_budget_is_opt_in_and_historical_defaults_unchanged():
    cfg = load_config(Path("configs/v8-nanbeige-output4k-baseline.yaml"))
    runtime = load_v4_config(cfg.runtime_config)
    assert runtime.max_new_tokens == 4096 and runtime.context_tokens == 12288
    assert runtime.max_reasoning_tokens == 1024
    assert smoke_contexts(cfg) == (8192, 12288)
    assert smoke_contexts(V8Config()) == (4096, 8192)
    old = load_config(Path("configs/v8-nanbeige-triples-baseline.yaml"))
    assert load_v4_config(old.runtime_config).max_new_tokens == V4Config().max_new_tokens == 2048
    assert cfg.max_cycles == old.max_cycles == 4
    assert cfg.task_seconds == old.task_seconds == 900
    assert cfg.seeds == old.seeds and cfg.modes == old.modes


@pytest.mark.parametrize(
    "settings",
    [{"max_new_tokens": 4097, "context_tokens": 8192}, {"max_new_tokens": 4096}],
)
def test_output_ceiling_and_full_context_reservation(settings):
    with pytest.raises(ValidationError):
        V4Config(**settings)


@pytest.fixture
def bound_policy(monkeypatch):
    import arc_agent.v8_cli as cli

    monkeypatch.setattr(cli, "source_hash", lambda: "source")
    monkeypatch.setattr(cli, "file_hash", lambda path: "manifest")
    cfg = V8Config(smoke_contexts="8k_then_12k")
    runtime = V4Config(context_tokens=12288, max_new_tokens=4096, prompt_cache_mib=0)
    binding = experiment_binding(cfg, runtime, {})
    report = {
        **binding,
        "passed": True,
        "context_tokens": 12288,
        "long_context_verified": True,
        "long_context_answer_allowance": 4096,
        "prompt_cache_disabled_verified": True,
    }
    return cfg, runtime, binding, report


def test_two_contexts_share_policy_but_cannot_skip_initial_fixture(bound_policy):
    cfg, runtime, binding, report = bound_policy
    initial = runtime.model_copy(update={"context_tokens": 8192})
    assert experiment_binding(cfg, initial, {}) == binding
    assert not matching_smoke(report, binding, 8192)
    require_pilot_smoke(report, binding, cfg, runtime)


@pytest.mark.parametrize(
    "change",
    [
        {"max_new_tokens": 2048},
        {"max_reasoning_tokens": 512},
        {"temperature": 0.5},
        {"cache_type": "f16"},
        {"task_seconds": 250},
        {"warning_pressure_policy": "record"},
        {"max_swap_growth_gib": 2},
        {"history": "strip"},
    ],
)
def test_config_only_changes_invalidate_smoke(bound_policy, change):
    cfg, runtime, binding, report = bound_policy
    other = experiment_binding(cfg, runtime.model_copy(update=change), {})
    assert other != binding and not matching_smoke(report, other, 12288)


@pytest.mark.parametrize("change", [{"max_cycles": 4}, {"smoke_contexts": "4k_then_8k"}])
def test_solver_policy_is_bound(bound_policy, change):
    cfg, runtime, binding, report = bound_policy
    other = experiment_binding(cfg.model_copy(update=change), runtime, {})
    assert not matching_smoke(report, other, 12288)


@pytest.mark.parametrize(
    "change",
    [
        {"context_tokens": 8192},
        {"long_context_answer_allowance": 2048},
        {"long_context_verified": False},
        {"runtime_policy_hash": None},
        {"prompt_cache_disabled_verified": False},
    ],
)
def test_pilot_rejects_incompatible_fixture(bound_policy, change):
    cfg, runtime, binding, report = bound_policy
    with pytest.raises(ValueError, match="matching runtime/output policy"):
        require_pilot_smoke({**report, **change}, binding, cfg, runtime)


def test_pilot_cannot_start_at_unvalidated_context(bound_policy):
    cfg, runtime, binding, report = bound_policy
    with pytest.raises(ValueError):
        require_pilot_smoke(
            report, binding, cfg, runtime.model_copy(update={"context_tokens": 8192})
        )


@pytest.mark.parametrize(
    "context,prompt,allowance,valid",
    [
        (12288, 5571, 4096, True),
        (8192, 5571, 4096, False),
        (12288, 4000, 4096, False),
        (12288, 5571, 2048, False),
    ],
)
def test_long_fixture_reserves_full_cap_without_requiring_waste(context, prompt, allowance, valid):
    cfg = V4Config(context_tokens=context, max_new_tokens=4096)
    calls = []
    reasoner = SimpleNamespace(
        config=cfg,
        prepare_stage=lambda *a, **kw: {"_prompt_tokens": prompt, "n_predict": allowance},
    )
    run = SimpleNamespace(
        request=lambda *a: (
            calls.append(a)
            or {
                "final": '{"predictions":[{"attempt_1":["5","6"],"attempt_2":"same_as_1"}]}',
                "generated_tokens": 50,
            }
        )
    )
    if valid:
        result = long_context_fixture(run, reasoner, None, lambda: None)
        assert result["long_context_verified"] and result["long_context_answer_allowance"] == 4096
        assert result["long_context_generated_tokens"] == 50
    else:
        with pytest.raises(ValueError, match="full answer allowance"):
            long_context_fixture(run, reasoner, None, lambda: None)
        assert not calls


def test_adaptive_4k_output_preserves_cap_deadline_and_single_process(
    adaptive_runtime, tmp_path, task
):
    import time

    adaptive, config, manifest, tracker, evidence = adaptive_runtime
    config = config.model_copy(update={"context_tokens": 12288, "max_new_tokens": 4096})
    messages = [{"test_token_count": 10000, "content": "all evidence remains"}]
    deadline = time.monotonic() + 300
    with adaptive.AdaptiveSession(config, manifest, tmp_path / "run", evidence_fn=evidence) as run:
        run.set_deadline(deadline)
        result = run.prepare(messages, [], seed=42, task=task)
        assert result["n_predict"] == 4096 and tracker.contexts == [12288, 16384]
        assert run.deadline == deadline and tracker.active == 1 and tracker.calls == 0
        assert run.evidence()["context_policy"].startswith("12288 token start")
        assert messages[0]["content"] == "all evidence remains"
    assert tracker.active == 0
