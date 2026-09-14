from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v2_models import ResponseSnapshot, ResponseUsage
from arc_agent.v2_openai import QuotaExhausted
from arc_agent.v3_config import V3ExperimentConfig
from arc_agent.v3_dsl import (
    SignatureError,
    mutate_signature,
    solve_signature,
    validate_signature,
    verify_signature,
)
from arc_agent.v3_experiment import V3Orchestrator
from arc_agent.v3_fingerprint import blind_dataset_sha256, fingerprint_task, select_pilot_tasks
from arc_agent.v3_matcher import generate_candidates, sanitize_task
from arc_agent.v3_models import (
    ParameterSpec,
    ProposedSignature,
    SignatureStep,
    StepArgument,
    finalize_signature,
    signature_hash,
)
from arc_agent.v3_prompt import build_signature_prompt
from arc_agent.v3_state import V3State
from arc_agent.v3_teacher import (
    V3ResponsesClient,
    parse_signature_response,
    signature_output_schema,
)


def _task(
    task_id: str,
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    *,
    test_input: list[list[int]] | None = None,
    test_output: list[list[int]] | None = None,
) -> ArcTask:
    return ArcTask(
        task_id=task_id,
        train=[ArcPair(input=input_grid, output=output_grid)],
        test=[
            ArcPair(
                input=test_input or input_grid,
                output=test_output if test_output is not None else output_grid,
            )
        ],
    )


def _identity_proposal(family: str = "rigid-geometry") -> ProposedSignature:
    return ProposedSignature.model_validate(
        {
            "schema_version": 1,
            "family": family,
            "hypothesis": "copy the même structure",
            "preconditions": [],
            "parameters": [],
            "pipeline": [{"op": "identity", "args": []}],
            "invariants": [],
        }
    )


def test_signature_schema_is_strict_canonical_and_harness_owned() -> None:
    first = _identity_proposal()
    second = first.model_copy(update={"hypothesis": "different explanation"})
    assert signature_hash(first) == signature_hash(second)
    finalized = finalize_signature(first, source_task_ids=["source-b", "source-a"])
    assert finalized.source_task_ids == ["source-a", "source-b"]
    assert finalized.canonical_hash == signature_hash(first)
    assert finalized.complexity == 2
    tampered = finalized.model_copy(update={"complexity": 999})
    with pytest.raises(SignatureError, match="complexity"):
        validate_signature(tampered)
    with pytest.raises(ValidationError):
        ProposedSignature.model_validate(
            {**first.model_dump(mode="json"), "python_source": "def solve(): pass"}
        )


def test_signature_rejects_unknown_ops_identifiers_and_excess_literals() -> None:
    unknown = _identity_proposal().model_copy(
        update={"pipeline": [SignatureStep(op="task_specific_magic", args=[])]}
    )
    with pytest.raises(SignatureError, match="unsupported"):
        validate_signature(unknown)
    embedded = _identity_proposal().model_copy(
        update={
            "pipeline": [
                SignatureStep(
                    op="flip",
                    args=[StepArgument(name="axis", value="task-deadbeef")],
                )
            ]
        }
    )
    with pytest.raises(SignatureError, match="identifier"):
        validate_signature(embedded, forbidden_identifiers=["task-deadbeef"])
    embedded_hypothesis = _identity_proposal().model_copy(
        update={"hypothesis": "special case task-deadbeef"}
    )
    with pytest.raises(SignatureError, match="identifier"):
        validate_signature(
            embedded_hypothesis,
            forbidden_identifiers=["task-deadbeef"],
        )
    literal_heavy = _identity_proposal().model_copy(
        update={
            "pipeline": [
                SignatureStep(
                    op="recolor",
                    args=[
                        StepArgument(
                            name="mapping",
                            value=list(range(10)),
                        )
                    ],
                )
            ]
        }
    )
    with pytest.raises(SignatureError, match="literal scalars"):
        validate_signature(literal_heavy, max_literal_scalars=5)


def test_semantic_parameter_generalizes_across_background_colors_and_sizes() -> None:
    proposal = ProposedSignature(
        family="object-relational",
        hypothesis="crop the foreground object",
        parameters=[ParameterSpec(name="bg", value_type="Color", source="background")],
        pipeline=[
            SignatureStep(
                op="crop_background",
                args=[StepArgument(name="background", value="$bg")],
            )
        ],
    )
    signature = finalize_signature(proposal, source_task_ids=["crop"])
    train = [([[0, 0, 0], [0, 2, 0]], [[2]])]
    assert solve_signature(signature, train, [[5, 5, 5, 5], [5, 3, 3, 5]]) == [[3, 3]]


def test_symbolic_recolor_uses_demonstrated_palette_semantics() -> None:
    proposal = ProposedSignature(
        family="color-attribute",
        parameters=[
            ParameterSpec(name="source", value_type="Color", source="rarest_input_color"),
            ParameterSpec(name="target", value_type="Color", source="output_only_color"),
        ],
        pipeline=[
            SignatureStep(
                op="recolor_color",
                args=[
                    StepArgument(name="source", value="$source"),
                    StepArgument(name="target", value="$target"),
                ],
            )
        ],
    )
    signature = finalize_signature(proposal, source_task_ids=["colors"])
    train = [([[0, 0], [0, 2]], [[0, 0], [0, 7]])]
    assert solve_signature(signature, train, [[0, 3, 0]]) == [[0, 7, 0]]
    assert mutate_signature(signature, source_task_ids=["colors"], limit=2)


def test_vector_parameter_infers_translation_from_demonstrations() -> None:
    proposal = ProposedSignature(
        family="object-relational",
        parameters=[
            ParameterSpec(name="offset", value_type="Vector", source="inferred_translation"),
            ParameterSpec(name="bg", value_type="Color", source="background"),
        ],
        pipeline=[
            SignatureStep(
                op="translate_by_vector",
                args=[
                    StepArgument(name="offset", value="$offset"),
                    StepArgument(name="background", value="$bg"),
                ],
            )
        ],
    )
    signature = finalize_signature(proposal, source_task_ids=["translate"])
    train = [
        (
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
        )
    ]
    assert solve_signature(signature, train, [[0, 3, 0], [0, 0, 0], [0, 0, 0]]) == [
        [0, 0, 0],
        [0, 0, 3],
        [0, 0, 0],
    ]


def test_typed_object_pipeline_executes_and_rejects_type_mismatches() -> None:
    proposal = ProposedSignature(
        family="object-relational",
        hypothesis="render the largest object as a cropped mask",
        parameters=[
            ParameterSpec(name="bg", value_type="Color", source="background"),
            ParameterSpec(name="ink", value_type="Color", source="constant", value=2),
        ],
        pipeline=[
            SignatureStep(
                op="extract_scene",
                input_type="Grid",
                output_type="Scene",
                args=[StepArgument(name="background", value="$bg")],
            ),
            SignatureStep(op="scene_objects", input_type="Scene", output_type="ObjectSet"),
            SignatureStep(
                op="select_object",
                input_type="ObjectSet",
                output_type="Object",
                args=[StepArgument(name="criterion", value="largest")],
            ),
            SignatureStep(op="object_mask", input_type="Object", output_type="Mask"),
            SignatureStep(
                op="render_mask",
                input_type="Mask",
                output_type="Grid",
                args=[
                    StepArgument(name="background", value="$bg"),
                    StepArgument(name="output_color", value="$ink"),
                    StepArgument(name="crop", value=True),
                ],
            ),
        ],
    )
    signature = finalize_signature(proposal, source_task_ids=["objects"])
    grid = [[0, 2, 2, 0], [0, 2, 0, 0], [3, 0, 0, 0]]
    assert solve_signature(signature, [(grid, [[2, 2], [2, 0]])], grid) == [
        [2, 2],
        [2, 0],
    ]

    invalid = proposal.model_copy(
        update={
            "pipeline": [
                SignatureStep(op="scene_objects", input_type="Scene", output_type="ObjectSet"),
                *proposal.pipeline[2:],
            ]
        }
    )
    with pytest.raises(SignatureError, match="type mismatch"):
        validate_signature(invalid)


def test_demonstrated_object_rank_transfers_to_new_object_sizes() -> None:
    proposal = ProposedSignature(
        family="object-relational",
        parameters=[
            ParameterSpec(name="bg", value_type="Color", source="background"),
            ParameterSpec(
                name="rank",
                value_type="Scalar",
                source="demonstrated_object_rank",
            ),
            ParameterSpec(name="ink", value_type="Color", source="ranked_object_color"),
        ],
        pipeline=[
            SignatureStep(
                op="extract_scene",
                input_type="Grid",
                output_type="Scene",
                args=[StepArgument(name="background", value="$bg")],
            ),
            SignatureStep(op="scene_objects", input_type="Scene", output_type="ObjectSet"),
            SignatureStep(
                op="select_object_by_rank",
                input_type="ObjectSet",
                output_type="Object",
                args=[
                    StepArgument(name="rank", value="$rank"),
                    StepArgument(name="order", value="area_desc"),
                ],
            ),
            SignatureStep(op="object_mask", input_type="Object", output_type="Mask"),
            SignatureStep(
                op="render_mask",
                input_type="Mask",
                output_type="Grid",
                args=[
                    StepArgument(name="background", value="$bg"),
                    StepArgument(name="output_color", value="$ink"),
                    StepArgument(name="crop", value=True),
                ],
            ),
        ],
    )
    signature = finalize_signature(proposal, source_task_ids=["rank"])
    training_input = [[2, 2, 0, 3], [2, 2, 0, 0], [2, 0, 4, 4]]
    train = [(training_input, [[4, 4]])]
    test_input = [[5, 5, 5, 0, 7], [5, 5, 0, 0, 0], [5, 0, 8, 8, 0]]
    assert solve_signature(signature, train, test_input) == [[8, 8]]


def test_semantic_parameter_source_and_argument_types_are_checked() -> None:
    wrong_source = ProposedSignature(
        family="rigid-geometry",
        parameters=[
            ParameterSpec(name="turns", value_type="Color", source="background"),
        ],
        pipeline=[
            SignatureStep(
                op="rotate",
                args=[StepArgument(name="turns", value="$turns")],
            )
        ],
    )
    with pytest.raises(SignatureError, match="needs Scalar"):
        validate_signature(wrong_source)

    incompatible_source = ProposedSignature(
        family="rigid-geometry",
        parameters=[
            ParameterSpec(name="turns", value_type="Scalar", source="background"),
        ],
        pipeline=[
            SignatureStep(
                op="rotate",
                args=[StepArgument(name="turns", value="$turns")],
            )
        ],
    )
    with pytest.raises(SignatureError, match="incompatible"):
        validate_signature(incompatible_source)


def test_fingerprint_and_prompt_never_use_test_outputs() -> None:
    first = _task(
        "blind",
        [[0]],
        [[1]],
        test_input=[[2, 2]],
        test_output=[[8, 9]],
    )
    second = first.model_copy(
        update={"test": [ArcPair(input=[[2, 2]], output=[[7, 7]])]}
    )
    assert fingerprint_task(first) == fingerprint_task(second)
    assert blind_dataset_sha256([first]) == blind_dataset_sha256([second])
    demo_only = fingerprint_task(first, include_test_inputs=False)
    different_test_input = first.model_copy(
        update={"test": [ArcPair(input=[[9], [9], [9]], output=[[8, 9]])]}
    )
    assert demo_only == fingerprint_task(different_test_input, include_test_inputs=False)
    prompt = build_signature_prompt(
        sanitize_task(first),
        family="color-attribute",
        round_index=0,
        previous=None,
        verification=None,
        independent=False,
    )
    assert json.dumps([[8, 9]], separators=(",", ":")) not in prompt
    assert "test outputs" not in prompt.lower()


def test_matcher_retrieves_then_falls_back_to_global_search() -> None:
    source = _task("source", [[1, 1]], [[1, 1]])
    target = _task("target", [[2, 2, 2]], [[2, 2, 2]])
    signature = finalize_signature(_identity_proposal(), source_task_ids=[source.task_id])
    fingerprints = {source.task_id: fingerprint_task(source)}
    settings = V3ExperimentConfig().search
    candidates, matches = generate_candidates(
        sanitize_task(target), [signature], fingerprints, settings=settings
    )
    assert matches[0].signature_hash == signature.canonical_hash
    assert any(candidate.source_kind == "retrieved" for candidate in candidates)
    assert any(candidate.verification.accepted for candidate in candidates)

    isolated, isolated_matches = generate_candidates(
        sanitize_task(target),
        [signature],
        fingerprints,
        settings=settings,
        excluded_task_ids={source.task_id},
    )
    assert isolated_matches == []
    assert any(candidate.source_kind == "family_search" for candidate in isolated)
    assert any(candidate.verification.accepted for candidate in isolated)


def test_teacher_payload_uses_sol_xhigh_and_signature_schema() -> None:
    settings = V3ExperimentConfig().teacher
    client = V3ResponsesClient(settings=settings, api_key="test", client=None)  # type: ignore[arg-type]
    payload = client.request_payload(
        prompt="task",
        task_id="example",
        phase="pilot",
        round_index=0,
        max_output_tokens=4096,
        previous_response_id=None,
    )
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"]["effort"] == "xhigh"
    assert payload["text"]["format"]["name"] == "arc_v3_game_signature"
    assert "python_source" not in json.dumps(payload["text"]["format"]["schema"])
    schema = signature_output_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False

    def object_schemas(node: object) -> list[dict[str, object]]:
        if isinstance(node, dict):
            nested = [node] if node.get("type") == "object" else []
            return nested + [item for value in node.values() for item in object_schemas(value)]
        if isinstance(node, list):
            return [item for value in node for item in object_schemas(value)]
        return []

    assert all(item.get("additionalProperties") is False for item in object_schemas(schema))


class _FakeTeacher:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def create(self, *, prompt: str, checkpoint=None, **_: object) -> ResponseSnapshot:
        self.calls += 1
        self.prompts.append(prompt)
        family = prompt.split("Target transformation family: ", 1)[1].split(".", 1)[0]
        proposal = _identity_proposal(family)
        body = {
            "id": f"response-{self.calls}",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": proposal.model_dump_json()}
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        snapshot = ResponseSnapshot(
            response_id=body["id"],
            status="completed",
            body=body,
            usage=ResponseUsage(input_tokens=10, output_tokens=5),
        )
        if checkpoint is not None:
            checkpoint(snapshot)
        return snapshot

    def retrieve(self, *_: object, **__: object) -> ResponseSnapshot:
        raise AssertionError("completed fake responses must not be polled")

    def close(self) -> None:
        pass


def _ten_identity_tasks() -> list[ArcTask]:
    tasks: list[ArcTask] = []
    for index in range(10):
        width = index + 1
        color = index % 9 + 1
        grid = [[color for _ in range(width)]]
        tasks.append(_task(f"pilot-{index:02d}", grid, grid))
    return tasks


def test_pilot_is_checkpointed_frozen_then_evaluated_offline(tmp_path: Path) -> None:
    config = V3ExperimentConfig()
    teacher = _FakeTeacher()
    tasks = _ten_identity_tasks()
    selected, selection = select_pilot_tasks(tasks)
    assert len(selected) == 10
    assert {row["family"] for row in selection["pairs"]} == {
        "color-attribute",
        "rigid-geometry",
        "object-relational",
        "counting-compression",
        "repetition-pattern",
    }

    with V3State(tmp_path) as state:
        orchestrator = V3Orchestrator(config, state, client=teacher, sleep=lambda _: None)
        result = orchestrator.build_pilot(tasks, dataset_hash="training")
        assert result.status == "complete"
        assert teacher.calls == 10
        assert state.get_meta("pilot_frozen") is True
        assert state.get_meta("pilot_validation")["own_signature_hidden_test_exact"] == 10
        assert state.summary("pilot")["completed_tasks"] == 10
        assert len(state.signatures()) == 5  # identical family signatures are deduplicated
        assert all('"test_inputs"' in prompt for prompt in teacher.prompts)
        assert all('"test_outputs"' not in prompt for prompt in teacher.prompts)
        assert (tmp_path / "pilot_prompts.jsonl").is_file()

        evaluation = [
            _task("evaluation-a", [[4, 4]], [[4, 4]]),
            _task("evaluation-b", [[6], [6]], [[6], [6]]),
        ]
        offline = V3Orchestrator(config, state, client=object())
        evaluated = offline.evaluate(evaluation, dataset_hash="evaluation")
        assert evaluated.status == "complete"
        report = json.loads((tmp_path / "v3_evaluation_report.json").read_text())
        assert report["strict_task_accuracy"] == 1.0
        assert report["pass_at_2"] == 1.0
        assert report["evaluation_used_network"] is False
        assert (tmp_path / "v3_submission.json").is_file()


def test_frozen_executor_supports_unlabelled_private_tasks(tmp_path: Path) -> None:
    config = V3ExperimentConfig()
    tasks = _ten_identity_tasks()
    with V3State(tmp_path) as state:
        built = V3Orchestrator(
            config, state, client=_FakeTeacher(), sleep=lambda _: None
        ).build_pilot(tasks, dataset_hash="training")
        assert built.status == "complete"
        private = ArcTask(
            task_id="private-a",
            train=[ArcPair(input=[[5]], output=[[5]])],
            test=[ArcPair(input=[[6]], output=None)],
        )
        evaluated = V3Orchestrator(config, state, client=object()).evaluate(
            [private], dataset_hash="private"
        )
        assert evaluated.status == "complete"
        report = json.loads((tmp_path / "v3_evaluation_report.json").read_text())
        assert report["labels_available"] is False
        assert report["strict_task_accuracy"] is None
        assert report["pass_at_2"] is None
        assert report["go_no_go"]["passed"] is None


def test_response_parser_accepts_only_signature_json() -> None:
    proposal = _identity_proposal()
    body = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": proposal.model_dump_json()}],
            }
        ]
    }
    assert parse_signature_response(body) == proposal
    with pytest.raises(ValueError):
        parse_signature_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"python_source":"x"}'}],
                    }
                ]
            }
        )


def test_verify_signature_leave_one_out_is_a_score_not_acceptance_gate() -> None:
    task = ArcTask(
        task_id="identity",
        train=[
            ArcPair(input=[[1]], output=[[1]]),
            ArcPair(input=[[2]], output=[[2]]),
        ],
        test=[ArcPair(input=[[3]], output=None)],
    )
    signature = finalize_signature(_identity_proposal(), source_task_ids=[])
    verification = verify_signature(signature, task)
    assert verification.accepted
    assert verification.leave_one_out_exact == verification.leave_one_out_total == 2

    false_invariant = finalize_signature(
        _identity_proposal().model_copy(update={"invariants": ["change_shape"]}),
        source_task_ids=[],
    )
    rejected = verify_signature(false_invariant, task)
    assert rejected.exact_pairs == 2
    assert rejected.invariant_exact == 0
    assert not rejected.accepted


def test_response_ledger_is_exactly_once_and_preserves_terminal_retry_usage(
    tmp_path: Path,
) -> None:
    with V3State(tmp_path) as state:
        state.initialize_phase(
            phase="pilot",
            tasks=[("task", "rigid-geometry")],
            dataset_hash="data",
            config_hash="config",
            dsl_hash="dsl",
            resume_only=False,
        )
        request = state.prepare_request(
            phase="pilot",
            task_id="task",
            round_index=0,
            prompt="prompt",
            previous_response_id=None,
            token_limit=100,
            model="gpt-5.6-sol",
        )
        failed = ResponseSnapshot(
            response_id="failed-response",
            status="failed",
            body={"id": "failed-response", "status": "failed"},
            usage=ResponseUsage(input_tokens=10, output_tokens=2),
        )
        state.record_response(request["request_key"], failed)
        state.record_response(request["request_key"], failed)
        assert state.usage().input_tokens == 10
        state.record_request_error(request["request_key"], "retrying", "temporary")
        state.reset_expired_response(request["request_key"])
        retried = state.request(request["request_key"])
        assert retried["idempotency_key"] != request["idempotency_key"]
        completed = ResponseSnapshot(
            response_id="completed-response",
            status="completed",
            body={"id": "completed-response", "status": "completed"},
            usage=ResponseUsage(input_tokens=5, output_tokens=3),
        )
        state.record_response(request["request_key"], completed)
        assert state.usage().input_tokens == 15
        assert state.usage().output_tokens == 5
        assert state.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


class _QuotaTeacher:
    def create(self, **_: object) -> ResponseSnapshot:
        raise QuotaExhausted("quota reset pending", code="usage_limit_reached")

    def close(self) -> None:
        pass


def test_quota_pause_does_not_advance_the_active_task(tmp_path: Path) -> None:
    tasks = _ten_identity_tasks()
    selected, _ = select_pilot_tasks(tasks)
    first_task_id = selected[0].task_id
    with V3State(tmp_path) as state:
        result = V3Orchestrator(
            V3ExperimentConfig(), state, client=_QuotaTeacher(), sleep=lambda _: None
        ).build_pilot(tasks, dataset_hash="training")
        assert result.status == "paused_quota"
        row = state.task_row("pilot", first_task_id)
        assert row["status"] == "running"
        assert row["round_index"] == 0
        assert state.summary("pilot")["completed_tasks"] == 0


class _CheckpointThenInterrupt(_FakeTeacher):
    def create(self, *, checkpoint=None, **kwargs: object) -> ResponseSnapshot:
        super().create(checkpoint=checkpoint, **kwargs)
        raise KeyboardInterrupt from None


def test_completed_response_is_reparsed_after_restart_without_duplicate_call(
    tmp_path: Path,
) -> None:
    tasks = _ten_identity_tasks()
    interrupted = _CheckpointThenInterrupt()
    with V3State(tmp_path) as state:
        first = V3Orchestrator(
            V3ExperimentConfig(), state, client=interrupted, sleep=lambda _: None
        ).build_pilot(tasks, dataset_hash="training")
        assert first.status == "paused_user"
        assert interrupted.calls == 1

    resumed_teacher = _FakeTeacher()
    resumed_teacher.calls = 1  # Keep fake response IDs globally unique across the restart.
    with V3State(tmp_path) as state:
        resumed = V3Orchestrator(
            V3ExperimentConfig(), state, client=resumed_teacher, sleep=lambda _: None
        ).build_pilot(tasks, dataset_hash="training", resume_only=True)
        assert resumed.status == "complete"
        assert resumed_teacher.calls == 10  # Nine new calls; the stored first response was reused.
