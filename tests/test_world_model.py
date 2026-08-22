from __future__ import annotations

import numpy as np
import pytest

from arc_agent.models import ArcPair, ArcTask, Program
from arc_agent.world_model import (
    PROGRAM_OPS,
    HybridWorldModel,
    WorldModelConfig,
    evaluate_world_model,
    mlx_package_available,
    mlx_runtime_available,
    train_world_model,
)
from arc_agent.world_model_data import (
    build_world_model_examples,
    collate_world_model_examples,
    extract_objects,
    infer_program_label,
    make_augmentations,
    program_label_from_verified_program,
)


def _task() -> ArcTask:
    return ArcTask(
        task_id="recolor-smoke",
        train=[
            ArcPair(input=[[0, 1], [1, 0]], output=[[0, 2], [2, 0]]),
            ArcPair(input=[[0, 2], [2, 0]], output=[[0, 1], [1, 0]]),
        ],
        test=[ArcPair(input=[[0, 1], [1, 0]])],
    )


def test_object_adapter_is_deterministic_and_uses_rich_scene_semantics() -> None:
    first = extract_objects([[0, 1, 0], [1, 1, 0]])
    second = extract_objects([[0, 1, 0], [1, 1, 0]])
    assert first == second
    assert len(first) == 1
    assert first[0].area == 3
    assert first[0].color == 1


def test_augmentation_and_program_labels_are_reproducible() -> None:
    assert make_augmentations(seed=3, count=4) == make_augmentations(seed=3, count=4)
    assert program_label_from_verified_program(Program(op="move")) == PROGRAM_OPS.index(
        "translate"
    )
    assert program_label_from_verified_program(Program(op="new_custom_op")) == PROGRAM_OPS.index(
        "unknown"
    )


def test_program_labels_use_evidenced_object_transitions() -> None:
    assert infer_program_label([[1, 0]], [[2, 0]]) == PROGRAM_OPS.index("recolor")
    assert infer_program_label(
        [[1, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 1, 0]],
    ) == PROGRAM_OPS.index("translate")
    assert infer_program_label([[1, 0], [0, 0]], [[1, 2], [3, 4]]) == PROGRAM_OPS.index(
        "unknown"
    )


def test_collation_pads_queries_and_demonstrations() -> None:
    examples = build_world_model_examples([_task()], augmentation_count=1)
    batch = collate_world_model_examples(examples, max_objects=4)
    assert batch.query_inputs.shape == (2, 2, 2)
    assert batch.demo_inputs.shape == (2, 1, 2, 2)
    assert batch.query_object_mask.shape == (2, 4)
    assert np.all(batch.target_mask == 1)


def test_shape_changing_transition_keeps_masks_separate() -> None:
    task = ArcTask(
        task_id="crop-smoke",
        train=[
            ArcPair(input=[[1, 0], [0, 0]], output=[[1]]),
            ArcPair(input=[[2, 0], [0, 0]], output=[[2]]),
        ],
        test=[ArcPair(input=[[1, 0], [0, 0]])],
    )
    examples = build_world_model_examples([task], augmentation_count=1)
    batch = collate_world_model_examples(examples, max_objects=4)
    assert batch.query_inputs.shape == (2, 2, 2)
    assert batch.targets.shape == (2, 2, 2)
    assert np.all(batch.target_mask[:, 0, 0] == 1)
    assert np.all(batch.target_mask[:, 1, 1] == 0)
    assert batch.demo_input_masks is not None
    assert batch.demo_output_masks is not None
    assert np.all(batch.demo_input_masks[:, :, 1, 1] == 1)
    assert np.all(batch.demo_output_masks[:, :, 1, 1] == 0)
    model = HybridWorldModel(
        WorldModelConfig(max_grid_size=4, hidden_size=8, num_slots=2, recurrent_steps=2)
    )
    assert model.forward_batch(batch).grid_logits.shape == (2, 2, 2, 10)


def test_mlx_probe_distinguishes_import_from_device_runtime() -> None:
    assert isinstance(mlx_package_available(), bool)
    assert isinstance(mlx_runtime_available(), bool)


def test_mlx_constructor_fails_clearly_when_device_is_unusable() -> None:
    if mlx_package_available() and not mlx_runtime_available():
        from arc_agent.world_model import MlxHybridWorldModel

        with pytest.raises(RuntimeError, match="current device"):
            MlxHybridWorldModel(WorldModelConfig(hidden_size=4, num_slots=1))


def test_numpy_world_model_forward_predict_and_checkpoint(tmp_path) -> None:
    examples = build_world_model_examples([_task()], augmentation_count=1)
    model = HybridWorldModel(
        WorldModelConfig(max_grid_size=4, hidden_size=8, num_slots=2, recurrent_steps=2, seed=11)
    )
    batch = collate_world_model_examples(examples, max_objects=2)
    output = model.forward_batch(batch)
    assert output.grid_logits.shape == (2, 2, 2, 10)
    assert output.program_logits.shape == (2, len(PROGRAM_OPS))
    prediction = model.predict(
        [[0, 1], [1, 0]], [([[0, 2], [2, 0]], [[0, 1], [1, 0]])]
    )
    assert len(prediction.grid) == 2
    assert len(prediction.program_candidates) == 3

    checkpoint = model.save_checkpoint(tmp_path / "world-model.ckpt")
    restored = HybridWorldModel.load_checkpoint(checkpoint)
    restored_output = restored.forward_batch(batch)
    np.testing.assert_allclose(output.grid_logits, restored_output.grid_logits)
    np.testing.assert_allclose(output.program_logits, restored_output.program_logits)


def test_numpy_smoke_training_and_evaluation_are_finite() -> None:
    examples = build_world_model_examples([_task()], augmentation_count=1)
    model = HybridWorldModel(
        WorldModelConfig(max_grid_size=4, hidden_size=8, num_slots=2, recurrent_steps=2, seed=7)
    )
    history = train_world_model(model, examples, epochs=2, batch_size=2, learning_rate=0.01)
    metrics = evaluate_world_model(model, examples, batch_size=2)
    assert len(history.epochs) == 2
    assert np.isfinite(history.final_loss)
    assert all(np.isfinite(value) for value in metrics.values())
