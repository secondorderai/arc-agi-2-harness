from collections import Counter

import pytest

from arc_agent.models import ArcPair, ArcTask
from arc_agent.ttt import (
    ASSISTANT,
    EOS,
    NEWLINE,
    USER,
    Augmentation,
    assistant_labels,
    build_sequence,
    grid_grammar_allowed_tokens,
    infer_output_shape,
    inverse_transform_grid,
    parse_grid_text,
    rank_candidate_keys,
    task_augmentations,
    training_converged,
    transform_grid,
)


def test_geometric_and_color_augmentations_round_trip() -> None:
    grid = [[0, 1, 2], [3, 4, 5]]
    color_map = (0, 2, 1, 4, 3, 5, 6, 7, 9, 8)
    for geometry in (
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "transpose",
        "anti_transpose",
    ):
        augmentation = Augmentation(geometry=geometry, color_map=color_map)
        assert inverse_transform_grid(transform_grid(grid, augmentation), augmentation) == grid


def test_sequence_masks_only_assistant_output() -> None:
    tokens = build_sequence([([[1, 2]], [[3], [4]])])
    assert tokens == [USER, NEWLINE, 1, 2, EOS, ASSISTANT, NEWLINE, 3, NEWLINE, 4, EOS]
    labels = assistant_labels(tokens)
    assert labels == [-100, -100, -100, -100, -100, -100, -100, 3, NEWLINE, 4, EOS]


def test_parse_grid_text_rejects_non_rectangular_output() -> None:
    assert parse_grid_text("12\n34") == [[1, 2], [3, 4]]
    assert parse_grid_text("12\n3") is None
    assert parse_grid_text("12\n34\n<eos>") == [[1, 2], [3, 4]]
    assert parse_grid_text("12\n34", expected_shape=(1, 4)) is None


def test_grid_grammar_for_fixed_shape() -> None:
    digit_tokens = tuple(range(10))
    assert [grid_grammar_allowed_tokens((2, 3), index) for index in range(8)] == [
        digit_tokens,
        digit_tokens,
        digit_tokens,
        (NEWLINE,),
        digit_tokens,
        digit_tokens,
        digit_tokens,
        (EOS,),
    ]


def test_grid_grammar_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        grid_grammar_allowed_tokens((0, 2), 0)
    with pytest.raises(ValueError):
        grid_grammar_allowed_tokens((2, 2), -1)


def test_training_convergence_requires_stable_low_loss() -> None:
    assert training_converged(
        [0.004, 0.008, 0.01],
        completed_epochs=2,
        min_epochs=2,
        median_threshold=0.01,
        max_threshold=0.05,
    )
    assert not training_converged(
        [0.004, 0.008, 0.08],
        completed_epochs=2,
        min_epochs=2,
        median_threshold=0.01,
        max_threshold=0.05,
    )
    assert not training_converged(
        [0.001],
        completed_epochs=1,
        min_epochs=2,
        median_threshold=0.01,
        max_threshold=0.05,
    )


def test_candidate_ranking_uses_augmented_nll_then_votes() -> None:
    votes = Counter({"popular": 3, "consistent": 1, "weak": 1})
    generation_nlls = {"popular": [2.0], "consistent": [4.0], "weak": [3.0]}
    assert rank_candidate_keys(votes, generation_nlls) == [
        "popular",
        "weak",
        "consistent",
    ]
    assert rank_candidate_keys(
        votes,
        generation_nlls,
        {"popular": 9.0, "consistent": 3.0, "weak": 5.0},
    ) == ["consistent", "weak", "popular"]


def test_task_augmentations_are_deterministic_and_keep_zero_fixed() -> None:
    task = ArcTask(
        task_id="example",
        train=[ArcPair(input=[[0]], output=[[1]])],
        test=[ArcPair(input=[[2]])],
    )
    first = task_augmentations(task.task_id, seed=42, count=10)
    second = task_augmentations(task.task_id, seed=42, count=10)
    assert first == second
    assert len(first) == 10
    assert all(augmentation.color_map[0] == 0 for augmentation in first)


def test_infer_output_shape_handles_fixed_and_same_size_outputs() -> None:
    fixed = ArcTask(
        task_id="fixed",
        train=[
            ArcPair(input=[[0]], output=[[1, 1]]),
            ArcPair(input=[[0, 0]], output=[[2, 2]]),
        ],
        test=[ArcPair(input=[[3, 3, 3]])],
    )
    assert infer_output_shape(fixed, 0) == (1, 2)

    same = ArcTask(
        task_id="same",
        train=[
            ArcPair(input=[[0]], output=[[1]]),
            ArcPair(input=[[0, 0], [0, 0]], output=[[1, 1], [1, 1]]),
        ],
        test=[ArcPair(input=[[3, 3, 3], [3, 3, 3]])],
    )
    assert infer_output_shape(same, 0) == (2, 3)
