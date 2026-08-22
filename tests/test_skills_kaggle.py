from __future__ import annotations

import json

import pytest

from arc_agent.kaggle_runner import find_challenges
from arc_agent.models import ArcPair, ArcTask
from arc_agent.skills import assert_no_eval_provenance, learn_skills, load_cards


def test_skill_generation_and_leakage_check(identity_task, tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "identity.json").write_text(
        json.dumps(
            {
                "train": [pair.model_dump(mode="json") for pair in identity_task.train],
                "test": [pair.model_dump(mode="json") for pair in identity_task.test],
            }
        )
    )
    output = tmp_path / "skills"
    learn_skills([identity_task], output, source_path=training)
    cards = load_cards(output)
    assert any(card.source_task_ids == ["identity"] for card in cards)
    evaluation = [
        ArcTask(
            task_id="identity",
            train=[ArcPair(input=[[1]], output=[[1]])],
            test=[ArcPair(input=[[2]], output=[[2]])],
        )
    ]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_eval_provenance(output, evaluation)


def test_kaggle_challenge_discovery(tmp_path):
    nested = tmp_path / "competition"
    nested.mkdir()
    challenge = nested / "arc-agi_test-challenges.json"
    challenge.write_text("{}")
    assert find_challenges(tmp_path) == challenge
