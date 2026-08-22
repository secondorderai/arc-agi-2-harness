from __future__ import annotations

import pytest

from arc_agent.models import ArcPair, ArcTask


@pytest.fixture
def identity_task() -> ArcTask:
    return ArcTask(
        task_id="identity",
        train=[ArcPair(input=[[1, 0], [0, 1]], output=[[1, 0], [0, 1]])],
        test=[ArcPair(input=[[2, 0], [0, 2]], output=[[2, 0], [0, 2]])],
    )


@pytest.fixture
def hard_task() -> ArcTask:
    return ArcTask(
        task_id="hard",
        train=[
            ArcPair(
                input=[[1, 2, 3], [4, 0, 5], [6, 7, 8]],
                output=[[8, 7], [5, 0]],
            ),
            ArcPair(
                input=[[2, 3, 4], [5, 1, 6], [7, 8, 9]],
                output=[[9, 8], [6, 1]],
            ),
        ],
        test=[ArcPair(input=[[1, 2], [3, 4]])],
    )
