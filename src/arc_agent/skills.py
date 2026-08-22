from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arc_agent.data import sha256_path
from arc_agent.dsl import program_family, search_programs
from arc_agent.features import task_features
from arc_agent.models import ArcTask, Program


class SkillCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    level: int = Field(ge=1, le=3)
    family: str
    cues: list[str]
    invariants: list[str]
    strategy: list[str]
    failure_modes: list[str]
    example_programs: list[Program] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)

    def markdown(self) -> str:
        lines = [
            f"## {self.name}",
            "",
            f"- Skill ID: `{self.skill_id}`",
            f"- Level: {self.level}",
            f"- Family: `{self.family}`",
            f"- Training examples: {len(self.source_task_ids)}",
            "",
            "### Detection cues",
            "",
            *[f"- {cue}" for cue in self.cues],
            "",
            "### Invariants",
            "",
            *[f"- {invariant}" for invariant in self.invariants],
            "",
            "### Strategy",
            "",
            *[f"{index}. {step}" for index, step in enumerate(self.strategy, start=1)],
            "",
            "### Failure modes",
            "",
            *[f"- {failure}" for failure in self.failure_modes],
        ]
        if self.example_programs:
            lines.extend(
                [
                    "",
                    "### Program templates",
                    "",
                    "```json",
                    json.dumps(
                        [program.model_dump(mode="json") for program in self.example_programs],
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        return "\n".join(lines)


BASE_CARDS = [
    SkillCard(
        skill_id="level2-composition",
        name="Compositional transformation",
        level=2,
        family="composition",
        cues=[
            "No single primitive fits every demonstration",
            "Output preserves several input relations",
        ],
        invariants=[
            "Apply one ordered program to every demonstration",
            "Do not special-case a grid",
        ],
        strategy=[
            "Describe objects and relations before proposing transformations.",
            "Compose the smallest spatial, color, selection, or repetition operations.",
            "Execute the full program on every demonstration and revise from exact diffs.",
        ],
        failure_modes=[
            "Correct operations in the wrong order",
            "A program that fits only one pair",
        ],
    ),
    SkillCard(
        skill_id="level3-symbolic",
        name="Symbolic interpretation",
        level=3,
        family="symbolic-interpretation",
        cues=["Shapes behave as tokens, instructions, or role labels rather than decoration"],
        invariants=["Meaning must remain consistent across demonstrations"],
        strategy=[
            "List plausible semantic roles for each repeated symbol.",
            "Test each role by predicting all demonstration outputs.",
            "Prefer the role assignment with the shortest exact program.",
        ],
        failure_modes=[
            "Treating every symbol as a geometric pattern",
            "Fixating on symmetry alone",
        ],
    ),
    SkillCard(
        skill_id="level3-context",
        name="Contextual rule application",
        level=3,
        family="contextual-rule",
        cues=[
            "The same local pattern changes differently depending on position "
            "or neighboring objects"
        ],
        invariants=["The branch condition must be observable in the input"],
        strategy=[
            "Identify contexts that partition objects into roles.",
            "Write the branch predicate before writing either transformation.",
            "Verify both branches independently, then verify their composition.",
        ],
        failure_modes=["Using a global rule where a conditional rule is required"],
    ),
]


FAMILY_GUIDANCE: dict[str, dict[str, list[str] | str | int]] = {
    "identity": {
        "name": "Identity and preservation",
        "level": 1,
        "cues": ["Output dimensions and cells match the input"],
        "invariants": ["Every cell and coordinate is preserved"],
        "strategy": ["Return a copy of the input grid."],
        "failure": ["Over-transforming an already complete grid"],
    },
    "spatial-transform": {
        "name": "Rigid spatial transformation",
        "level": 1,
        "cues": ["Colors and object shapes are preserved while coordinates change"],
        "invariants": ["Color counts and object topology remain constant"],
        "strategy": ["Test rotations, reflections, and transpose; verify exactly."],
        "failure": ["Confusing horizontal and vertical reflection"],
    },
    "color-mapping": {
        "name": "Consistent color mapping",
        "level": 1,
        "cues": ["Input and output shapes match and each source color has one target color"],
        "invariants": ["The mapping is consistent across every cell and pair"],
        "strategy": ["Infer a global source-to-target color map and preserve unmapped colors."],
        "failure": ["Using position-dependent recoloring without evidence"],
    },
    "object-selection": {
        "name": "Object extraction and selection",
        "level": 2,
        "cues": ["Output is a crop or one connected component from a larger scene"],
        "invariants": ["Selected object cells preserve their relative coordinates"],
        "strategy": ["Segment components, compare roles, select by a stable property, then crop."],
        "failure": ["Assuming the largest object is always selected"],
    },
    "size-and-repetition": {
        "name": "Scaling, tiling, and framing",
        "level": 2,
        "cues": ["Output dimensions are a simple multiple or padded form of the input"],
        "invariants": ["Repeated copies preserve cell order"],
        "strategy": ["Compare dimension ratios, distinguish pixel scaling from whole-grid tiling."],
        "failure": ["Confusing scale with tile"],
    },
    "logical-composition": {
        "name": "Logical subgrid composition",
        "level": 2,
        "cues": ["A uniform divider separates two equally sized pattern grids"],
        "invariants": ["Output occupancy follows the same Boolean relation at every cell"],
        "strategy": [
            "Split at the divider and test union, intersection, and exclusive-or occupancy."
        ],
        "failure": ["Using colour equality when foreground occupancy is the intended signal"],
    },
    "relational-placement": {
        "name": "Relational object placement",
        "level": 2,
        "cues": ["One object moves while another remains as a spatial anchor"],
        "invariants": ["Object shape is preserved and the final relation is consistent"],
        "strategy": [
            "Identify the moving and anchor objects, then infer contact, alignment, or spacing."
        ],
        "failure": ["Using a fixed offset when the displacement depends on the anchor"],
    },
    "symbolic-composition": {
        "name": "Symbol-derived mosaic composition",
        "level": 3,
        "cues": [
            "A small cropped symbol controls both a periodic background and a centered overlay"
        ],
        "invariants": [
            "The same symbol mask is permuted for the base tile and repeated for the overlay"
        ],
        "strategy": [
            "Crop the symbol, test row and column permutations of its complement, tile the base, "
            "then overlay repetitions of the original mask in the marker color."
        ],
        "failure": ["Treating the output as independent fixed-size blocks"],
    },
    "pattern-repair": {
        "name": "Periodic panel error repair",
        "level": 3,
        "cues": ["Bordered panels contain nearly periodic patterns with sparse outlier cells"],
        "invariants": ["Panel geometry and all non-outlier cells are preserved"],
        "strategy": [
            "Split at uniform separators, trim borders, infer the shortest low-error 2D period, "
            "and replace only cells inconsistent with the majority tile."
        ],
        "failure": [
            "Choosing a long period that memorizes corruptions instead of compressing them"
        ],
    },
    "panel-selection": {
        "name": "Unique panel selection by row band",
        "level": 2,
        "cues": ["A separator grid contains repeated panels and one odd panel in each row band"],
        "invariants": ["The selected panel is copied without changing its internal geometry"],
        "strategy": [
            "Split at full separator lines, group panels within each row band, and stack the "
            "single panel whose contents occur once."
        ],
        "failure": ["Selecting one global panel when the odd panel changes by row band"],
    },
}


def learn_skills(
    tasks: Iterable[ArcTask], output_dir: str | Path, *, source_path: Path
) -> list[SkillCard]:
    task_list = list(tasks)
    examples: dict[str, list[tuple[str, Program]]] = defaultdict(list)
    unsolved: list[str] = []
    for task in task_list:
        candidates = search_programs(task, limit=1)
        if not candidates or candidates[0].program is None:
            unsolved.append(task.task_id)
            continue
        family = program_family(candidates[0].program)
        examples[family].append((task.task_id, candidates[0].program))

    cards: list[SkillCard] = []
    for family, family_examples in sorted(examples.items()):
        guidance = FAMILY_GUIDANCE[family]
        unique_programs: dict[str, Program] = {}
        for _, program in family_examples:
            key = json.dumps(program.model_dump(mode="json"), sort_keys=True)
            unique_programs.setdefault(key, program)
        cards.append(
            SkillCard(
                skill_id=f"learned-{family}",
                name=str(guidance["name"]),
                level=int(guidance["level"]),
                family=family,
                cues=list(guidance["cues"]),
                invariants=list(guidance["invariants"]),
                strategy=list(guidance["strategy"]),
                failure_modes=list(guidance["failure"]),
                example_programs=list(unique_programs.values())[:8],
                source_task_ids=[task_id for task_id, _ in family_examples],
            )
        )
    cards.extend(card.model_copy(deep=True) for card in BASE_CARDS)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for level in (1, 2, 3):
        content = [f"# ARC-AGI-2 Level {level} Skills", ""]
        content.extend(card.markdown() for card in cards if card.level == level)
        (destination / f"level_{level}.md").write_text("\n\n".join(content).rstrip() + "\n")
    (destination / "cards.json").write_text(
        json.dumps([card.model_dump(mode="json") for card in cards], indent=2, sort_keys=True)
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source_path),
        "source_sha256": sha256_path(source_path),
        "task_count": len(task_list),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(sorted(task.task_id for task in task_list)).encode()
        ).hexdigest(),
        "solved_by_library": len(task_list) - len(unsolved),
        "unsolved_task_ids": unsolved,
        "family_counts": Counter(
            program_family(candidate.program)
            for task in task_list
            for candidate in search_programs(task, limit=1)
            if candidate.program is not None
        ),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return cards


def load_cards(path: str | Path) -> list[SkillCard]:
    source = Path(path)
    cards_path = source / "cards.json" if source.is_dir() else source
    return [SkillCard.model_validate(card) for card in json.loads(cards_path.read_text())]


def retrieve_cards(
    cards: Iterable[SkillCard], task: ArcTask, *, level: int, limit: int = 4
) -> list[SkillCard]:
    features = task_features(task)
    preferred: set[str] = set()
    if features["dimension_change_ratio"] > 0:
        preferred.update({"size-and-repetition", "object-selection"})
    if features["mean_components"] > 2:
        preferred.update({"object-selection", "contextual-rule"})
    if features["mean_input_colors"] > 4:
        preferred.update({"color-mapping", "symbolic-interpretation"})
    ranked = sorted(
        cards,
        key=lambda card: (
            card.level > level,
            card.family not in preferred,
            abs(card.level - level),
            card.skill_id,
        ),
    )
    return ranked[:limit]


def assert_no_eval_provenance(skills_dir: str | Path, evaluation_tasks: Iterable[ArcTask]) -> None:
    eval_ids = {task.task_id for task in evaluation_tasks}
    leaked = {
        task_id
        for card in load_cards(skills_dir)
        for task_id in card.source_task_ids
        if task_id in eval_ids
    }
    if leaked:
        raise ValueError(f"evaluation leakage detected in skill provenance: {sorted(leaked)}")
