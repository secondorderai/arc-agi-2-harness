from __future__ import annotations

import json

from arc_agent.models import ArcTask
from arc_agent.v3_dsl import operator_catalog
from arc_agent.v3_models import ProposedSignature, SignatureVerification


def _task_payload(task: ArcTask) -> dict[str, object]:
    return {
        "train": [
            {"input": pair.input, "output": pair.output}
            for pair in task.train
        ],
        "test_inputs": [pair.input for pair in task.test],
    }


def build_signature_prompt(
    task: ArcTask,
    *,
    family: str,
    round_index: int,
    previous: ProposedSignature | None,
    verification: SignatureVerification | None,
    independent: bool,
    max_attempts: int = 12,
) -> str:
    sections = [
        "Synthesize one compact executable JSON game signature for this ARC task.",
        "The pipeline executes from top to bottom. A step argument may reference a declared "
        'parameter with the string "$parameter_name". Use semantic parameter sources whenever '
        "possible so colors, dimensions, offsets, and counts adapt to the supplied grid and "
        "demonstrations.",
        "Only these DSL operations and argument names are available:\n"
        + json.dumps(operator_catalog(), sort_keys=True, separators=(",", ":")),
        "Allowed semantic parameter sources: constant, background, rarest_input_color, "
        "most_frequent_input_color, output_only_color, most_frequent_output_color, "
        "ranked_object_color, "
        "input_height, input_width, output_height, output_width, row_scale, column_scale, "
        "inferred_translation_row, inferred_translation_column, inferred_translation, "
        "demonstrated_object_rank, train_pair_count.",
        f"Target transformation family: {family}.",
        "Task data (test inputs are deliberately unlabelled):\n"
        + json.dumps(_task_payload(task), sort_keys=True, separators=(",", ":")),
        f"Refinement attempt: {round_index + 1} of {max_attempts}.",
    ]
    if independent:
        sections.append(
            "This is an independent resynthesis after a repeated failure signature. Do not "
            "reconstruct or imitate the previous configuration; derive a different hypothesis."
        )
    elif previous is not None and verification is not None:
        sections.append(
            "Previous signature:\n"
            + previous.model_dump_json(indent=2)
            + "\nVerifier result:\n"
            + verification.model_dump_json(indent=2)
            + "\nRepair the underlying rule rather than special-casing a failing grid."
        )
    sections.append(
        "Do not include task IDs, grids as literal values, Python, serialized outputs, or "
        "complete-grid equality conditions. Return only the schema-constrained JSON object."
    )
    return "\n\n".join(sections)
