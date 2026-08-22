from __future__ import annotations

import ast
import base64
import io
import json
import re
import time
from collections import Counter
from typing import Any, Protocol

import httpx
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from arc_agent.dsl import connected_components
from arc_agent.models import ArcTask, Grid, Program, Usage
from arc_agent.skills import SkillCard

MAX_PYTHON_SOURCE_CHARS = 1_800
ARC_PALETTE = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)


class GeneratedCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypothesis: str = ""
    program: Program | None = None
    python_source: str | None = None
    predictions: list[Grid] = Field(default_factory=list)
    source: str = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Generation(BaseModel):
    candidates: list[GeneratedCandidate] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw_text: str = ""
    reasoning_text: str = ""


class ModelAdapter(Protocol):
    def generate(
        self,
        task: ArcTask,
        skills: list[SkillCard],
        *,
        level: int,
        feedback: list[str] | None = None,
        search_mode: str | None = None,
    ) -> Generation: ...


DSL_REFERENCE = """Supported DSL operations:
- {"op":"identity"}
- {"op":"rotate","args":{"turns":1|2|3}}
- {"op":"flip","args":{"axis":"horizontal"|"vertical"}}
- {"op":"transpose"}
- {"op":"recolor","args":{"mapping":{"source":"target"}}}
- {"op":"crop_background","args":{"background":0}}
- {"op":"select_component","args":{"background":0,
  "criterion":"largest"|"smallest"|"widest"|"tallest",
  "connectivity":4|8,"multicolor":false}}
- {"op":"scale","args":{"rows":2,"columns":2}}
- {"op":"tile","args":{"rows":2,"columns":2}}
- {"op":"pad","args":{"top":1,"bottom":1,"left":1,"right":1,"color":0}}
- {"op":"translate","args":{"rows":1,"columns":-2,"background":0}}
- {"op":"concat","args":{"axis":"horizontal"|"vertical",
  "transform":"identity"|"rotate_90"|"rotate_180"|"rotate_270"|
  "flip_horizontal"|"flip_vertical"|"transpose",
  "order":"input_first"|"transformed_first"}}
- {"op":"complete_symmetry","args":{"axis":"horizontal"|"vertical"|"rotational","background":0}}
- {"op":"split_overlay","args":{"axis":"horizontal"|"vertical",
  "mode":"union"|"intersection"|"xor","background":0,
  "separator_color":5,"output_color":2}}
- {"op":"move_color_to_contact","args":{"moving_color":2,
  "target_color":8,"background":0}}
- {"op":"pattern_mosaic","args":{"background":7,"empty_color":0,
  "base_color":7,"marker_color":9,"row_order":[1,0,2],
  "column_order":[1,0,2],"output_rows":16,"output_columns":16}}
- {"op":"repair_periodic_panels"}
- {"op":"select_unique_panel_per_band"}
- {"op":"deduplicate_repeated_half"}
- {"op":"fold_quadrants_overlap"}
- {"op":"mark_most_frequent_above_separator"}
- {"op":"keep_most_frequent_colors","args":{"keep_count":2,"output_color":7}}
- {"op":"pack_points_serpentine"}
- {"op":"fill_between_row_markers","args":{"fill_color":2}}
- {"op":"expand_seed_to_width"}
- {"op":"summarize_nonbackground_count"}
- {"op":"fill_with_most_frequent_color"}
- {"op":"frequency_histogram_columns"}
- {"op":"recolor_singleton_components"}
- {"op":"move_center_block_to_corners"}
- {"op":"compose","steps":[PROGRAM, PROGRAM]}
"""


def serialize_task(task: ArcTask) -> str:
    payload = {
        "task_id": task.task_id,
        "train": [pair.model_dump(mode="json") for pair in task.train],
        "test": [{"input": pair.input} for pair in task.test],
    }
    return json.dumps(payload, separators=(",", ":"))


def _grid_structure(grid: Grid) -> str:
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    histogram = Counter(value for row in grid for value in row)
    components = connected_components(grid, background=background)
    component_text: list[str] = []
    for component in sorted(components, key=lambda item: (min(item), len(item)))[:12]:
        rows, columns = zip(*component, strict=True)
        color = grid[component[0][0]][component[0][1]]
        component_text.append(
            f"{color}:{len(component)}@{min(rows)},{min(columns)}-{max(rows)},{max(columns)}"
        )
    histogram_text = ",".join(f"{color}:{count}" for color, count in sorted(histogram.items()))
    return (
        f"{len(grid)}x{len(grid[0])} bg={background} hist={histogram_text} "
        f"components=[{';'.join(component_text)}]"
    )


def _pair_delta(input_grid: Grid, output_grid: Grid) -> str:
    if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
        input_shape = f"{len(input_grid)}x{len(input_grid[0])}"
        output_shape = f"{len(output_grid)}x{len(output_grid[0])}"
        return f"resize {input_shape}->{output_shape}"
    changes = [
        (row, column, input_grid[row][column], output_grid[row][column])
        for row in range(len(input_grid))
        for column in range(len(input_grid[0]))
        if input_grid[row][column] != output_grid[row][column]
    ]
    if not changes:
        return "unchanged"
    transitions = Counter((source, target) for _, _, source, target in changes)
    transition_text = ",".join(
        f"{source}>{target}:{count}"
        for (source, target), count in sorted(transitions.items())
    )
    rows = [row for row, _, _, _ in changes]
    columns = [column for _, column, _, _ in changes]
    return (
        f"changed={len(changes)} bbox={min(rows)},{min(columns)}-{max(rows)},{max(columns)} "
        f"transitions={transition_text}"
    )


def structural_summary(task: ArcTask) -> str:
    lines: list[str] = []
    for index, pair in enumerate(task.train, start=1):
        output = pair.output or [[]]
        lines.append(
            f"train{index} input({_grid_structure(pair.input)}) "
            f"output({_grid_structure(output)}) delta({_pair_delta(pair.input, output)})"
        )
    lines.extend(
        f"test{index} input({_grid_structure(pair.input)})"
        for index, pair in enumerate(task.test, start=1)
    )
    return "\n".join(lines)


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    grid: Grid,
    *,
    left: int,
    top: int,
    cell: int,
) -> None:
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            x0 = left + column_index * cell
            y0 = top + row_index * cell
            draw.rectangle(
                (x0, y0, x0 + cell - 1, y0 + cell - 1),
                fill=ARC_PALETTE[value],
                outline="#303030",
            )


def render_task_data_url(task: ArcTask) -> str:
    """Render train relations and test inputs as one compact ARC color collage."""
    items: list[tuple[str, Grid, Grid | None]] = [
        (f"TRAIN {index}", pair.input, pair.output)
        for index, pair in enumerate(task.train, start=1)
    ] + [
        (f"TEST {index}", pair.input, None)
        for index, pair in enumerate(task.test, start=1)
    ]
    columns = 2 if len(items) > 3 else 1
    rows = (len(items) + columns - 1) // columns
    tile_width = 520
    tile_height = min(260, max(150, 940 // max(1, rows)))
    image = Image.new("RGB", (tile_width * columns, tile_height * rows), "#F4F4F4")
    draw = ImageDraw.Draw(image)
    for item_index, (label, input_grid, output_grid) in enumerate(items):
        tile_column = item_index % columns
        tile_row = item_index // columns
        tile_left = tile_column * tile_width
        tile_top = tile_row * tile_height
        draw.text((tile_left + 10, tile_top + 8), label, fill="#111111")
        max_height = max(len(input_grid), len(output_grid or input_grid))
        total_width = len(input_grid[0]) + (len(output_grid[0]) if output_grid else 0)
        cell = max(
            4,
            min(
                16,
                (tile_height - 42) // max(1, max_height),
                (tile_width - 62) // max(1, total_width),
            ),
        )
        grid_top = tile_top + 30
        input_left = tile_left + 10
        _draw_grid(draw, input_grid, left=input_left, top=grid_top, cell=cell)
        arrow_left = input_left + len(input_grid[0]) * cell + 8
        draw.text((arrow_left, grid_top + max_height * cell // 2 - 5), "->", fill="#111111")
        if output_grid is not None:
            output_left = arrow_left + 24
            _draw_grid(draw, output_grid, left=output_left, top=grid_top, cell=cell)
        else:
            draw.text((arrow_left + 28, grid_top + max_height * cell // 2 - 5), "?", fill="#111111")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", optimize=True)
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_prompt(
    task: ArcTask,
    skills: list[SkillCard],
    *,
    level: int,
    feedback: list[str] | None,
    search_mode: str | None = None,
    structural_summary_enabled: bool = False,
) -> str:
    skill_text = "\n\n".join(
        "\n".join(
            [
                f"## {card.name} ({card.family})",
                *[f"Cue: {cue}" for cue in card.cues[:2]],
                *[f"Step: {step}" for step in card.strategy[:3]],
                *[f"Avoid: {failure}" for failure in card.failure_modes[:1]],
            ]
        )
        for card in skills
    )
    feedback_text = "\n".join(f"- {item}" for item in feedback or []) or "- No prior attempt."
    if level >= 3:
        method_instructions = (
            "The deterministic DSL search has already failed. Do not emit JSON or a DSL program. "
            "Infer the rule and implement it as pure Python. The API enforces a structured "
            "response whose only field is python_source. Put executable code there immediately "
            "instead of explaining your analysis."
        )
        dsl_reference = ""
        solution_contract = f"""Return only python_source. It must begin with def solve(grid):.
It must be at most {MAX_PYTHON_SOURCE_CHARS:,} characters and end with an explicit return.
Every line must be executable: no comments, docstrings, hypotheses, matrix restatements, or prose.
Spend output tokens on executable code. Prefer short loops and comprehensions; omit defensive
checks not required by the examples."""
    else:
        method_instructions = (
            "Use the strict whitelist below when it expresses the entire rule. "
            "Otherwise emit pure Python."
        )
        dsl_reference = DSL_REFERENCE
        solution_contract = """For a DSL solution, return JSON only with exactly this form:
{"candidates":[{"hypothesis":"short rule", "program":PROGRAM}]}
If the DSL cannot express the rule, return a hypothesis and fenced solve(grid) Python."""
    structure_text = (
        f"\nDETERMINISTIC STRUCTURAL SUMMARY:\n{structural_summary(task)}\n"
        if structural_summary_enabled
        else ""
    )
    return f"""You solve ARC-AGI-2 tasks by proposing executable, general rules.
The integer matrices are authoritative. A candidate is useful only when the same rule exactly
reproduces every training output. Avoid hard-coded output grids and task IDs. When verifier
feedback contains prior code, repair its concrete failures or replace its hypothesis with a
demonstrably better one.

METHOD:
{method_instructions}

{dsl_reference}

TASK:
{serialize_task(task)}
{structure_text}

OUTPUT CONTRACT:
{solution_contract}

Produce exactly one best candidate.
Python may define pure helper functions but must contain solve(grid), with no imports, I/O,
files, network, classes, external state, or hard-coded demonstration outputs. The enforced
source limit is absolute; finish a complete, executable rule before reaching it.

DIFFICULTY LEVEL: {level}

RELEVANT SKILLS:
{skill_text}

VERIFIER FEEDBACK FROM PRIOR ATTEMPTS:
{feedback_text}

INDEPENDENT SEARCH LENS:
{search_mode or "Verifier-guided synthesis using the strongest available evidence."}

Return the single best corrected candidate now.
"""


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Small reasoning models occasionally omit the closing quote on a final
        # python_source string while still closing the candidate array/object.
        if (
            '"python_source"' in stripped
            and stripped.endswith("}]}")
            and len(stripped) >= 4
            and stripped[-4] != '"'
        ):
            return json.loads(f'{stripped[:-3]}"{stripped[-3:]}')
        raise


def _normalize_program_payload(raw: Any) -> Any:
    if isinstance(raw, list):
        return {"op": "compose", "steps": [_normalize_program_payload(step) for step in raw]}
    if not isinstance(raw, dict):
        return raw
    normalized = dict(raw)
    if "op" not in normalized and isinstance(normalized.get("compose"), list):
        normalized = {"op": "compose", "steps": normalized["compose"]}
    if isinstance(normalized.get("steps"), list):
        normalized["steps"] = [_normalize_program_payload(step) for step in normalized["steps"]]
    return normalized


def _decode_json_string_prefix(fragment: str) -> str | None:
    """Decode a possibly truncated JSON string after its opening quote."""
    escaped = False
    end = len(fragment)
    for index, char in enumerate(fragment):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            end = index
            break
    encoded = fragment[:end]
    for trim in range(min(7, len(encoded) + 1)):
        candidate = encoded[: len(encoded) - trim] if trim else encoded
        try:
            return json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            continue
    return None


def _recover_truncated_python(text: str) -> str | None:
    """Recover only a complete solve function from an unfinished JSON envelope."""
    match = re.search(r'"python_source"\s*:\s*"', text)
    if not match:
        return None
    source = _decode_json_string_prefix(text[match.end() :])
    if not source or "def solve(" not in source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    solve_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve"
    ]
    if not solve_functions or not any(
        isinstance(node, ast.Return) for node in ast.walk(solve_functions[0])
    ):
        return None
    return source.strip()


def _is_complete_solve(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    solve_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve"
    ]
    return bool(solve_functions) and any(
        isinstance(node, ast.Return) for node in ast.walk(solve_functions[0])
    )


def _recover_embedded_python(text: str) -> str | None:
    """Extract the longest complete solve function from a prose/error payload."""
    start = text.find("def solve(grid):")
    if start < 0:
        return None
    source = text[start:]
    fence = source.find("```")
    if fence >= 0:
        source = source[:fence]
    lines = source.strip().splitlines()
    while lines:
        candidate = "\n".join(lines).strip()
        if _is_complete_solve(candidate):
            return candidate
        lines.pop()
    return None


def parse_candidates(text: str) -> list[GeneratedCandidate]:
    try:
        parsed = _extract_json(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        recovered = _recover_truncated_python(text)
        if recovered:
            return [GeneratedCandidate(python_source=recovered)]
        fenced = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            prefix = text[: fenced.start()].strip()
            hypothesis = re.sub(r"^HYPOTHESIS:\s*", "", prefix, flags=re.IGNORECASE)
            return [
                GeneratedCandidate(
                    hypothesis=hypothesis[-500:],
                    python_source=fenced.group(1).strip(),
                )
            ]
        recovered = _recover_embedded_python(text)
        if recovered:
            return [GeneratedCandidate(python_source=recovered)]
        raise
    raw_candidates = parsed.get("candidates", [])
    if "python_source" in parsed or "program" in parsed:
        raw_candidates = [parsed]
    candidates: list[GeneratedCandidate] = []
    for raw_item in raw_candidates:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        if "program" in item:
            item["program"] = _normalize_program_payload(item["program"])
        try:
            candidates.append(GeneratedCandidate.model_validate(item))
        except ValueError:
            continue
    if not candidates:
        fenced = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.append(
                GeneratedCandidate(
                    hypothesis=text[: fenced.start()].strip()[-500:],
                    python_source=fenced.group(1).strip(),
                )
            )
    return candidates


class OpenAICompatibleAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "none",
        max_tokens: int = 1024,
        thinking_budget_tokens: int = 512,
        temperature: float = 0.7,
        timeout_seconds: float = 300.0,
        vision_enabled: bool = False,
        structural_summary_enabled: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.vision_enabled = vision_enabled
        self.structural_summary_enabled = structural_summary_enabled

    def generate(
        self,
        task: ArcTask,
        skills: list[SkillCard],
        *,
        level: int,
        feedback: list[str] | None = None,
        search_mode: str | None = None,
    ) -> Generation:
        prompt = build_prompt(
            task,
            skills,
            level=level,
            feedback=feedback,
            search_mode=search_mode,
            structural_summary_enabled=self.structural_summary_enabled,
        )
        started = time.monotonic()
        user_content: str | list[dict[str, Any]] = prompt
        if self.vision_enabled:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": render_task_data_url(task)},
                },
            ]
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a precise program synthesizer."},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": max(0.6, self.temperature) if feedback else self.temperature,
                    "top_p": 0.95,
                    "max_tokens": self.max_tokens,
                    "thinking_budget_tokens": self.thinking_budget_tokens,
                    **(
                        {
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "arc_python_candidate",
                                    "strict": True,
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "python_source": {
                                                "type": "string",
                                                "maxLength": MAX_PYTHON_SOURCE_CHARS,
                                                "pattern": (
                                                    rf"^[^#]{{1,{MAX_PYTHON_SOURCE_CHARS}}}$"
                                                ),
                                            },
                                        },
                                        "required": ["python_source"],
                                        "additionalProperties": False,
                                    },
                                },
                            }
                        }
                        if level >= 3
                        else {}
                    ),
                },
            )
            if response.status_code >= 500:
                try:
                    error_text = str(response.json().get("error", {}).get("message", ""))
                except (TypeError, ValueError):
                    error_text = response.text
                recovered = _recover_embedded_python(error_text)
                if recovered:
                    return Generation(
                        candidates=[GeneratedCandidate(python_source=recovered)],
                        raw_text=error_text,
                        usage=Usage(calls=1, latency_seconds=time.monotonic() - started),
                    )
            response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        raw = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        try:
            candidates = parse_candidates(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            candidates = []
        if reasoning:
            reasoning_summary = reasoning[-1_200:].strip()
            for candidate in candidates:
                if not candidate.hypothesis:
                    candidate.hypothesis = reasoning_summary
        usage = body.get("usage") or {}
        return Generation(
            candidates=candidates,
            raw_text=raw,
            reasoning_text=reasoning,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                calls=1,
                latency_seconds=time.monotonic() - started,
            ),
        )


class StaticAdapter:
    """Deterministic adapter used by tests and notebook dry-runs."""

    def __init__(self, generations: list[Generation]) -> None:
        self.generations = list(generations)

    def generate(
        self,
        task: ArcTask,
        skills: list[SkillCard],
        *,
        level: int,
        feedback: list[str] | None = None,
        search_mode: str | None = None,
    ) -> Generation:
        del task, skills, level, feedback, search_mode
        return self.generations.pop(0) if self.generations else Generation()
