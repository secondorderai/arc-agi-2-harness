"""Bound final-answer syntax, never task solutions or reasoning content."""

from __future__ import annotations

import json

from arc_agent.v4_tools import program_catalog


def literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def choices(values) -> str:
    return "(" + " | ".join(literal(json.dumps(v)) for v in values) + ")"


def final_grammar(mode: str, test_count: int, grid_ids: list[str]) -> str:
    if mode not in {"direct", "program"} or not 1 <= test_count <= 100:
        raise ValueError("invalid structured-output mode or test count")
    rules = [
        r"ws ::= [ \t\n\r]*",
        r"color ::= [0-9]",
        'integer ::= "-"? ("0" | [1-9] | [12] [0-9] | "30")',
        'positive ::= [1-9] | [12] [0-9] | "30"',
        'nonnegative ::= "0" | positive',
        'boolean ::= "true" | "false"',
    ]

    def obj(fields):
        parts = [literal("{") + " ws"]
        for index, (name, rule) in enumerate(fields):
            if index:
                parts.append(literal(",") + " ws")
            parts.extend([literal(json.dumps(name)), "ws", literal(":"), "ws", rule, "ws"])
        parts.append(literal("}"))
        return " ".join(parts)

    if mode == "direct":
        rules += [
            r'row ::= "\"" [0-9]{1,30} "\""',
            'grid ::= "[" ws row (ws "," ws row){0,29} ws "]"',
            "pair ::= "
            + obj(
                [("attempt_1", "grid"), ("attempt_2", "(grid | " + literal('"same_as_1"') + ")")]
            ),
        ]
        array = (
            '"[" ws pair'
            + (' (ws "," ws pair){' + str(test_count - 1) + "}" if test_count > 1 else "")
            + ' ws "]"'
        )
        body = obj([("predictions", array)])
    else:
        # Mapping has at most ten entries; execution validates color keys and values.
        rules += [
            'mapentry ::= "\\"" color "\\"" ws ":" ws color',
            'mapping ::= "{" ws (mapentry (ws "," ws mapentry){0,9})? ws "}"',
        ]
        values = {
            "background": "color",
            "color": "color",
            "mapping": "mapping",
            "rows": "integer",
            "columns": "integer",
            "turns": "integer",
            "top": "nonnegative",
            "bottom": "nonnegative",
            "left": "nonnegative",
            "right": "nonnegative",
            "multicolor": "boolean",
            "connectivity": choices([4, 8]),
            "axis": choices(["horizontal", "vertical"]),
            "criterion": choices(["largest", "smallest", "widest", "tallest"]),
            "transform": choices(
                [
                    "identity",
                    "rotate_90",
                    "rotate_180",
                    "rotate_270",
                    "flip_horizontal",
                    "flip_vertical",
                    "transpose",
                ]
            ),
            "order": choices(["input_first", "transformed_first"]),
        }
        alternatives = []
        for op, names in program_catalog().items():
            if op == "compose":
                continue
            rule = op.replace("_", "-")
            alternatives.append(rule)
            fields = [("op", literal(json.dumps(op)))]
            # Both defaults and a fully specified argument object are legal.
            variants = [obj(fields)]
            if names:
                args = [
                    (
                        name,
                        "positive"
                        if op in {"scale", "tile"} and name in {"rows", "columns"}
                        else values[name],
                    )
                    for name in names
                ]
                variants.append(obj(fields + [("args", obj(args))]))
            rules.append(rule + " ::= " + " | ".join(variants))
        rules.append("leaf ::= " + " | ".join(alternatives))
        # Two composition levels and at most three children: at most thirteen total nodes.
        child = "leaf"
        for depth in range(2):
            name = f"program{depth}"
            array = '"[" ws ' + child + ' (ws "," ws ' + child + '){1,2} ws "]"'
            rules.append(
                name + " ::= leaf | " + obj([("op", literal('"compose"')), ("steps", array)])
            )
            child = name
        grid_choices = "(" + " | ".join(literal(name) for name in grid_ids) + ")"
        run_call = " ws ".join(
            [
                literal("<tool_call>"),
                literal("<function=run_program>"),
                literal("<parameter=program>"),
                child,
                literal("</parameter>"),
                literal("</function>"),
                literal("</tool_call>"),
            ]
        )
        call_variants = [run_call]
        for tool in ("read_grid", "inspect_scene"):
            call_variants.append(
                " ws ".join(
                    [
                        literal("<tool_call>"),
                        literal(f"<function={tool}>"),
                        literal("<parameter=grid_id>"),
                        grid_choices,
                        literal("</parameter>"),
                        literal("</function>"),
                        literal("</tool_call>"),
                    ]
                )
            )
        body = "(" + obj([("program", child)]) + " | " + " | ".join(call_variants) + ")"
    # The lazy trigger is the native end-of-thinking token, which is replayed into the grammar.
    return "root ::= " + literal("</think>") + " ws " + body + " ws\n" + "\n".join(rules) + "\n"
