"""Native Nanbeige rendering and a raw, loopback-only llama.cpp transport."""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import httpx
from jinja2.sandbox import ImmutableSandboxedEnvironment

from arc_agent.models import ArcTask
from arc_agent.v4_config import V4Config
from arc_agent.v4_grammar import final_grammar
from arc_agent.v4_tools import task_grids


class ContextOverflow(ValueError):
    pass


def separate_reasoning(raw: str) -> tuple[str, str]:
    # Generation starts after the native template's opening <think>.
    if "</think>" not in raw:
        return raw.removeprefix("<think>").strip(), ""
    reasoning, final = raw.split("</think>", 1)
    return reasoning.removeprefix("<think>").strip(), final.replace("<|im_end|>", "").strip()


def without_reasoning(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if "</think>" in content:
                content = content.split("</think>", 1)[1]
            elif "<think>" in content:
                content = content.split("<think>", 1)[0]
            message["content"] = content
            message["reasoning_content"] = ""
            message.pop("reasoning", None)
    return result


class NativeTemplate:
    def __init__(self, tokenizer_config: Path):
        config = json.loads(tokenizer_config.read_text())
        self.source = config["chat_template"]
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs
        )

        def raise_exception(message):
            raise ValueError(message)

        environment.globals["raise_exception"] = raise_exception
        self.template = environment.from_string(self.source)

    def render(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        retain: bool = True,
        thinking: bool = True,
    ) -> str:
        # Explicit filtering is necessary: the upstream flag retains same-query tool reasoning.
        history = messages if retain else without_reasoning(messages)
        return self.template.render(
            messages=history,
            tools=tools or [],
            preserve_thinking=True,
            enable_thinking=thinking,
            tool_call_format="xml",
            add_generation_prompt=True,
        )


def final_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.DOTALL)
        if not match:
            raise ValueError("malformed JSON fence")
        value = match.group(1)
    parsed = json.loads(
        value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number"))
    )
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def parse_tool_calls(text: str) -> list[dict]:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not blocks or len(blocks) > 4:
        raise ValueError("expected one to four complete tool calls")
    remainder = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    if remainder:
        raise ValueError("unexpected text outside tool calls")
    result = []
    for block in blocks:
        if block.startswith("{"):
            call = final_json(block)
            if set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict):
                raise ValueError("invalid tool call")
            result.append(call)
            continue
        match = re.fullmatch(r"<function=([a-z_]+)>\s*(.*?)\s*</function>", block, re.DOTALL)
        if not match:
            raise ValueError("invalid native tool call")
        name, body = match.groups()
        args = {}
        for parameter, value in re.findall(
            r"<parameter=([a-z_]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL
        ):
            if parameter in args:
                raise ValueError("duplicate tool parameter")
            try:
                args[parameter] = json.loads(value)
            except json.JSONDecodeError:
                args[parameter] = value
        if re.sub(r"<parameter=[a-z_]+>.*?</parameter>", "", body, flags=re.DOTALL).strip():
            raise ValueError("malformed tool parameters")
        result.append({"name": name, "arguments": args})
    return result


class LocalReasoner:
    def __init__(self, config: V4Config, manifest: dict):
        self.config = config
        self.template = NativeTemplate(Path(manifest["files"]["tokenizer_config"]["path"]))
        self.client = httpx.Client(
            base_url=config.base_url, trust_env=False, follow_redirects=False, timeout=10
        )

    def prepare(self, messages: list[dict], tools: list[dict], *, seed: int, task: ArcTask) -> dict:
        prompt = self.template.render(messages, tools=tools, retain=self.config.history == "retain")
        response = self.client.post("/tokenize", json={"content": prompt, "add_special": False})
        response.raise_for_status()
        tokens = response.json()["tokens"]
        available = self.config.context_tokens - len(tokens) - 8
        if available < 128:
            raise ContextOverflow(f"prompt has {len(tokens)} tokens; no silent truncation allowed")
        grid_ids = list(task_grids(task))
        grammar = final_grammar("program" if tools else "direct", len(task.test), grid_ids)
        generated = min(available, self.config.max_new_tokens)
        return {
            "prompt": tokens,
            "n_predict": generated,
            "reasoning_budget_tokens": min(self.config.max_reasoning_tokens, generated // 2),
            "reasoning_budget_start_tag": "<think>",
            "reasoning_budget_end_tags": ["</think>"],
            "reasoning_budget_message": "",
            "generation_prompt": "<think>\n",
            "grammar": grammar,
            "grammar_lazy": True,
            "preserved_tokens": ["</think>"],
            "grammar_triggers": [{"type": 1, "value": "</think>"}],
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "seed": seed,
            "stream": False,
            "cache_prompt": False,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "id_slot": 0,
            "_rendered_prompt": prompt,
            "_prompt_tokens": len(tokens),
        }

    def complete(self, prepared: dict, *, timeout: float) -> dict:
        start = time.monotonic()
        response = self.client.post(
            "/completion",
            json={k: v for k, v in prepared.items() if not k.startswith("_")},
            timeout=max(0.05, timeout),
        )
        response.raise_for_status()
        raw = response.json()
        if raw.get("truncated"):
            raise ContextOverflow("server truncated a prompt despite token preflight")
        reasoning, final = separate_reasoning(raw["content"])
        return {
            "raw": raw,
            "reasoning": reasoning,
            "final": final,
            "prompt_tokens": prepared["_prompt_tokens"],
            "generated_tokens": raw.get("tokens_predicted", 0),
            "elapsed_seconds": time.monotonic() - start,
        }

    def close(self):
        self.client.close()
