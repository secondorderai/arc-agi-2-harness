"""Expand context without dropping history, replaying calls, or resetting task time."""

from __future__ import annotations

import json
import time
from pathlib import Path

from arc_agent.v4_config import V4Config
from arc_agent.v4_reasoner import ContextOverflow, LocalReasoner
from arc_agent.v4_runtime import LocalServer
from arc_agent.v4_state import atomic_json

CONTEXT_STEP = 4096


def next_context(current: int, ceiling: int) -> int:
    if current >= ceiling:
        raise ContextOverflow(f"verified model context ceiling reached: {ceiling} tokens")
    return min(current + CONTEXT_STEP, ceiling)


class AdaptiveSession:
    def __init__(self, config, manifest, workspace, *, evidence_fn, ceiling=262144):
        self.base_config, self.manifest, self.workspace = config, manifest, workspace
        model = json.loads(Path(manifest["files"]["model_config"]["path"]).read_text())
        self.ceiling = min(ceiling, model["max_position_embeddings"], 262144)
        if not 8192 <= config.context_tokens <= self.ceiling:
            raise ValueError("adaptive local run must start at least at 8K within model metadata")
        self.path = workspace / "adaptive-state.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "version": 1,
                "context_tokens": config.context_tokens,
                "ceiling": self.ceiling,
                "step": CONTEXT_STEP,
                "events": [],
                "segments": [],
                "swap_baseline_bytes": None,
            }
        )
        if self.state["ceiling"] != self.ceiling or not (
            config.context_tokens <= self.state["context_tokens"] <= self.ceiling
        ):
            raise ValueError("adaptive context checkpoint policy changed")
        self.evidence_fn = evidence_fn
        self.server = self.reasoner = None
        self.deadline = None
        self.task_key = None
        self.failure = None

    def _save(self):
        atomic_json(self.path, self.state)

    def __enter__(self):
        try:
            self._start(self.state["context_tokens"])
            return self
        except BaseException:
            self.close()
            raise

    def set_deadline(self, deadline):
        self.deadline = deadline

    def _remaining(self):
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def _start(self, context):
        if self._remaining() <= 0:
            raise TimeoutError("task deadline reached before context expansion")
        cfg = V4Config.model_validate({**self.base_config.model_dump(), "context_tokens": context})
        segment = (
            self.workspace
            / "runtime-segments"
            / (f"{len(self.state['segments']):03d}-context-{context}")
        )
        # Each attempt has its own evidence; no model overlaps the previous segment.
        while segment.exists():
            # Keep an interrupted segment as unverified evidence, never overwrite its
            # log or infer a safe exit. The process lock still prevents duplicate models.
            self.state["segments"].append(
                {
                    "context_tokens": context,
                    "path": str(segment),
                    "offline_enforced": False,
                    "metal_verified": False,
                    "memory_safe": False,
                    "guard_error": "unfinalized segment after interruption; safety unverified",
                    "sample_count": 0,
                }
            )
            self._save()
            segment = (
                self.workspace
                / "runtime-segments"
                / (f"{len(self.state['segments']):03d}-context-{context}")
            )
        self.server = LocalServer(
            cfg,
            self.manifest,
            segment,
            deadline=self.deadline,
            swap_baseline_bytes=self.state["swap_baseline_bytes"],
        )
        self.segment_path, self.segment_context = segment, context
        try:
            self.server.__enter__()
            self.reasoner = LocalReasoner(cfg, self.manifest)
            if self.state["swap_baseline_bytes"] is None:
                self.state["swap_baseline_bytes"] = self.server.baseline["swap_used_bytes"]
            self.state["context_tokens"] = context
            self._save()
        except BaseException as error:
            self.failure = str(error)
            self.close()
            raise

    def check(self):
        if self.failure:
            raise RuntimeError("adaptive runtime stopped: " + self.failure)
        if self.server is None:
            raise RuntimeError("adaptive runtime has no active server")
        self.server.check()

    def prepare(self, messages, tools, *, seed, task):
        self.task_key = task.task_id + (":program" if tools else ":direct")
        while True:
            self.check()
            if self._remaining() <= 0:
                raise TimeoutError("task deadline reached during prompt preparation")
            try:
                prepared = self.reasoner.prepare(messages, tools, seed=seed, task=task)
                if prepared["n_predict"] < self.base_config.max_new_tokens:
                    raise ContextOverflow(
                        f"prompt has {prepared['_prompt_tokens']} tokens; "
                        "full per-call output allowance does not fit"
                    )
                prepared["_context_tokens"] = self.state["context_tokens"]
                return prepared
            except ContextOverflow as error:
                context = next_context(self.state["context_tokens"], self.ceiling)
                # Avoid beginning another model load at the end of a task's allowance.
                if self._remaining() < 15:
                    raise TimeoutError(
                        "insufficient task time remaining for context expansion"
                    ) from error
                event = {
                    "time": time.time(),
                    "task": self.task_key,
                    "reason": str(error),
                    "from_context": self.state["context_tokens"],
                    "to_context": context,
                    "status": "requested",
                }
                self.state["events"].append(event)
                self._save()
                print(json.dumps({"context_expansion": event}), flush=True)
                # History and raw artifacts stay unchanged. Reload/tokenization time is
                # inside solve_one's existing task timer, including subsequent retries.
                self.close()
                try:
                    self._start(context)
                except BaseException:
                    event["status"] = "failed"
                    self._save()
                    raise
                event["status"] = "ready"
                self._save()

    def complete(self, prepared, *, timeout):
        self.check()
        remaining = min(timeout, self._remaining())
        if remaining <= 0:
            raise TimeoutError("task deadline reached before generation")
        response = self.reasoner.complete(prepared, timeout=remaining)
        response["context_tokens"] = prepared["_context_tokens"]
        return response

    def close(self):
        if self.reasoner:
            self.reasoner.close()
            self.reasoner = None
        if self.server:
            self.server.__exit__(None, None, None)
            evidence = self.evidence_fn(self.server, self.segment_path)
            self.state["segments"].append(
                {
                    "context_tokens": self.segment_context,
                    "path": str(self.segment_path),
                    **evidence,
                }
            )
            self.server = None
            self._save()

    def evidence(self):
        segments = list(self.state["segments"])
        if self.server:
            segments.append(self.evidence_fn(self.server, self.segment_path))
        evidence = {
            key: bool(segments) and all(s.get(key) for s in segments)
            for key in ("offline_enforced", "metal_verified", "memory_safe")
        }
        for key in ("peak_server_rss_bytes", "swap_growth_bytes", "peak_pressure_level"):
            evidence[key] = max((s.get(key, 0) for s in segments), default=0)
        evidence.update(
            {
                "sample_count": sum(s.get("sample_count", 0) for s in segments),
                "warning_pressure_policy": self.base_config.warning_pressure_policy,
                "prompt_cache_mib": self.base_config.prompt_cache_mib,
                "prompt_cache_disabled_verified": bool(segments)
                and all(s.get("prompt_cache_disabled_verified") for s in segments),
                "warning_pressure_samples": sum(
                    s.get("warning_pressure_samples", 0) for s in segments
                ),
                "guard_error": [s["guard_error"] for s in segments if s.get("guard_error")],
                "context_tokens": self.state["context_tokens"],
                "context_policy": (
                    f"{self.base_config.context_tokens} token start; "
                    "+4K on insufficient answer space; no compaction"
                ),
                "context_expansions": self.state["events"],
                "initial_swap_used_bytes": self.state["swap_baseline_bytes"],
            }
        )
        return evidence

    def __exit__(self, *_):
        self.close()
