"""Immutable artifacts, frozen data splits and transactional, exclusively-owned runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arc_agent.models import ArcTask, validate_grid
from arc_agent.v4_config import content_hash


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, value: object) -> str:
        key = content_hash(value)
        path = self.root / f"{key}.json"
        if path.exists():
            self.get(key)
        else:
            atomic_json(path, value)
        return key

    def get(self, key: str) -> object:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid artifact reference")
        value = json.loads((self.root / f"{key}.json").read_text())
        if content_hash(value) != key:
            raise ValueError("corrupt artifact")
        return value


class Definition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=1000)
    depends_on: list[str] = Field(default_factory=list, max_length=16)


class GroundedObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact: str
    row: int = Field(ge=0, le=29)
    column: int = Field(ge=0, le=29)
    color: int = Field(ge=0, le=9, strict=True)
    note: str = Field(default="", max_length=500)


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=2000)
    status: Literal["possible", "supported", "refuted"] = "possible"
    evidence: list[str] = Field(default_factory=list, max_length=32)
    counterexamples: list[str] = Field(default_factory=list, max_length=32)


class SymbolicTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    task_id: str
    definitions: dict[str, Definition] = Field(default_factory=dict, max_length=32)
    observations: list[GroundedObservation] = Field(default_factory=list, max_length=128)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=32)
    unresolved: list[str] = Field(default_factory=list, max_length=32)
    program_refs: list[str] = Field(default_factory=list, max_length=32)
    artifact_refs: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def valid_definitions(self) -> SymbolicTaskState:
        active, done = set(), set()

        def visit(name):
            if name not in self.definitions:
                raise ValueError(f"undefined concept: {name}")
            if name in active:
                raise ValueError("cyclic definition")
            if name in done:
                return
            active.add(name)
            for dependency in self.definitions[name].depends_on:
                visit(dependency)
            active.remove(name)
            done.add(name)

        for name in self.definitions:
            visit(name)
        return self

    def verify_grounding(self, artifacts: ArtifactStore) -> None:
        for key in self.artifact_refs + self.program_refs:
            artifacts.get(key)
        for hypothesis in self.hypotheses:
            for key in hypothesis.evidence + hypothesis.counterexamples:
                if key not in self.artifact_refs + self.program_refs:
                    raise ValueError("hypothesis references undeclared evidence")
                artifacts.get(key)
        for observation in self.observations:
            if observation.artifact not in self.artifact_refs:
                raise ValueError("observation references undeclared artifact")
            grid = artifacts.get(observation.artifact)
            validate_grid(grid)
            try:
                actual = grid[observation.row][observation.column]
            except IndexError as error:
                raise ValueError("observation outside grid") from error
            if actual != observation.color:
                raise ValueError("observation contradicts authoritative grid")


def task_hash(task_id: str) -> str:
    return hashlib.sha256(("arc-v4-split-v1:" + task_id).encode()).hexdigest()


def freeze_split(tasks: list[ArcTask], path: Path) -> dict:
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("split needs nonempty unique tasks")
    groups = {"training": [], "development": [], "lockbox": []}
    hashes = {}
    for task in sorted(tasks, key=lambda t: task_hash(t.task_id)):
        bucket = int(task_hash(task.task_id), 16) % 100
        group = "training" if bucket < 80 else "development" if bucket < 90 else "lockbox"
        groups[group].append(task.task_id)
        hashes[task.task_id] = content_hash(task.model_dump(mode="json"))
    manifest = {
        "version": 1,
        "algorithm": "sha256(arc-v4-split-v1:task_id) mod 100",
        "groups": groups,
        "task_hashes": hashes,
    }
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("frozen split or source data changed; refusing overwrite")
    if not path.exists():
        atomic_json(path, manifest)
    return manifest


def blind_task(task: ArcTask) -> ArcTask:
    payload = task.model_dump(mode="json")
    payload["test"] = [{"input": pair.input} for pair in task.test]
    return ArcTask.model_validate(payload)


class ExperimentRun:
    """One process owns the run; every returned response is durable before interpretation."""

    def __init__(self, root: Path, identity: dict, *, resume: bool = False):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.lock = (root / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("another process owns this run") from None
        self.db = sqlite3.connect(root / "run.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS work (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT, error TEXT);
        """)
        prior = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if prior and (not resume or json.loads(prior[0]) != identity):
            self.close()
            raise ValueError("existing run requires --resume and identical configuration/artifacts")
        if not prior:
            with self.db:
                self.db.execute(
                    "INSERT INTO metadata VALUES ('identity', ?)",
                    (json.dumps(identity, sort_keys=True),),
                )
        self.artifacts = ArtifactStore(root / "artifacts")

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT payload FROM work WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, key: str, value: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO work VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True, allow_nan=False)),
            )

    def request(self, key: str, request: dict) -> dict | None:
        row = self.db.execute(
            "SELECT request,response,error FROM calls WHERE key=?", (key,)
        ).fetchone()
        if row:
            if json.loads(row[0]) != request:
                raise ValueError("resumed request changed")
            if row[1] is not None:
                return json.loads(row[1])
            raise ValueError(
                row[2] or "interrupted in-flight request; no completed response to replay"
            )
        with self.db:
            self.db.execute(
                "INSERT INTO calls(key,request) VALUES (?,?)",
                (key, json.dumps(request, sort_keys=True)),
            )
        return None

    def saved_request(self, key: str) -> dict | None:
        """Recover the exact prepared request before re-rendering an adaptive context."""
        row = self.db.execute("SELECT request FROM calls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def response(self, key: str, response: dict) -> None:
        with self.db:
            self.db.execute(
                "UPDATE calls SET response=? WHERE key=?",
                (json.dumps(response, allow_nan=False), key),
            )

    def call_error(self, key: str, error: str) -> None:
        with self.db:
            self.db.execute("UPDATE calls SET error=? WHERE key=?", (error, key))

    def close(self) -> None:
        self.db.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
