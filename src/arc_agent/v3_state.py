from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc_agent.models import Attempt
from arc_agent.v2_models import ResponseSnapshot, ResponseUsage
from arc_agent.v3_models import (
    GameSignature,
    SignatureCandidate,
    SignatureVerification,
    TaskFingerprint,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class WorkspaceBusy(RuntimeError):
    pass


class StateMismatch(RuntimeError):
    pass


class V3State:
    def __init__(
        self,
        workspace: str | Path,
        *,
        read_only: bool = False,
        acquire_workspace_lock: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        if read_only and not self.workspace.exists():
            raise FileNotFoundError(self.workspace)
        if not read_only:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._lock_stream: Any | None = None
        if not read_only and acquire_workspace_lock:
            self._lock_stream = (self.workspace / ".lock").open("a+")
            try:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock_stream.close()
                raise WorkspaceBusy(f"workspace is already active: {self.workspace}") from exc
        database = self.workspace / "state.sqlite3"
        if read_only:
            self.connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        else:
            self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._create_schema()

    def __enter__(self) -> V3State:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS phase_tasks (
                phase TEXT NOT NULL,
                task_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                family TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                round_index INTEGER NOT NULL DEFAULT 0,
                no_progress INTEGER NOT NULL DEFAULT 0,
                best_score REAL NOT NULL DEFAULT -1,
                best_candidate_id TEXT,
                previous_response_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (phase, task_id),
                UNIQUE (phase, position)
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL,
                phase TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt TEXT NOT NULL,
                previous_response_id TEXT,
                token_limit INTEGER NOT NULL,
                model TEXT NOT NULL,
                response_id TEXT UNIQUE,
                status TEXT NOT NULL,
                body_json TEXT,
                usage_json TEXT,
                ingested INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                request_key TEXT NOT NULL,
                status TEXT NOT NULL,
                body_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (request_key) REFERENCES requests(request_key)
            );
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                source_kind TEXT NOT NULL,
                response_id TEXT,
                signature_json TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                failure_signature TEXT NOT NULL,
                score REAL NOT NULL,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signatures (
                canonical_hash TEXT PRIMARY KEY,
                source_task_ids_json TEXT NOT NULL,
                signature_json TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fingerprints (
                task_id TEXT PRIMARY KEY,
                fingerprint_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS matches (
                task_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                match_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (task_id, rank)
            );
            CREATE TABLE IF NOT EXISTS evaluation_outputs (
                task_id TEXT PRIMARY KEY,
                attempts_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS v3_requests_task_idx
                ON requests(phase, task_id, round_index);
            CREATE INDEX IF NOT EXISTS v3_responses_request_idx
                ON responses(request_key);
            CREATE INDEX IF NOT EXISTS v3_candidates_task_idx
                ON candidates(phase, task_id, score DESC);
            """
        )
        self.connection.commit()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterable[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("read-only V3 state cannot be mutated")
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def set_meta(self, key: str, value: Any) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO meta(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, _json(value), now),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row is not None else default

    def initialize_phase(
        self,
        *,
        phase: str,
        tasks: list[tuple[str, str]],
        dataset_hash: str,
        config_hash: str,
        dsl_hash: str,
        resume_only: bool,
    ) -> None:
        existing = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM phase_tasks WHERE phase=?", (phase,)
            ).fetchone()[0]
        )
        if resume_only and not existing:
            raise StateMismatch(f"no {phase} run exists in {self.workspace}")
        task_ids = [task_id for task_id, _ in tasks]
        order_hash = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
        wanted = {
            f"{phase}_dataset_hash": dataset_hash,
            f"{phase}_config_hash": config_hash,
            f"{phase}_dsl_hash": dsl_hash,
            f"{phase}_order_hash": order_hash,
        }
        if existing:
            for key, value in wanted.items():
                actual = self.get_meta(key)
                if actual != value:
                    raise StateMismatch(f"{key} changed; expected {actual}, got {value}")
        else:
            now = _now()
            with self.transaction() as connection:
                connection.executemany(
                    """
                    INSERT INTO phase_tasks(
                        phase, task_id, position, family, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    [
                        (phase, task_id, position, family, now, now)
                        for position, (task_id, family) in enumerate(tasks)
                    ],
                )
            for key, value in wanted.items():
                self.set_meta(key, value)
        self.begin_active(phase)

    def begin_active(self, phase: str) -> None:
        self.set_meta(f"{phase}_status", "running")
        self.set_meta(f"{phase}_active_started", time.time())
        self.set_meta(f"{phase}_worker_pid", os.getpid())

    def checkpoint_active(self, phase: str) -> float:
        started = float(self.get_meta(f"{phase}_active_started", time.time()))
        accumulated = float(self.get_meta(f"{phase}_active_seconds", 0.0))
        now = time.time()
        accumulated += max(0.0, now - started)
        self.set_meta(f"{phase}_active_seconds", accumulated)
        self.set_meta(f"{phase}_active_started", now)
        return accumulated

    def finish_phase(self, phase: str, status: str, message: str = "") -> None:
        self.checkpoint_active(phase)
        self.set_meta(f"{phase}_status", status)
        self.set_meta(f"{phase}_message", message)
        self.set_meta(f"{phase}_worker_pid", None)

    def task_row(self, phase: str, task_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM phase_tasks WHERE phase=? AND task_id=?", (phase, task_id)
        ).fetchone()
        if row is None:
            raise KeyError((phase, task_id))
        return dict(row)

    def next_task(
        self,
        phase: str,
        statuses: tuple[str, ...] = ("pending", "running"),
    ) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in statuses)
        row = self.connection.execute(
            f"""
            SELECT * FROM phase_tasks
            WHERE phase=? AND status IN ({placeholders})
            ORDER BY position LIMIT 1
            """,
            (phase, *statuses),
        ).fetchone()
        return dict(row) if row else None

    def update_task(self, phase: str, task_id: str, **values: Any) -> None:
        if not values:
            return
        allowed = {
            "status",
            "round_index",
            "no_progress",
            "best_score",
            "best_candidate_id",
            "previous_response_id",
            "error",
        }
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"unsupported task fields: {sorted(unexpected)}")
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE phase_tasks SET {assignments} WHERE phase=? AND task_id=?",
                (*values.values(), phase, task_id),
            )

    def prepare_request(
        self,
        *,
        phase: str,
        task_id: str,
        round_index: int,
        prompt: str,
        previous_response_id: str | None,
        token_limit: int,
        model: str,
    ) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        request_key = hashlib.sha256(
            f"v3\0{phase}\0{task_id}\0{round_index}\0{prompt_hash}\0{model}".encode()
        ).hexdigest()
        now = _now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO requests(
                    request_key, idempotency_key, phase, task_id, round_index, prompt_hash, prompt,
                    previous_response_id, token_limit, model, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
                """,
                (
                    request_key,
                    request_key,
                    phase,
                    task_id,
                    round_index,
                    prompt_hash,
                    prompt,
                    previous_response_id,
                    token_limit,
                    model,
                    now,
                    now,
                ),
            )
        return self.request(request_key)

    def request(self, request_key: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM requests WHERE request_key=?", (request_key,)
        ).fetchone()
        if row is None:
            raise KeyError(request_key)
        return dict(row)

    def request_manifest(self, phase: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT request_key, idempotency_key, task_id, round_index, prompt_hash,
                    previous_response_id, token_limit, model, response_id, status
                FROM requests WHERE phase=? ORDER BY task_id, round_index, request_key
                """,
                (phase,),
            ).fetchall()
        ]

    def prompts(self, phase: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT request_key, task_id, round_index, prompt_hash, prompt
                FROM requests WHERE phase=? ORDER BY task_id, round_index, request_key
                """,
                (phase,),
            ).fetchall()
        ]

    def record_response(self, request_key: str, snapshot: ResponseSnapshot) -> None:
        with self.transaction(immediate=True) as connection:
            now = _now()
            connection.execute(
                """
                INSERT INTO responses(
                    response_id, request_key, status, body_json, usage_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(response_id) DO UPDATE SET
                    status=excluded.status, body_json=excluded.body_json,
                    usage_json=excluded.usage_json, updated_at=excluded.updated_at
                """,
                (
                    snapshot.response_id,
                    request_key,
                    snapshot.status,
                    _json(snapshot.body),
                    snapshot.usage.model_dump_json(),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE requests SET response_id=?, status=?, body_json=?, usage_json=?,
                    error=NULL, updated_at=? WHERE request_key=?
                """,
                (
                    snapshot.response_id,
                    snapshot.status,
                    _json(snapshot.body),
                    snapshot.usage.model_dump_json(),
                    now,
                    request_key,
                ),
            )

    def record_request_error(self, request_key: str, status: str, error: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE requests SET status=?, error=?, retry_count=retry_count+1, updated_at=?
                WHERE request_key=?
                """,
                (status, error, _now(), request_key),
            )

    def reset_expired_response(self, request_key: str) -> None:
        current_idempotency_key = str(
            self.connection.execute(
                "SELECT idempotency_key FROM requests WHERE request_key=?", (request_key,)
            ).fetchone()[0]
        )
        idempotency_key = hashlib.sha256(
            f"{request_key}\0retry\0{current_idempotency_key}".encode()
        ).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE requests SET response_id=NULL, status='prepared', body_json=NULL,
                    usage_json=NULL, previous_response_id=NULL, idempotency_key=?,
                    updated_at=? WHERE request_key=?
                """,
                (idempotency_key, _now(), request_key),
            )

    def mark_ingested(self, request_key: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE requests SET ingested=1, updated_at=? WHERE request_key=?",
                (_now(), request_key),
            )

    def record_candidate(
        self,
        *,
        phase: str,
        task_id: str,
        round_index: int,
        source_kind: str,
        response_id: str | None,
        signature: GameSignature,
        verification: SignatureVerification,
        failure_signature: str,
    ) -> str:
        candidate_id = hashlib.sha256(
            f"{phase}\0{task_id}\0{round_index}\0{signature.canonical_hash}".encode()
        ).hexdigest()
        now = _now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO candidates(
                    candidate_id, phase, task_id, round_index, source_kind, response_id,
                    signature_json, verification_json, failure_signature, score, accepted,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    phase,
                    task_id,
                    round_index,
                    source_kind,
                    response_id,
                    signature.model_dump_json(),
                    verification.model_dump_json(),
                    failure_signature,
                    verification.score,
                    int(verification.accepted),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT best_score FROM phase_tasks WHERE phase=? AND task_id=?",
                (phase, task_id),
            ).fetchone()
            if row is not None and verification.score > float(row["best_score"]):
                connection.execute(
                    """
                    UPDATE phase_tasks SET best_score=?, best_candidate_id=?, updated_at=?
                    WHERE phase=? AND task_id=?
                    """,
                    (verification.score, candidate_id, now, phase, task_id),
                )
        return candidate_id

    def candidates(self, phase: str, task_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM candidates WHERE phase=? AND task_id=?
                ORDER BY score DESC, candidate_id
                """,
                (phase, task_id),
            ).fetchall()
        ]

    def recent_failure_signatures(self, phase: str, task_id: str, limit: int) -> list[str]:
        return [
            str(row["failure_signature"])
            for row in self.connection.execute(
                """
                SELECT failure_signature FROM candidates
                WHERE phase=? AND task_id=? ORDER BY round_index DESC LIMIT ?
                """,
                (phase, task_id, limit),
            ).fetchall()
        ]

    def accept_signature(
        self,
        *,
        task_id: str,
        signature: GameSignature,
        verification: SignatureVerification,
        candidate_id: str,
    ) -> None:
        now = _now()
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT source_task_ids_json FROM signatures WHERE canonical_hash=?",
                (signature.canonical_hash,),
            ).fetchone()
            sources = set(signature.source_task_ids)
            if existing is not None:
                sources.update(json.loads(existing["source_task_ids_json"]))
            stored = signature.model_copy(update={"source_task_ids": sorted(sources)})
            connection.execute(
                """
                INSERT INTO signatures(
                    canonical_hash, source_task_ids_json, signature_json, verification_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_hash) DO UPDATE SET
                    source_task_ids_json=excluded.source_task_ids_json,
                    signature_json=excluded.signature_json,
                    verification_json=excluded.verification_json,
                    updated_at=excluded.updated_at
                """,
                (
                    stored.canonical_hash,
                    _json(stored.source_task_ids),
                    stored.model_dump_json(),
                    verification.model_dump_json(),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE phase_tasks SET status='accepted', best_candidate_id=?,
                    best_score=?, error=NULL, updated_at=?
                WHERE phase='pilot' AND task_id=?
                """,
                (candidate_id, verification.score, now, task_id),
            )

    def signatures(self) -> list[GameSignature]:
        return [
            GameSignature.model_validate_json(row["signature_json"])
            for row in self.connection.execute(
                "SELECT signature_json FROM signatures ORDER BY canonical_hash"
            ).fetchall()
        ]

    def record_fingerprint(self, fingerprint: TaskFingerprint) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fingerprints(task_id, fingerprint_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    fingerprint_json=excluded.fingerprint_json, updated_at=excluded.updated_at
                """,
                (fingerprint.task_id, fingerprint.model_dump_json(), now, now),
            )

    def fingerprints(self) -> dict[str, TaskFingerprint]:
        return {
            row["task_id"]: TaskFingerprint.model_validate_json(row["fingerprint_json"])
            for row in self.connection.execute(
                "SELECT task_id, fingerprint_json FROM fingerprints"
            ).fetchall()
        }

    def record_matches(self, task_id: str, matches: list[dict[str, Any]]) -> None:
        with self.transaction(immediate=True) as connection:
            connection.execute("DELETE FROM matches WHERE task_id=?", (task_id,))
            connection.executemany(
                "INSERT INTO matches(task_id, rank, match_json, created_at) VALUES (?, ?, ?, ?)",
                [
                    (task_id, rank, _json(match), _now())
                    for rank, match in enumerate(matches, start=1)
                ],
            )

    def save_evaluation_output(
        self,
        *,
        task_id: str,
        attempts: list[Attempt],
        provenance: dict[str, Any],
        candidates: list[SignatureCandidate],
    ) -> None:
        now = _now()
        summaries = [
            {
                "candidate_id": candidate.candidate_id,
                "source_kind": candidate.source_kind,
                "signature_hash": (
                    candidate.signature.canonical_hash if candidate.signature else None
                ),
                "accepted": candidate.verification.accepted,
                "score": candidate.verification.score,
                "predictions": candidate.verification.predictions,
            }
            for candidate in candidates
        ]
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO evaluation_outputs(
                    task_id, attempts_json, provenance_json, candidates_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    attempts_json=excluded.attempts_json,
                    provenance_json=excluded.provenance_json,
                    candidates_json=excluded.candidates_json,
                    updated_at=excluded.updated_at
                """,
                (
                    task_id,
                    _json([attempt.model_dump(mode="json") for attempt in attempts]),
                    _json(provenance),
                    _json(summaries),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE phase_tasks SET status='accepted', updated_at=?
                WHERE phase='evaluation' AND task_id=?
                """,
                (now, task_id),
            )

    def evaluation_outputs(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM evaluation_outputs ORDER BY task_id"
            ).fetchall()
        ]

    def usage(self) -> ResponseUsage:
        total = ResponseUsage()
        for row in self.connection.execute(
            "SELECT usage_json FROM responses WHERE usage_json IS NOT NULL"
        ).fetchall():
            usage = ResponseUsage.model_validate_json(row["usage_json"])
            total.input_tokens += usage.input_tokens
            total.cached_input_tokens += usage.cached_input_tokens
            total.cache_write_input_tokens += usage.cache_write_input_tokens
            total.output_tokens += usage.output_tokens
            total.reasoning_tokens += usage.reasoning_tokens
        return total

    def summary(self, phase: str) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM phase_tasks WHERE phase=? GROUP BY status",
            (phase,),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        active = self.next_task(phase)
        requests = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM requests WHERE phase=?", (phase,)
            ).fetchone()[0]
        )
        last_response = self.connection.execute(
            """
            SELECT response_id, status, updated_at FROM requests
            WHERE phase=? AND response_id IS NOT NULL
            ORDER BY updated_at DESC LIMIT 1
            """,
            (phase,),
        ).fetchone()
        usage = self.usage()
        return {
            "phase": phase,
            "status": self.get_meta(f"{phase}_status", "not_started"),
            "active_task_id": active["task_id"] if active else None,
            "refinement_round": int(active["round_index"]) if active else None,
            "no_progress_count": int(active["no_progress"]) if active else None,
            "completed_tasks": counts.get("accepted", 0),
            "total_tasks": sum(counts.values()),
            "task_status_counts": counts,
            "model_requests": requests,
            "last_response": dict(last_response) if last_response else None,
            "usage": usage.model_dump(mode="json"),
            "active_seconds": float(self.get_meta(f"{phase}_active_seconds", 0.0)),
            "worker_pid": self.get_meta(f"{phase}_worker_pid"),
            "frozen": bool(self.get_meta(f"{phase}_frozen", False)),
            "message": self.get_meta(f"{phase}_message", ""),
        }
