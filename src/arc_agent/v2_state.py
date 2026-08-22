from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc_agent.v2_models import BankProgram, InductionVerification, ResponseSnapshot


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


class WorkspaceBusy(RuntimeError):
    pass


class StateMismatch(RuntimeError):
    pass


class V2State:
    def __init__(self, workspace: str | Path, *, read_only: bool = False) -> None:
        self.workspace = Path(workspace)
        if read_only and not self.workspace.exists():
            raise FileNotFoundError(self.workspace)
        if not read_only:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._lock_stream: Any | None = None
        if not read_only:
            lock_path = self.workspace / ".lock"
            self._lock_stream = lock_path.open("a+")
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
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._create_schema()

    def close(self) -> None:
        self.connection.close()
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def __enter__(self) -> V2State:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
                status TEXT NOT NULL DEFAULT 'pending',
                round_index INTEGER NOT NULL DEFAULT 0,
                lens_index INTEGER NOT NULL DEFAULT 0,
                token_index INTEGER NOT NULL DEFAULT 0,
                no_progress INTEGER NOT NULL DEFAULT 0,
                best_score REAL NOT NULL DEFAULT -1,
                best_candidate_id TEXT,
                previous_response_id TEXT,
                direct_cursor INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (phase, task_id),
                UNIQUE (phase, position)
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt TEXT NOT NULL,
                previous_response_id TEXT,
                token_limit INTEGER NOT NULL,
                response_id TEXT UNIQUE,
                status TEXT NOT NULL,
                body_json TEXT,
                usage_json TEXT,
                ingested INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_retry_after REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                source_kind TEXT NOT NULL,
                response_id TEXT,
                hypothesis TEXT NOT NULL,
                strategy_tags_json TEXT NOT NULL,
                invariants_json TEXT NOT NULL,
                python_source TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                predictions_hash TEXT NOT NULL,
                score REAL NOT NULL,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS programs (
                program_hash TEXT PRIMARY KEY,
                source_task_ids_json TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                strategy_tags_json TEXT NOT NULL,
                invariants_json TEXT NOT NULL,
                python_source TEXT NOT NULL,
                canonical_ast TEXT NOT NULL,
                features_json TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compatibility (
                program_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                score REAL NOT NULL,
                exact INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                verification_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (program_hash, task_id)
            );
            CREATE TABLE IF NOT EXISTS evaluation_outputs (
                task_id TEXT PRIMARY KEY,
                attempts_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS requests_task_index
                ON requests(phase, task_id, round_index);
            CREATE INDEX IF NOT EXISTS candidates_task_index
                ON candidates(phase, task_id, score DESC);
            """
        )
        columns = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(requests)").fetchall()
        }
        if "retry_count" not in columns:
            self.connection.execute(
                "ALTER TABLE requests ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
            )
        if "last_retry_after" not in columns:
            self.connection.execute("ALTER TABLE requests ADD COLUMN last_retry_after REAL")
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("read-only state cannot be mutated")
        with self.connection:
            yield self.connection

    def set_meta(self, key: str, value: Any) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO meta(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, _json(value), _now()),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row is not None else default

    def initialize_phase(
        self,
        *,
        phase: str,
        task_ids: list[str],
        dataset_hash: str,
        config_hash: str,
        resume_only: bool = False,
    ) -> None:
        existing = self.connection.execute(
            "SELECT COUNT(*) AS count FROM phase_tasks WHERE phase=?", (phase,)
        ).fetchone()["count"]
        if resume_only and not existing:
            raise StateMismatch(f"no unfinished {phase} run exists in {self.workspace}")
        order_hash = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
        if existing:
            for key, wanted in (
                (f"{phase}_dataset_hash", dataset_hash),
                (f"{phase}_config_hash", config_hash),
                (f"{phase}_order_hash", order_hash),
            ):
                actual = self.get_meta(key)
                if actual != wanted:
                    raise StateMismatch(f"{key} changed; expected {actual}, got {wanted}")
        else:
            created_at = _now()
            with self.transaction() as connection:
                connection.executemany(
                    """
                    INSERT INTO phase_tasks(
                        phase, task_id, position, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    [
                        (phase, task_id, position, created_at, created_at)
                        for position, task_id in enumerate(task_ids)
                    ],
                )
            self.set_meta(f"{phase}_dataset_hash", dataset_hash)
            self.set_meta(f"{phase}_config_hash", config_hash)
            self.set_meta(f"{phase}_order_hash", order_hash)
            self.set_meta(f"{phase}_task_ids", task_ids)
        self.set_running(phase)

    def active_task(self, phase: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM phase_tasks
            WHERE phase=? AND status != 'accepted'
            ORDER BY position LIMIT 1
            """,
            (phase,),
        ).fetchone()
        return dict(row) if row is not None else None

    def task_row(self, phase: str, task_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM phase_tasks WHERE phase=? AND task_id=?", (phase, task_id)
        ).fetchone()
        if row is None:
            raise KeyError((phase, task_id))
        return dict(row)

    def update_task(self, phase: str, task_id: str, **updates: Any) -> None:
        if not updates:
            return
        updates["updated_at"] = _now()
        allowed = {
            "status",
            "round_index",
            "lens_index",
            "token_index",
            "no_progress",
            "best_score",
            "best_candidate_id",
            "previous_response_id",
            "direct_cursor",
            "updated_at",
        }
        if set(updates) - allowed:
            raise ValueError(f"unsupported task updates: {sorted(set(updates) - allowed)}")
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE phase_tasks SET {assignments} WHERE phase=? AND task_id=?",
                (*updates.values(), phase, task_id),
            )

    def restart_active_task(self, phase: str) -> str:
        task = self.active_task(phase)
        if task is None:
            raise StateMismatch(f"{phase} is already complete")
        self.update_task(
            phase,
            task["task_id"],
            round_index=int(task["round_index"]) + 1,
            lens_index=int(task["lens_index"]) + 1,
            no_progress=0,
            previous_response_id=None,
            status="pending",
        )
        return str(task["task_id"])

    def set_running(self, phase: str) -> None:
        if not self.get_meta(f"{phase}_started_at"):
            self.set_meta(f"{phase}_started_at", _now())
        paused_at = self.get_meta(f"{phase}_paused_at")
        if paused_at:
            try:
                paused = datetime.fromisoformat(paused_at)
                seconds = (datetime.now(UTC) - paused).total_seconds()
            except (TypeError, ValueError):
                seconds = 0.0
            self.set_meta(
                f"{phase}_paused_seconds",
                float(self.get_meta(f"{phase}_paused_seconds", 0.0)) + max(0.0, seconds),
            )
            self.set_meta(f"{phase}_paused_at", None)
        self.set_meta(f"{phase}_status", "running")

    def pause(self, phase: str, status: str, message: str) -> None:
        if status not in {"paused_quota", "paused_configuration", "paused_user"}:
            raise ValueError(status)
        self.set_meta(f"{phase}_status", status)
        self.set_meta(f"{phase}_last_error", message)
        if not self.get_meta(f"{phase}_paused_at"):
            self.set_meta(f"{phase}_paused_at", _now())
        if status == "paused_quota":
            self.set_meta(
                f"{phase}_quota_pause_count",
                int(self.get_meta(f"{phase}_quota_pause_count", 0)) + 1,
            )

    def complete_phase(self, phase: str) -> None:
        self.set_meta(f"{phase}_status", "complete")
        self.set_meta(f"{phase}_completed_at", _now())

    def prepare_request(
        self,
        *,
        request_key: str,
        phase: str,
        task_id: str,
        round_index: int,
        prompt: str,
        previous_response_id: str | None,
        token_limit: int,
    ) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        created_at = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO requests(
                    request_key, phase, task_id, round_index, prompt_hash, prompt,
                    previous_response_id, token_limit, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
                ON CONFLICT(request_key) DO NOTHING
                """,
                (
                    request_key,
                    phase,
                    task_id,
                    round_index,
                    prompt_hash,
                    prompt,
                    previous_response_id,
                    token_limit,
                    created_at,
                    created_at,
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM requests WHERE request_key=?", (request_key,)
        ).fetchone()
        if row is None:
            raise RuntimeError("request was not persisted")
        result = dict(row)
        if result["prompt_hash"] != prompt_hash:
            raise StateMismatch("request key was reused for a different prompt")
        return result

    def pending_request(self, phase: str, task_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM requests
            WHERE phase=? AND task_id=? AND ingested=0
            ORDER BY round_index DESC, created_at DESC LIMIT 1
            """,
            (phase, task_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def record_response(self, request_key: str, snapshot: ResponseSnapshot) -> None:
        with self.transaction() as connection:
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
                    _now(),
                    request_key,
                ),
            )

    def record_request_error(
        self,
        request_key: str,
        status: str,
        error: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE requests SET status=?, error=?, retry_count=retry_count+1,
                    last_retry_after=?, updated_at=? WHERE request_key=?
                """,
                (status, error, retry_after, _now(), request_key),
            )

    def mark_request_ingested(self, request_key: str) -> None:
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
        hypothesis: str,
        strategy_tags: list[str],
        invariants: list[str],
        python_source: str,
        verification: InductionVerification,
    ) -> str:
        payload = f"{phase}\0{task_id}\0{source_kind}\0{python_source}"
        candidate_id = hashlib.sha256(payload.encode()).hexdigest()
        predictions_hash = hashlib.sha256(_json(verification.predictions).encode()).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO candidates(
                    candidate_id, phase, task_id, round_index, source_kind, response_id,
                    hypothesis, strategy_tags_json, invariants_json, python_source,
                    verification_json, predictions_hash, score, accepted, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    verification_json=excluded.verification_json,
                    score=excluded.score, accepted=excluded.accepted
                """,
                (
                    candidate_id,
                    phase,
                    task_id,
                    round_index,
                    source_kind,
                    response_id,
                    hypothesis,
                    _json(strategy_tags),
                    _json(invariants),
                    python_source,
                    verification.model_dump_json(),
                    predictions_hash,
                    verification.score,
                    int(verification.accepted),
                    _now(),
                ),
            )
        task = self.task_row(phase, task_id)
        if verification.score > float(task["best_score"]):
            self.update_task(
                phase,
                task_id,
                best_score=verification.score,
                best_candidate_id=candidate_id,
            )
        return candidate_id

    def best_candidate(self, phase: str, task_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM candidates WHERE phase=? AND task_id=?
            ORDER BY score DESC, created_at ASC LIMIT 1
            """,
            (phase, task_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def accepted_candidates(self, phase: str, task_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM candidates WHERE phase=? AND task_id=? AND accepted=1
            ORDER BY score DESC, created_at ASC
            """,
            (phase, task_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def candidates(self, phase: str, task_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM candidates WHERE phase=? AND task_id=?
            ORDER BY accepted DESC, score DESC, created_at ASC
            """,
            (phase, task_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def accept_training_task(self, task_id: str, program: BankProgram, candidate_id: str) -> None:
        existing = self.connection.execute(
            "SELECT source_task_ids_json FROM programs WHERE program_hash=?",
            (program.program_hash,),
        ).fetchone()
        source_ids = set(program.source_task_ids)
        if existing is not None:
            source_ids.update(json.loads(existing["source_task_ids_json"]))
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO programs(
                    program_hash, source_task_ids_json, hypothesis, strategy_tags_json,
                    invariants_json, python_source, canonical_ast, features_json,
                    verification_json, complexity, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(program_hash) DO UPDATE SET
                    source_task_ids_json=excluded.source_task_ids_json,
                    updated_at=excluded.updated_at
                """,
                (
                    program.program_hash,
                    _json(sorted(source_ids)),
                    program.hypothesis,
                    _json(program.strategy_tags),
                    _json(program.invariants),
                    program.python_source,
                    program.canonical_ast,
                    _json(program.features),
                    program.verification.model_dump_json(),
                    program.complexity,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE phase_tasks SET status='accepted', best_candidate_id=?, updated_at=?
                WHERE phase='training' AND task_id=?
                """,
                (candidate_id, now, task_id),
            )

    def list_programs(self) -> list[BankProgram]:
        rows = self.connection.execute("SELECT * FROM programs ORDER BY program_hash").fetchall()
        return [
            BankProgram(
                program_hash=row["program_hash"],
                source_task_ids=json.loads(row["source_task_ids_json"]),
                hypothesis=row["hypothesis"],
                strategy_tags=json.loads(row["strategy_tags_json"]),
                invariants=json.loads(row["invariants_json"]),
                python_source=row["python_source"],
                canonical_ast=row["canonical_ast"],
                features=json.loads(row["features_json"]),
                verification=InductionVerification.model_validate_json(row["verification_json"]),
                complexity=int(row["complexity"]),
            )
            for row in rows
        ]

    def record_compatibility(
        self,
        *,
        program_hash: str,
        task_id: str,
        features: list[float],
        verification: InductionVerification,
    ) -> None:
        compact = verification.model_copy(update={"failures": [], "predictions": []})
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO compatibility(
                    program_hash, task_id, score, exact, features_json,
                    verification_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    program_hash,
                    task_id,
                    verification.score,
                    int(verification.accepted),
                    _json(features),
                    compact.model_dump_json(),
                    _now(),
                ),
            )

    def has_compatibility(self, program_hash: str, task_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM compatibility WHERE program_hash=? AND task_id=?",
            (program_hash, task_id),
        ).fetchone()
        return row is not None

    def compatibility_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT program_hash, task_id, score, exact, features_json
                FROM compatibility ORDER BY task_id, program_hash
                """
            ).fetchall()
        ]

    def save_evaluation_output(
        self, task_id: str, attempts: list[dict[str, Any]], provenance: dict[str, Any]
    ) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_outputs(
                    task_id, attempts_json, provenance_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET attempts_json=excluded.attempts_json,
                    provenance_json=excluded.provenance_json, updated_at=excluded.updated_at
                """,
                (task_id, _json(attempts), _json(provenance), now, now),
            )
            connection.execute(
                """
                UPDATE phase_tasks SET status='accepted', updated_at=?
                WHERE phase='evaluation' AND task_id=?
                """,
                (now, task_id),
            )

    def evaluation_outputs(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.connection.execute(
            "SELECT task_id, attempts_json FROM evaluation_outputs ORDER BY task_id"
        ).fetchall()
        return {row["task_id"]: json.loads(row["attempts_json"]) for row in rows}

    def usage_totals(self) -> dict[str, float | int]:
        totals: dict[str, float | int] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        rows = self.connection.execute(
            """
            SELECT usage_json FROM requests
            WHERE response_id IS NOT NULL AND usage_json IS NOT NULL
            """
        ).fetchall()
        from arc_agent.v2_models import ResponseUsage

        for row in rows:
            usage = ResponseUsage.model_validate_json(row["usage_json"])
            totals["input_tokens"] += usage.input_tokens
            totals["cached_input_tokens"] += usage.cached_input_tokens
            totals["cache_write_input_tokens"] += usage.cache_write_input_tokens
            totals["output_tokens"] += usage.output_tokens
            totals["reasoning_tokens"] += usage.reasoning_tokens
            totals["estimated_cost_usd"] += usage.estimated_cost_usd
        return totals

    def summary(self, phase: str) -> dict[str, Any]:
        counts = {
            row["status"]: int(row["count"])
            for row in self.connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM phase_tasks
                WHERE phase=? GROUP BY status
                """,
                (phase,),
            ).fetchall()
        }
        active = self.active_task(phase)
        last = self.connection.execute(
            """
            SELECT response_id, status, error, updated_at FROM requests
            WHERE phase=? ORDER BY updated_at DESC LIMIT 1
            """,
            (phase,),
        ).fetchone()
        paused_seconds = float(self.get_meta(f"{phase}_paused_seconds", 0.0))
        paused_at = self.get_meta(f"{phase}_paused_at")
        if paused_at:
            with suppress(TypeError, ValueError):
                paused_seconds += max(
                    0.0, (datetime.now(UTC) - datetime.fromisoformat(paused_at)).total_seconds()
                )
        active_seconds = 0.0
        started_at = self.get_meta(f"{phase}_started_at")
        completed_at = self.get_meta(f"{phase}_completed_at")
        if started_at:
            with suppress(TypeError, ValueError):
                start = datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(completed_at) if completed_at else datetime.now(UTC)
                active_seconds = max(0.0, (end - start).total_seconds() - paused_seconds)
        return {
            "phase": phase,
            "status": self.get_meta(f"{phase}_status", "not_started"),
            "authentication": self.get_meta("openai_auth_mode"),
            "active_task": active,
            "counts": counts,
            "last_response": dict(last) if last is not None else None,
            "usage": self.usage_totals(),
            "quota_pause_count": self.get_meta(f"{phase}_quota_pause_count", 0),
            "paused_seconds": paused_seconds,
            "active_seconds": active_seconds,
            "stage": self.get_meta(f"{phase}_stage"),
            "ranker_state": self.get_meta("ranker_state") if phase == "training" else None,
        }


def atomic_write(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, target)
    return target
