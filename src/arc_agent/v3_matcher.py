from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any

from arc_agent.models import ArcTask, Attempt, Grid
from arc_agent.v3_config import V3SearchConfig
from arc_agent.v3_dsl import (
    deterministic_search_signatures,
    make_candidate,
    mutate_signature,
    preconditions_pass,
    signature_complexity,
    verify_signature,
)
from arc_agent.v3_fingerprint import fingerprint_distance, fingerprint_task
from arc_agent.v3_models import (
    GameSignature,
    SignatureCandidate,
    SignatureMatch,
    SignatureVerification,
    TaskFingerprint,
)


def sanitize_task(task: ArcTask) -> ArcTask:
    return task.model_copy(
        update={
            "test": [pair.model_copy(update={"output": None}) for pair in task.test],
        }
    )


def _parameter_feasibility(verification: SignatureVerification) -> float:
    if any(failure.case == "static" for failure in verification.failures):
        return 0.0
    failed_executions = sum(
        failure.case.startswith("demo_") and failure.actual is None
        for failure in verification.failures
    )
    return max(0.0, 1.0 - failed_executions / max(1, verification.total_pairs))


def rank_signatures(
    task: ArcTask,
    signatures: list[GameSignature],
    source_fingerprints: dict[str, TaskFingerprint],
    *,
    top_k: int = 3,
    excluded_task_ids: set[str] | None = None,
    task_fingerprint: TaskFingerprint | None = None,
) -> list[tuple[GameSignature, SignatureMatch]]:
    target = task_fingerprint or fingerprint_task(task)
    excluded = excluded_task_ids or set()
    ranked: list[tuple[GameSignature, SignatureMatch]] = []
    for signature in signatures:
        if excluded.intersection(signature.source_task_ids):
            continue
        if not preconditions_pass(signature, task):
            continue
        sources = [
            source_fingerprints[task_id]
            for task_id in signature.source_task_ids
            if task_id in source_fingerprints
        ]
        similarity = (
            max(1.0 / (1.0 + fingerprint_distance(target, source)) for source in sources)
            if sources
            else 0.0
        )
        if signature.family == target.predicted_family:
            similarity = min(1.0, similarity + 0.15)
        verification = verify_signature(
            signature,
            task,
            predict_tests=False,
            run_leave_one_out=False,
            run_declared_invariants=False,
        )
        feasibility = _parameter_feasibility(verification)
        demo = (
            verification.exact_pairs / max(1, verification.total_pairs) * 0.8
            + verification.balanced_accuracy * 0.2
        )
        score = demo * 1_000.0 + feasibility * 100.0 + similarity * 50.0
        ranked.append(
            (
                signature,
                SignatureMatch(
                    signature_hash=signature.canonical_hash,
                    source_task_ids=signature.source_task_ids,
                    family=signature.family,
                    hard_preconditions_passed=True,
                    fingerprint_similarity=similarity,
                    parameter_feasibility=feasibility,
                    demonstration_score=demo,
                    score=score,
                ),
            )
        )
    return sorted(
        ranked,
        key=lambda item: (-item[1].score, signature_complexity(item[0]), item[0].canonical_hash),
    )[:top_k]


def generate_candidates(
    task: ArcTask,
    signatures: list[GameSignature],
    source_fingerprints: dict[str, TaskFingerprint],
    *,
    settings: V3SearchConfig,
    excluded_task_ids: set[str] | None = None,
    task_fingerprint: TaskFingerprint | None = None,
) -> tuple[list[SignatureCandidate], list[SignatureMatch]]:
    sanitized = sanitize_task(task)
    target_fingerprint = task_fingerprint or fingerprint_task(sanitized)
    retrieved = rank_signatures(
        sanitized,
        signatures,
        source_fingerprints,
        top_k=settings.retrieved_signatures,
        excluded_task_ids=excluded_task_ids,
        task_fingerprint=target_fingerprint,
    )
    candidates: list[SignatureCandidate] = []
    seen_signatures: set[str] = set()

    def add(
        signature: GameSignature,
        source_kind: str,
        retrieval_rank: int | None = None,
    ) -> None:
        if (
            signature.canonical_hash in seen_signatures
            or len(candidates) >= settings.max_candidates_per_task
        ):
            return
        seen_signatures.add(signature.canonical_hash)
        verification = verify_signature(
            signature,
            sanitized,
            leave_one_out_weight=settings.leave_one_out_weight,
            invariant_weight=settings.invariant_weight,
            complexity_weight=settings.complexity_weight,
        )
        candidates.append(
            make_candidate(
                sanitized,
                source_kind,
                signature,
                verification,
                retrieval_rank=retrieval_rank,
            )
        )

    for rank, (signature, _) in enumerate(retrieved, start=1):
        add(signature, "retrieved", rank)
        for mutation in mutate_signature(
            signature,
            source_task_ids=signature.source_task_ids,
            limit=settings.mutations_per_signature,
        ):
            add(mutation, "mutation", rank)

    predicted_family = target_fingerprint.predicted_family
    if not any(candidate.verification.accepted for candidate in candidates):
        search_pool = deterministic_search_signatures(
            sanitized,
            family=None,
            limit=max(settings.family_search_limit, settings.global_search_limit),
        )
        family_pool = [
            signature for signature in search_pool if signature.family == predicted_family
        ][: settings.family_search_limit]
        for signature in family_pool:
            add(signature, "family_search")
    if not any(candidate.verification.accepted for candidate in candidates):
        for signature in search_pool[: settings.global_search_limit]:
            add(signature, "global_search")
    return candidates, [match for _, match in retrieved]


def _candidate_family(candidate: SignatureCandidate) -> str:
    return candidate.signature.family if candidate.signature is not None else "fallback"


def select_attempts(
    task: ArcTask,
    candidates: list[SignatureCandidate],
) -> tuple[list[Attempt], dict[str, Any]]:
    viable = [
        candidate
        for candidate in candidates
        if len(candidate.verification.predictions) == len(task.test)
    ]
    groups: dict[str, list[SignatureCandidate]] = defaultdict(list)
    for candidate in viable:
        key = json.dumps(candidate.verification.predictions, sort_keys=True, separators=(",", ":"))
        groups[key].append(candidate)
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            not any(candidate.verification.accepted for candidate in group),
            -len(group),
            -max(candidate.verification.score for candidate in group),
            min(
                signature_complexity(candidate.signature)
                for candidate in group
                if candidate.signature is not None
            ),
            min(candidate.candidate_id for candidate in group),
        ),
    )
    selected: list[tuple[SignatureCandidate | None, list[Grid]]] = []
    if ranked_groups:
        first = max(ranked_groups[0], key=lambda candidate: candidate.verification.score)
        selected.append((first, first.verification.predictions))
        first_family = _candidate_family(first)
        alternatives = [
            group
            for group in ranked_groups[1:]
            if any(_candidate_family(candidate) != first_family for candidate in group)
        ] or ranked_groups[1:]
        if alternatives:
            second = max(alternatives[0], key=lambda candidate: candidate.verification.score)
            selected.append((second, second.verification.predictions))

    fallbacks = [
        [[list(row) for row in pair.input] for pair in task.test],
        [[[0 for _ in row] for row in pair.input] for pair in task.test],
    ]
    for predictions in fallbacks:
        if len(selected) == 2:
            break
        if all(predictions != existing for _, existing in selected):
            selected.append((None, deepcopy(predictions)))
    if len(selected) == 1:
        selected.append(selected[0])
    attempts = [
        Attempt(attempt_1=selected[0][1][index], attempt_2=selected[1][1][index])
        for index in range(len(task.test))
    ]
    return attempts, {
        "attempt_1_candidate": selected[0][0].candidate_id if selected[0][0] else None,
        "attempt_2_candidate": selected[1][0].candidate_id if selected[1][0] else None,
        "attempt_1_family": _candidate_family(selected[0][0]) if selected[0][0] else "fallback",
        "attempt_2_family": _candidate_family(selected[1][0]) if selected[1][0] else "fallback",
        "guarded_candidates": sum(candidate.verification.accepted for candidate in candidates),
    }
