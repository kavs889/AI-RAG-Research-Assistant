from __future__ import annotations

import math
from collections.abc import Iterable


def _validate_k(k: int) -> None:
    """Validate a retrieval cutoff."""
    if k <= 0:
        raise ValueError("k must be greater than zero")


def precision_at_k(
    retrieved_ids: Iterable[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate precision@k."""

    _validate_k(k)

    retrieved = list(retrieved_ids)[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for document_id in retrieved
        if document_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(
    retrieved_ids: Iterable[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate recall@k."""

    _validate_k(k)

    if not relevant_ids:
        return 0.0

    retrieved = list(retrieved_ids)[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for document_id in retrieved
        if document_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Iterable[str],
    relevant_ids: set[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Iterable[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain@k."""

    _validate_k(k)

    retrieved = list(retrieved_ids)[:k]

    if not retrieved:
        return 0.0

    dcg = 0.0

    for rank, document_id in enumerate(retrieved, start=1):
        relevance = max(
            0.0,
            float(relevance_scores.get(document_id, 0.0)),
        )

        dcg += relevance / math.log2(rank + 1)

    ideal_scores = sorted(
        (
            max(0.0, float(score))
            for score in relevance_scores.values()
        ),
        reverse=True,
    )[:k]

    if not ideal_scores:
        return 0.0

    idcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg