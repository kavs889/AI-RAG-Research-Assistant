from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Evaluation metrics for one retrieval cutoff."""

    precision: float
    recall: float
    mrr: float
    ndcg: float


class RetrievalEvaluator:
    """Evaluate retrieval quality using standard ranking metrics."""

    def evaluate(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> RetrievalEvaluation:
        """Evaluate a retrieval result at one cutoff."""

        if k <= 0:
            raise ValueError("k must be greater than zero")

        relevance_scores = {
            document_id: 1.0
            for document_id in relevant_ids
        }

        return RetrievalEvaluation(
            precision=precision_at_k(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=k,
            ),
            recall=recall_at_k(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=k,
            ),
            mrr=reciprocal_rank(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
            ),
            ndcg=ndcg_at_k(
                retrieved_ids=retrieved_ids,
                relevance_scores=relevance_scores,
                k=k,
            ),
        )

    def evaluate_at_k_values(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k_values: list[int],
    ) -> dict[int, RetrievalEvaluation]:
        """Evaluate retrieval quality at multiple cutoff values."""

        if not k_values:
            return {}

        return {
            k: self.evaluate(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=k,
            )
            for k in k_values
        }