from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import Retriever


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A document chunk with its fused retrieval score."""

    chunk: DocumentChunk
    score: float


class HybridRetriever:
    """
    Combine semantic and BM25 retrieval using Reciprocal Rank Fusion.

    RRF score:

        score(d) = sum(
            1 / (rrf_k + rank)
        )

    where rank starts at 1.
    """

    DEFAULT_RRF_K = 60

    def __init__(
        self,
        semantic_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HybridRetrievalResult]:
        """Retrieve and rank chunks using semantic + BM25 retrieval."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Retrieve more candidates than requested so RRF has
        # enough results from both retrieval strategies to fuse.
        candidate_k = max(top_k, 10)

        semantic_results = self.semantic_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        fused_scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}

        # Add semantic retrieval contribution.
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.chunk.chunk_id

            fused_scores[chunk_id] = fused_scores.get(
                chunk_id,
                0.0,
            ) + self._rrf_score(rank)

            chunks[chunk_id] = result.chunk

        # Add BM25 retrieval contribution.
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk.chunk_id

            fused_scores[chunk_id] = fused_scores.get(
                chunk_id,
                0.0,
            ) + self._rrf_score(rank)

            chunks[chunk_id] = result.chunk

        # Deterministic ordering:
        # 1. Highest RRF score first.
        # 2. Chunk ID alphabetically for ties.
        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                chunk_id,
            ),
        )

        return [
            HybridRetrievalResult(
                chunk=chunks[chunk_id],
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids[:top_k]
        ]

    def _rrf_score(self, rank: int) -> float:
        """Calculate the Reciprocal Rank Fusion contribution."""

        return 1.0 / (self.rrf_k + rank)