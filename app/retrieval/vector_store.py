from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.ingestion.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A document chunk returned by similarity search."""

    chunk: DocumentChunk
    score: float


class InMemoryVectorStore:
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self._items: list[tuple[DocumentChunk, list[float]]] = []

    def add(
        self,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> None:
        """Store a document chunk and its embedding."""

        if not embedding:
            raise ValueError("embedding cannot be empty")

        self._items.append((chunk, embedding))

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the top-k most similar document chunks."""

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scored_results: list[RetrievalResult] = []

        for chunk, embedding in self._items:
            if len(embedding) != len(query_embedding):
                raise ValueError(
                    "query and document embeddings must have the same dimension"
                )

            score = self._cosine_similarity(
                query_embedding,
                embedding,
            )

            scored_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                )
            )

        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return scored_results[:top_k]

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sqrt(
            sum(value * value for value in vector_a)
        )

        magnitude_b = sqrt(
            sum(value * value for value in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)