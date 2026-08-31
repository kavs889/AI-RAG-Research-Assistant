from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_store import RetrievalResult


class BM25Retriever:
    """Keyword-based document retrieval using BM25."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks

        tokenized_documents = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the top-k keyword-relevant chunks."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                )
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into lowercase word tokens."""

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )