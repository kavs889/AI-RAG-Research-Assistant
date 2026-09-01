from __future__ import annotations

from typing import Any

from app.retrieval.vector_store import RetrievalResult


class Retriever:
    """Retrieve relevant document chunks using vector similarity."""

    def __init__(
        self,
        embedding_generator: Any,
        vector_store: Any,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Embed a query and retrieve the most relevant chunks."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = self.embedding_generator.embed_text(
            query
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # ChromaVectorStore and InMemoryVectorStore both return
        # RetrievalResult objects. Keep this layer intentionally
        # simple so the vector backend can be swapped without
        # changing the retriever API.
        return results