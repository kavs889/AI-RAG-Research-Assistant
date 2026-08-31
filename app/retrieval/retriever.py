from __future__ import annotations

from typing import Any

from app.retrieval.vector_store import InMemoryVectorStore, RetrievalResult


class Retriever:
    """Retrieve relevant document chunks for a user query."""

    def __init__(
        self,
        embedding_generator: Any,
        vector_store: InMemoryVectorStore,
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

        query_embedding = self.embedding_generator.embed_text(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )