from __future__ import annotations

from typing import Any

import chromadb

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_store import RetrievalResult


class ChromaVectorStore:
    """Persistent vector store backed by ChromaDB."""

    DEFAULT_COLLECTION = "rag_documents"

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = DEFAULT_COLLECTION,
        client: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if client is None:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
            )
        else:
            self.client = client

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> None:
        """Add or update a single document chunk."""

        if not embedding:
            raise ValueError("embedding cannot be empty")

        self.collection.upsert(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[
                {
                    "source": chunk.source,
                    **chunk.metadata,
                }
            ],
        )

    def add_many(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add or update multiple document chunks."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length"
            )

        if not chunks:
            return

        if any(not embedding for embedding in embeddings):
            raise ValueError(
                "embeddings cannot contain empty values"
            )

        metadatas = [
            {
                "source": chunk.source,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Search the vector store using cosine similarity."""

        if not query_embedding:
            raise ValueError(
                "query_embedding cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        search_results: list[RetrievalResult] = []

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=documents[index],
                source=metadata.get("source", ""),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key != "source"
                },
            )

            # Chroma returns cosine distance.
            # Convert distance to similarity so the rest of
            # the retrieval system uses a consistent score.
            score = 1.0 - distances[index]

            search_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                )
            )

        return search_results

    def delete(
        self,
        chunk_id: str,
    ) -> None:
        """Delete a chunk from the vector store."""

        self.collection.delete(
            ids=[chunk_id],
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self.collection.count()

    def clear(self) -> None:
        """Remove all chunks from the collection."""

        self.client.delete_collection(
            name=self.collection_name,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        