from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.embeddings.embedder import EmbeddingGenerator
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever
from app.storage.chroma_store import ChromaVectorStore


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a completed indexing operation."""

    documents_processed: int
    chunks_created: int
    vectors_indexed: int
    bm25_chunks_indexed: int


class DocumentIndexer:
    """Build searchable indexes from documents on disk."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: Any | None = None,
        persist_directory: str = "data/chroma",
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

        # Use persistent ChromaDB by default.
        # A custom vector store can still be injected for unit tests.
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=persist_directory,
        )

        self.bm25_retriever: BM25Retriever | None = None
        self.chunks: list[DocumentChunk] = []

    def index_directory(
        self,
        directory: str | Path,
    ) -> IndexingResult:
        """Index all supported documents in a directory."""

        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"directory does not exist: {directory_path}"
            )

        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"path is not a directory: {directory_path}"
            )

        files = sorted(
            path
            for path in directory_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        documents_processed = 0
        all_chunks: list[DocumentChunk] = []

        for file_path in files:
            document = self.loader.load(file_path)

            chunks = self.chunker.chunk_document(document)

            all_chunks.extend(chunks)
            documents_processed += 1

        if all_chunks:
            embeddings = self.embedding_generator.embed_texts(
                [chunk.text for chunk in all_chunks]
            )

            # Add embeddings to the persistent vector store.
            if hasattr(self.vector_store, "add_many"):
                self.vector_store.add_many(
                    chunks=all_chunks,
                    embeddings=embeddings,
                )
            else:
                # Keep compatibility with simpler/custom vector stores.
                for chunk, embedding in zip(
                    all_chunks,
                    embeddings,
                    strict=True,
                ):
                    self.vector_store.add(
                        chunk=chunk,
                        embedding=embedding,
                    )

        self.chunks = all_chunks

        # Build the keyword index from the current document set.
        self.bm25_retriever = BM25Retriever(
            chunks=all_chunks,
        )

        return IndexingResult(
            documents_processed=documents_processed,
            chunks_created=len(all_chunks),
            vectors_indexed=len(all_chunks),
            bm25_chunks_indexed=len(all_chunks),
        )

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Search the BM25 index after indexing documents."""

        if self.bm25_retriever is None:
            raise RuntimeError(
                "documents must be indexed before BM25 search"
            )

        return self.bm25_retriever.search(
            query=query,
            top_k=top_k,
        )

    def search_vectors(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Generate a query embedding and search the vector store."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedding_generator.embed_text(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def count_vectors(self) -> int:
        """Return the number of vectors stored."""

        if hasattr(self.vector_store, "count"):
            return self.vector_store.count()

        return 0