from dataclasses import dataclass

import pytest

from app.indexing.indexer import DocumentIndexer
from app.ingestion.models import DocumentChunk


@dataclass
class FakeDocument:
    text: str
    source: str


class FakeLoader:
    """Deterministic document loader for tests."""

    def load(self, path):
        return FakeDocument(
            text=path.read_text(encoding="utf-8"),
            source=str(path),
        )


class FakeChunker:
    """Deterministic chunker for tests."""

    def chunk_document(self, document):
        return [
            DocumentChunk(
                chunk_id="chunk-1",
                text=document.text,
                source=document.source,
                metadata={},
            )
        ]


class FakeEmbeddingGenerator:
    """Deterministic embedding generator for tests."""

    def embed_texts(self, texts):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


class FakeVectorStore:
    """Simple vector store spy for tests."""

    def __init__(self):
        self.items = []

    def add(self, chunk, embedding):
        self.items.append(
            {
                "chunk": chunk,
                "embedding": embedding,
            }
        )


def test_index_directory_processes_supported_documents(tmp_path):
    (tmp_path / "document.txt").write_text(
        "Python is a programming language.",
        encoding="utf-8",
    )

    (tmp_path / "notes.md").write_text(
        "# Python\nPython is widely used.",
        encoding="utf-8",
    )

    (tmp_path / "ignored.csv").write_text(
        "id,value",
        encoding="utf-8",
    )

    vector_store = FakeVectorStore()

    indexer = DocumentIndexer(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_generator=FakeEmbeddingGenerator(),
        vector_store=vector_store,
    )

    result = indexer.index_directory(tmp_path)

    assert result.documents_processed == 2
    assert result.chunks_created == 2
    assert result.vectors_indexed == 2
    assert result.bm25_chunks_indexed == 2

    assert len(vector_store.items) == 2
    assert len(indexer.chunks) == 2
    assert indexer.bm25_retriever is not None


def test_index_directory_searches_bm25_after_indexing(tmp_path):
    (tmp_path / "python.txt").write_text(
        "Python programming language",
        encoding="utf-8",
    )

    indexer = DocumentIndexer(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_generator=FakeEmbeddingGenerator(),
        vector_store=FakeVectorStore(),
    )

    indexer.index_directory(tmp_path)

    results = indexer.search_bm25(
        query="Python",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.text == "Python programming language"


def test_index_directory_rejects_missing_directory(tmp_path):
    missing_directory = tmp_path / "does-not-exist"

    indexer = DocumentIndexer(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_generator=FakeEmbeddingGenerator(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(FileNotFoundError):
        indexer.index_directory(missing_directory)


def test_index_directory_rejects_file_path(tmp_path):
    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Some document",
        encoding="utf-8",
    )

    indexer = DocumentIndexer(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_generator=FakeEmbeddingGenerator(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(NotADirectoryError):
        indexer.index_directory(file_path)


def test_bm25_search_requires_indexing():
    indexer = DocumentIndexer(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        embedding_generator=FakeEmbeddingGenerator(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(RuntimeError, match="indexed"):
        indexer.search_bm25(
            query="Python",
            top_k=5,
        )