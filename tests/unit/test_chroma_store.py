import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_store import RetrievalResult
from app.storage.chroma_store import ChromaVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "sample.txt",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        metadata={},
    )


def test_chroma_store_adds_and_searches_chunks(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunk = make_chunk(
        "chunk-1",
        "Python is a programming language.",
    )

    store.add(
        chunk=chunk,
        embedding=[1.0, 0.0, 0.0],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.text == (
        "Python is a programming language."
    )
    assert results[0].chunk.source == "sample.txt"
    assert results[0].score == pytest.approx(1.0)


def test_chroma_store_persists_data(tmp_path):
    persist_directory = str(tmp_path / "chroma")

    chunk = make_chunk(
        "chunk-1",
        "Persistent document content.",
    )

    first_store = ChromaVectorStore(
        persist_directory=persist_directory,
    )

    first_store.add(
        chunk=chunk,
        embedding=[1.0, 0.0, 0.0],
    )

    second_store = ChromaVectorStore(
        persist_directory=persist_directory,
    )

    results = second_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"


def test_chroma_store_add_many(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunks = [
        make_chunk("chunk-1", "Python programming"),
        make_chunk("chunk-2", "SQL database"),
        make_chunk("chunk-3", "Machine learning"),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.6, 0.0],
    ]

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert store.count() == 3

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"


def test_chroma_store_respects_top_k(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunks = [
        make_chunk(
            f"chunk-{index}",
            f"Document {index}",
        )
        for index in range(5)
    ]

    embeddings = [
        [1.0, 0.0, 0.0]
        for _ in range(5)
    ]

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert len(results) == 3


def test_chroma_store_rejects_empty_embedding(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunk = make_chunk(
        "chunk-1",
        "Some content",
    )

    with pytest.raises(
        ValueError,
        match="embedding cannot be empty",
    ):
        store.add(
            chunk=chunk,
            embedding=[],
        )


def test_chroma_store_rejects_invalid_top_k(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=0,
        )


def test_chroma_store_rejects_empty_query_embedding(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    with pytest.raises(
        ValueError,
        match="query_embedding",
    ):
        store.search(
            query_embedding=[],
            top_k=5,
        )


def test_chroma_store_delete(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunk = make_chunk(
        "chunk-1",
        "Python programming",
    )

    store.add(
        chunk=chunk,
        embedding=[1.0, 0.0, 0.0],
    )

    assert store.count() == 1

    store.delete("chunk-1")

    assert store.count() == 0


def test_chroma_store_clear(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
    )

    chunks = [
        make_chunk("chunk-1", "Python"),
        make_chunk("chunk-2", "SQL"),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert store.count() == 2

    store.clear()

    assert store.count() == 0