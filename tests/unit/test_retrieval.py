from app.ingestion.models import DocumentChunk
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import InMemoryVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "sample.txt",
) -> DocumentChunk:
    """Create a DocumentChunk using the project's existing model."""

    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
    )


class FakeEmbeddingGenerator:
    """Deterministic embeddings for retrieval unit tests."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embeddings[text]


def test_vector_store_adds_and_searches_chunks():
    store = InMemoryVectorStore()

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
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].score == 1.0


def test_vector_store_returns_most_similar_chunks():
    store = InMemoryVectorStore()

    store.add(
        chunk=make_chunk(
            "chunk-1",
            "Python programming",
        ),
        embedding=[1.0, 0.0, 0.0],
    )

    store.add(
        chunk=make_chunk(
            "chunk-2",
            "Database systems",
        ),
        embedding=[0.0, 1.0, 0.0],
    )

    store.add(
        chunk=make_chunk(
            "chunk-3",
            "Machine learning",
        ),
        embedding=[0.8, 0.6, 0.0],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[1].chunk.chunk_id == "chunk-3"
    assert results[0].score > results[1].score


def test_vector_store_respects_top_k():
    store = InMemoryVectorStore()

    for index in range(5):
        store.add(
            chunk=make_chunk(
                f"chunk-{index}",
                f"Document content {index}",
            ),
            embedding=[1.0, 0.0, 0.0],
        )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert len(results) == 3


def test_vector_store_rejects_invalid_top_k():
    store = InMemoryVectorStore()

    chunk = make_chunk(
        "chunk-1",
        "Some content",
    )

    store.add(
        chunk=chunk,
        embedding=[1.0, 0.0],
    )

    try:
        store.search(
            query_embedding=[1.0, 0.0],
            top_k=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_retriever_embeds_query_and_returns_results():
    store = InMemoryVectorStore()

    store.add(
        chunk=make_chunk(
            "chunk-1",
            "Python programming",
        ),
        embedding=[1.0, 0.0, 0.0],
    )

    store.add(
        chunk=make_chunk(
            "chunk-2",
            "Database systems",
        ),
        embedding=[0.0, 1.0, 0.0],
    )

    embedder = FakeEmbeddingGenerator(
        {
            "What is Python?": [1.0, 0.0, 0.0],
        }
    )

    retriever = Retriever(
        embedding_generator=embedder,
        vector_store=store,
    )

    results = retriever.retrieve(
        query="What is Python?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"


def test_retriever_rejects_empty_query():
    store = InMemoryVectorStore()

    embedder = FakeEmbeddingGenerator({})

    retriever = Retriever(
        embedding_generator=embedder,
        vector_store=store,
    )

    try:
        retriever.retrieve(
            "",
            top_k=5,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass