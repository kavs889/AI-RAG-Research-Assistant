from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import InMemoryVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "sample.txt",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
    )


class FakeEmbeddingGenerator:
    """Deterministic embeddings for hybrid retrieval tests."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embeddings[text]


def build_semantic_retriever() -> Retriever:
    store = InMemoryVectorStore()

    store.add(
        chunk=make_chunk(
            "chunk-1",
            "Python programming language",
        ),
        embedding=[1.0, 0.0, 0.0],
    )

    store.add(
        chunk=make_chunk(
            "chunk-2",
            "Machine learning algorithms",
        ),
        embedding=[0.8, 0.6, 0.0],
    )

    store.add(
        chunk=make_chunk(
            "chunk-3",
            "SQL database systems",
        ),
        embedding=[0.0, 1.0, 0.0],
    )

    embedder = FakeEmbeddingGenerator(
        {
            "Python programming": [1.0, 0.0, 0.0],
            "database query": [0.0, 1.0, 0.0],
        }
    )

    return Retriever(
        embedding_generator=embedder,
        vector_store=store,
    )


def build_bm25_retriever() -> BM25Retriever:
    chunks = [
        make_chunk(
            "chunk-1",
            "Python programming language",
        ),
        make_chunk(
            "chunk-2",
            "Machine learning algorithms",
        ),
        make_chunk(
            "chunk-3",
            "SQL database systems",
        ),
    ]

    return BM25Retriever(chunks)


def test_hybrid_retriever_returns_results():
    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    results = hybrid.retrieve(
        query="Python programming",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"


def test_hybrid_retriever_combines_both_retrievers():
    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    results = hybrid.retrieve(
        query="database query",
        top_k=3,
    )

    chunk_ids = [result.chunk.chunk_id for result in results]

    assert "chunk-3" in chunk_ids


def test_hybrid_retriever_respects_top_k():
    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    results = hybrid.retrieve(
        query="Python programming",
        top_k=1,
    )

    assert len(results) == 1


def test_hybrid_retriever_rejects_empty_query():
    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
    )

    try:
        hybrid.retrieve("", top_k=5)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rrf_promotes_documents_present_in_both_rankings():
    semantic = build_semantic_retriever()
    bm25 = build_bm25_retriever()

    hybrid = HybridRetriever(
        semantic_retriever=semantic,
        bm25_retriever=bm25,
        rrf_k=60,
    )

    results = hybrid.retrieve(
        query="Python programming",
        top_k=3,
    )

    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].score > results[1].score