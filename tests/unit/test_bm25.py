from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever


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


class TestBM25Retriever:
    """Unit tests for BM25 keyword retrieval."""

    def test_bm25_returns_relevant_keyword_match(self):
        chunks = [
            make_chunk(
                "chunk-1",
                "Python is a programming language.",
            ),
            make_chunk(
                "chunk-2",
                "SQL is used for relational databases.",
            ),
            make_chunk(
                "chunk-3",
                "Kubernetes manages containerized workloads.",
            ),
        ]

        retriever = BM25Retriever(chunks)

        results = retriever.search(
            "Python programming",
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chunk-1"

    def test_bm25_respects_top_k(self):
        chunks = [
            make_chunk("chunk-1", "Python programming language"),
            make_chunk("chunk-2", "Python data engineering"),
            make_chunk("chunk-3", "Python machine learning"),
            make_chunk("chunk-4", "SQL database systems"),
        ]

        retriever = BM25Retriever(chunks)

        results = retriever.search(
            "Python",
            top_k=2,
        )

        assert len(results) == 2

    def test_bm25_returns_scores(self):
        chunks = [
            make_chunk(
                "chunk-1",
                "Python Python Python programming",
            ),
            make_chunk(
                "chunk-2",
                "SQL database systems",
            ),
            make_chunk(
                "chunk-3",
                "Kubernetes container orchestration",
            ),
        ]

        retriever = BM25Retriever(chunks)

        results = retriever.search(
            "Python",
            top_k=3,
        )

        assert len(results) == 3
        assert results[0].chunk.chunk_id == "chunk-1"
        assert results[0].score > results[1].score

    def test_bm25_rejects_empty_query(self):
        chunks = [
            make_chunk(
                "chunk-1",
                "Python programming",
            )
        ]

        retriever = BM25Retriever(chunks)

        try:
            retriever.search("", top_k=5)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_bm25_handles_empty_collection(self):
        retriever = BM25Retriever([])

        results = retriever.search(
            "Python",
            top_k=5,
        )

        assert results == []