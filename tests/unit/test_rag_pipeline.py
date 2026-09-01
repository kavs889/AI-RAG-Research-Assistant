from dataclasses import dataclass

import pytest

from app.ingestion.models import DocumentChunk
from app.pipeline.rag_pipeline import RAGPipeline


@dataclass
class FakeRetrievalResult:
    chunk: DocumentChunk
    score: float


class FakeRetriever:
    """Deterministic retriever for pipeline tests."""

    def __init__(self, results):
        self.results = results
        self.received_query = None
        self.received_top_k = None

    def retrieve(self, query: str, top_k: int):
        self.received_query = query
        self.received_top_k = top_k
        return self.results[:top_k]


class FakeGenerator:
    """Deterministic generator for pipeline tests."""

    def __init__(self):
        self.received_query = None
        self.received_chunks = None

    def generate(self, query: str, chunks):
        self.received_query = query
        self.received_chunks = chunks

        return "Python is a programming language."


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source="sample.txt",
        metadata={},
    )


def test_pipeline_retrieves_context_and_generates_answer():
    results = [
        FakeRetrievalResult(
            chunk=make_chunk(
                "chunk-1",
                "Python is a programming language.",
            ),
            score=0.95,
        ),
        FakeRetrievalResult(
            chunk=make_chunk(
                "chunk-2",
                "Python supports object-oriented programming.",
            ),
            score=0.80,
        ),
    ]

    retriever = FakeRetriever(results)
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    response = pipeline.query(
        question="What is Python?",
        top_k=2,
    )

    assert response.answer == "Python is a programming language."

    assert len(response.sources) == 2
    assert response.sources[0].chunk_id == "chunk-1"
    assert response.sources[1].chunk_id == "chunk-2"

    assert response.retrieval_scores == [0.95, 0.80]

    assert retriever.received_query == "What is Python?"
    assert retriever.received_top_k == 2

    assert generator.received_query == "What is Python?"
    assert len(generator.received_chunks) == 2


def test_pipeline_respects_top_k():
    results = [
        FakeRetrievalResult(
            chunk=make_chunk("chunk-1", "First document."),
            score=0.90,
        ),
        FakeRetrievalResult(
            chunk=make_chunk("chunk-2", "Second document."),
            score=0.80,
        ),
        FakeRetrievalResult(
            chunk=make_chunk("chunk-3", "Third document."),
            score=0.70,
        ),
    ]

    retriever = FakeRetriever(results)
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    response = pipeline.query(
        question="Tell me about the documents.",
        top_k=2,
    )

    assert len(response.sources) == 2
    assert len(response.retrieval_scores) == 2


def test_pipeline_rejects_empty_question():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(ValueError, match="question cannot be empty"):
        pipeline.query("", top_k=5)


def test_pipeline_rejects_whitespace_question():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(ValueError, match="question cannot be empty"):
        pipeline.query("   ", top_k=5)


def test_pipeline_rejects_invalid_top_k():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(ValueError, match="top_k"):
        pipeline.query(
            "What is Python?",
            top_k=0,
        )