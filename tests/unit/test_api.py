from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.api.routes import create_app
from app.ingestion.models import DocumentChunk
from app.pipeline.rag_pipeline import RAGResponse


@dataclass
class FakePipeline:
    """Deterministic RAG pipeline for API tests."""

    def query(
        self,
        question: str,
        top_k: int,
    ) -> RAGResponse:
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            text="Python is a programming language.",
            source="sample.txt",
            metadata={},
        )

        return RAGResponse(
            answer="Python is a programming language.",
            sources=[chunk],
            retrieval_scores=[0.95],
        )


def create_test_client() -> TestClient:
    app = create_app(pipeline=FakePipeline())
    return TestClient(app)


def test_query_endpoint_returns_answer_and_sources():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "What is Python?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Python is a programming language."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "chunk-1"
    assert data["sources"][0]["source"] == "sample.txt"
    assert data["retrieval_scores"] == [0.95]


def test_query_endpoint_uses_default_top_k():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "What is Python?",
        },
    )

    assert response.status_code == 200


def test_query_endpoint_rejects_empty_question():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 400


def test_query_endpoint_rejects_whitespace_question():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 400


def test_query_endpoint_rejects_invalid_top_k():
    client = create_test_client()

    response = client.post(
        "/query",
        json={
            "question": "What is Python?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422