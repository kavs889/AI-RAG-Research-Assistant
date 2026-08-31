from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceResponse


def create_app(pipeline: Any) -> FastAPI:
    """Create the FastAPI application with an injected RAG pipeline."""

    app = FastAPI(
        title="AI RAG Research Assistant",
        description="Retrieval-Augmented Generation API",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return application health status."""

        return {"status": "ok"}

    @app.post(
        "/query",
        response_model=QueryResponse,
    )
    def query(request: QueryRequest) -> QueryResponse:
        """Answer a user question using the RAG pipeline."""

        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="question cannot be empty",
            )

        try:
            response = pipeline.query(
                question=request.question.strip(),
                top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        sources = [
            SourceResponse(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
            )
            for chunk in response.sources
        ]

        return QueryResponse(
            answer=response.answer,
            sources=sources,
            retrieval_scores=response.retrieval_scores,
        )

    return app