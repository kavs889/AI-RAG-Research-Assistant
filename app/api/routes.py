from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.pipeline.rag_pipeline import RAGPipeline


def create_app(pipeline: RAGPipeline) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="AI RAG Research Assistant",
        description="Answer questions using a retrieval-augmented generation pipeline.",
        version="1.0.0",
    )

    @app.post(
        "/query",
        response_model=QueryResponse,
    )
    def query(request: QueryRequest) -> QueryResponse:
        """Answer a user question using the RAG pipeline."""

        # API-level validation.
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="question cannot be empty",
            )

        if request.top_k <= 0:
            raise HTTPException(
                status_code=400,
                detail="top_k must be greater than zero",
            )

        try:
            response = pipeline.query(
                question=request.question,
                top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return QueryResponse(
            answer=response.answer,
            sources=[
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                }
                for chunk in response.sources
            ],
            retrieval_scores=response.retrieval_scores,
        )

    return app