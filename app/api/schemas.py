from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the RAG query endpoint."""

    question: str = Field(
        ...,
        description="Question to answer using the RAG system",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of retrieval results",
    )


class SourceResponse(BaseModel):
    """Source metadata returned with an answer."""

    chunk_id: str
    source: str


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint."""

    answer: str
    sources: list[SourceResponse]
    retrieval_scores: list[float]