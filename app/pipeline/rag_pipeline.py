from __future__ import annotations

from dataclasses import dataclass

from app.generation.generator import AnswerGenerator
from app.ingestion.models import DocumentChunk
from app.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class RAGResponse:
    """Final response returned by the RAG pipeline."""

    answer: str
    sources: list[DocumentChunk]
    retrieval_scores: list[float]


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Retrieve relevant context and generate an answer."""

        if not question or not question.strip():
            raise ValueError(
                "question cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        chunks = [
            result.chunk
            for result in results
        ]

        scores = [
            result.score
            for result in results
        ]

        answer = self.generator.generate(
            query=question,
            chunks=chunks,
        )

        return RAGResponse(
            answer=answer,
            sources=chunks,
            retrieval_scores=scores,
        )