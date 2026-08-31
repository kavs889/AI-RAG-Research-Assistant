from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.generation.prompts import build_rag_prompt


@dataclass(frozen=True)
class GenerationResult:
    """Result produced by the answer generation layer."""

    answer: str


class RAGGenerator:
    """Generate answers from a question and text context."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer from a question and text context."""

        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        if not context or not context.strip():
            raise ValueError("context cannot be empty")

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        return self.llm.generate(prompt)


class AnswerGenerator:
    """Application-level answer generator for retrieved document chunks."""

    def __init__(self, llm: Any | None = None) -> None:
        if llm is None:
            llm = _DefaultLLM()

        self.generator = RAGGenerator(llm=llm)

    def generate(
        self,
        question: str | None = None,
        context: str | None = None,
        *,
        query: str | None = None,
        chunks: list[Any] | None = None,
    ) -> str:
        """Generate an answer using either text context or document chunks."""

        # Support the pipeline interface:
        # generate(query="...", chunks=[...])
        if query is not None or chunks is not None:
            actual_question = query if query is not None else question

            if not actual_question or not actual_question.strip():
                raise ValueError("question cannot be empty")

            if not chunks:
                raise ValueError("context cannot be empty")

            context_parts: list[str] = []

            for chunk in chunks:
                if hasattr(chunk, "text"):
                    text = chunk.text
                else:
                    text = str(chunk)

                if text and text.strip():
                    context_parts.append(text.strip())

            actual_context = "\n\n".join(context_parts)

            if not actual_context:
                raise ValueError("context cannot be empty")

            return self.generator.generate(
                question=actual_question,
                context=actual_context,
            )

        # Support the generation unit-test interface:
        # generate(question="...", context="...")
        if question is None:
            raise ValueError("question cannot be empty")

        return self.generator.generate(
            question=question,
            context=context or "",
        )


class _DefaultLLM:
    """Small deterministic LLM adapter used for local development."""

    def generate(self, prompt: str) -> str:
        """Return the supplied prompt."""

        return prompt