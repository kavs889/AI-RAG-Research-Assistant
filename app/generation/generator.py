from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

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
        """Generate an answer using text context or document chunks."""

        # Pipeline interface:
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

        # Generation test interface:
        # generate(question="...", context="...")
        if question is None:
            raise ValueError("question cannot be empty")

        return self.generator.generate(
            question=question,
            context=context or "",
        )


class _DefaultLLM:
    """OpenRouter LLM adapter used by the production RAG application."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured. "
                "Set it in your environment or .env file."
            )

        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=api_key,
        )

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-4o-mini",
        )

    def generate(self, prompt: str) -> str:
        """Generate a grounded response through OpenRouter."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "OpenRouter returned an empty response."
            )

        return answer.strip()