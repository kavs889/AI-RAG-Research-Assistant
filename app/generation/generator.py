from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.generation.prompts import build_rag_prompt


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by the RAG generation layer."""

    answer: str
    model: str | None = None
    prompt: str | None = None


class AnswerGenerator:
    """Generate answers using retrieved context."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        llm: Any | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.llm = llm
        self.model_name = model_name

    def _get_llm(self) -> Any:
        """Create the OpenAI client lazily when no LLM is injected."""

        if self.llm is None:
            from openai import OpenAI

            self.llm = OpenAI()

        return self.llm

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer using the supplied retrieval context."""

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        llm = self._get_llm()

        # Support the deterministic fake LLM used by unit tests.
        if hasattr(llm, "generate"):
            response = llm.generate(prompt)

            if not isinstance(response, str):
                raise TypeError("LLM generate() must return a string")

            return response.strip()

        # OpenAI client path.
        response = llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Answer questions using only the provided context. "
                        "If the context does not contain enough information, "
                        "say that the information is not available in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise ValueError("LLM returned an empty response")

        return answer.strip()


class RAGGenerator(AnswerGenerator):
    """Backward-compatible name for the RAG generation component."""

    pass