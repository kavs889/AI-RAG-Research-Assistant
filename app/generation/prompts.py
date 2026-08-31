from __future__ import annotations

from app.ingestion.models import DocumentChunk


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """Build a grounded RAG prompt."""

    if not question or not question.strip():
        raise ValueError("question cannot be empty")

    if not context or not context.strip():
        raise ValueError("context cannot be empty")

    return f"""You are a research assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say that the
information is not available in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""


def build_context(chunks: list[DocumentChunk]) -> str:
    """Convert document chunks into a single context string."""

    if not chunks:
        raise ValueError("chunks cannot be empty")

    context_parts: list[str] = []

    for chunk in chunks:
        if chunk.text and chunk.text.strip():
            context_parts.append(
                chunk.text.strip()
            )

    context = "\n\n".join(context_parts)

    if not context:
        raise ValueError("context cannot be empty")

    return context