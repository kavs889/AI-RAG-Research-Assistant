from __future__ import annotations


SYSTEM_INSTRUCTIONS = """You are a research assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not invent information.
2. Do not use information that is not present in the context.
3. If the context does not contain enough information, clearly say so.
4. Keep the answer accurate, concise, and useful.
5. When appropriate, explain the answer using the evidence from the context.
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """Build a grounded RAG prompt from a question and retrieved context."""

    if not question or not question.strip():
        raise ValueError("question cannot be empty")

    if not context or not context.strip():
        raise ValueError("context cannot be empty")

    return f"""{SYSTEM_INSTRUCTIONS}

Context:
{context.strip()}

Question:
{question.strip()}

Answer:
"""