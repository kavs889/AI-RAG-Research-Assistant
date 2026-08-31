from __future__ import annotations

from typing import Any


class EmbeddingGenerator:
    """Generate normalized vector embeddings for text."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        encoder: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder

    @property
    def encoder(self) -> Any:
        """Load the embedding model lazily."""

        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)

        return self._encoder

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text."""

        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        embeddings = self.encoder.encode(
            [text],
            normalize_embeddings=True,
        )

        return self._to_list(embeddings[0])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("texts cannot contain empty values")

        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
        )

        return [self._to_list(embedding) for embedding in embeddings]

    @staticmethod
    def _to_list(embedding: Any) -> list[float]:
        """Convert an embedding to a plain Python list."""

        if hasattr(embedding, "tolist"):
            return embedding.tolist()

        return list(embedding)