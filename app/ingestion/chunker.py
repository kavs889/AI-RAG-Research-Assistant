from __future__ import annotations

import hashlib

from app.ingestion.models import DocumentChunk


class TextChunker:
    """Split documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, source: str) -> list[DocumentChunk]:
        """Split text into deterministic overlapping DocumentChunk objects."""

        if not text:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.overlap

        start = 0
        chunk_index = 0

        while start < len(text):
            chunk_text = text[start:start + self.chunk_size]

            chunk_id = self._create_chunk_id(
                source=source,
                chunk_index=chunk_index,
                text=chunk_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    metadata={
                        "chunk_index": chunk_index,
                        "start": start,
                        "end": start + len(chunk_text),
                    },
                )
            )

            if start + self.chunk_size >= len(text):
                break

            start += step
            chunk_index += 1

        return chunks

    @staticmethod
    def _create_chunk_id(
        source: str,
        chunk_index: int,
        text: str,
    ) -> str:
        """Create a deterministic ID for a chunk."""

        payload = f"{source}:{chunk_index}:{text}".encode("utf-8")

        return hashlib.sha256(payload).hexdigest()[:16]