from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Normalized representation of a chunk used throughout the RAG pipeline."""

    text: str
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)