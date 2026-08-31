from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import DocumentChunk


class DocumentLoader:
    """Load supported documents and normalize them into DocumentChunk objects."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(self, file_path: str | Path) -> list[DocumentChunk]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)

        return self._load_text(path)

    def _load_pdf(self, path: Path) -> list[DocumentChunk]:
        reader = PdfReader(str(path))
        chunks: list[DocumentChunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            chunks.append(
                DocumentChunk(
                    text=text,
                    source=str(path),
                    chunk_id=f"{path.name}:page-{page_number}",
                    metadata={
                        "page": page_number,
                        "file_type": path.suffix.lower(),
                    },
                )
            )

        return chunks

    def _load_text(self, path: Path) -> list[DocumentChunk]:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            return []

        return [
            DocumentChunk(
                text=text,
                source=str(path),
                chunk_id=f"{path.name}:document",
                metadata={
                    "page": None,
                    "file_type": path.suffix.lower(),
                },
            )
        ]