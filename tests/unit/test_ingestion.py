from io import BytesIO

from pypdf import PdfWriter

from pathlib import Path

import pytest

from app.ingestion.loader import DocumentLoader
from app.ingestion.models import DocumentChunk


def test_document_chunk_creation() -> None:
    chunk = DocumentChunk(
        text="Retrieval augmented generation combines retrieval with generation.",
        source="sample.pdf",
        chunk_id="sample.pdf:0",
        metadata={"page": 1},
    )

    assert chunk.text.startswith("Retrieval")
    assert chunk.source == "sample.pdf"
    assert chunk.chunk_id == "sample.pdf:0"
    assert chunk.metadata["page"] == 1


def test_load_text_document(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text(
        "Retrieval augmented generation improves question answering.",
        encoding="utf-8",
    )

    chunks = DocumentLoader().load(document)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Retrieval")
    assert chunks[0].metadata["file_type"] == ".txt"


def test_load_markdown_document(tmp_path: Path) -> None:
    document = tmp_path / "sample.md"
    document.write_text(
        "# RAG\n\nRetrieval augmented generation uses retrieved context.",
        encoding="utf-8",
    )

    chunks = DocumentLoader().load(document)

    assert len(chunks) == 1
    assert "# RAG" in chunks[0].text
    assert chunks[0].metadata["file_type"] == ".md"


def test_reject_unsupported_file_type(tmp_path: Path) -> None:
    document = tmp_path / "sample.csv"
    document.write_text("id,name\n1,test", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load(document)
def test_load_pdf_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with pdf_path.open("wb") as file:
        writer.write(file)

    chunks = DocumentLoader().load(pdf_path)

    assert isinstance(chunks, list)   