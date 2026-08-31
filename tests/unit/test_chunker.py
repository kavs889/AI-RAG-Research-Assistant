from app.ingestion.chunker import TextChunker
from app.ingestion.models import DocumentChunk


def test_chunker_splits_long_text():
    text = "A" * 1000

    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.split_text(
        text=text,
        source="sample.txt",
    )

    assert len(chunks) > 1
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_chunker_preserves_source_metadata():
    text = "This is a sample document for testing chunk metadata."

    chunker = TextChunker(chunk_size=20, overlap=5)
    chunks = chunker.split_text(
        text=text,
        source="research.pdf",
    )

    assert len(chunks) > 0
    assert all(chunk.source == "research.pdf" for chunk in chunks)


def test_chunker_assigns_deterministic_chunk_ids():
    text = "A" * 300

    chunker = TextChunker(chunk_size=100, overlap=20)

    first_run = chunker.split_text(
        text=text,
        source="sample.txt",
    )

    second_run = chunker.split_text(
        text=text,
        source="sample.txt",
    )

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]


def test_short_document_creates_one_chunk():
    text = "Short document."

    chunker = TextChunker(chunk_size=100, overlap=20)

    chunks = chunker.split_text(
        text=text,
        source="short.txt",
    )

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_document_returns_no_chunks():
    chunker = TextChunker(chunk_size=100, overlap=20)

    chunks = chunker.split_text(
        text="",
        source="empty.txt",
    )

    assert chunks == []


def test_invalid_chunk_configuration_is_rejected():
    try:
        TextChunker(chunk_size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass