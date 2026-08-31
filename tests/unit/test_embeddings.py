from app.embeddings.embedder import EmbeddingGenerator


class FakeEncoder:
    """Deterministic encoder used for unit testing."""

    def encode(self, texts, normalize_embeddings=True):
        return [
            [float(len(text)), float(index + 1), 0.5]
            for index, text in enumerate(texts)
        ]


def test_embed_single_text():
    generator = EmbeddingGenerator(encoder=FakeEncoder())

    result = generator.embed_text("hello world")

    assert len(result) == 3
    assert result[0] == 11.0


def test_embed_multiple_texts():
    generator = EmbeddingGenerator(encoder=FakeEncoder())

    result = generator.embed_texts(
        [
            "first document",
            "second document",
        ]
    )

    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 3


def test_embedding_is_deterministic_for_same_input():
    generator = EmbeddingGenerator(encoder=FakeEncoder())

    first = generator.embed_text("same text")
    second = generator.embed_text("same text")

    assert first == second


def test_empty_text_is_rejected():
    generator = EmbeddingGenerator(encoder=FakeEncoder())

    try:
        generator.embed_text("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_empty_text_list_returns_empty_result():
    generator = EmbeddingGenerator(encoder=FakeEncoder())

    result = generator.embed_texts([])

    assert result == []