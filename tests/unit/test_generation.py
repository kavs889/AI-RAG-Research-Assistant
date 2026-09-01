from app.generation.generator import AnswerGenerator
from app.generation.prompts import build_rag_prompt


class FakeLLM:
    """Deterministic fake LLM for unit tests."""

    def __init__(self, response: str = "This is a generated answer.") -> None:
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_build_rag_prompt_contains_question_and_context():
    prompt = build_rag_prompt(
        question="What is Python?",
        context="Python is a programming language.",
    )

    assert "What is Python?" in prompt
    assert "Python is a programming language." in prompt


def test_build_rag_prompt_rejects_empty_question():
    try:
        build_rag_prompt(
            question="",
            context="Some context",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_build_rag_prompt_rejects_empty_context():
    try:
        build_rag_prompt(
            question="What is Python?",
            context="",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_answer_generator_generates_response():
    llm = FakeLLM(
        response="Python is a programming language."
    )

    generator = AnswerGenerator(llm=llm)

    result = generator.generate(
        question="What is Python?",
        context="Python is a programming language.",
    )

    assert result == "Python is a programming language."


def test_answer_generator_sends_rag_prompt_to_llm():
    llm = FakeLLM()

    generator = AnswerGenerator(llm=llm)

    generator.generate(
        question="What is Python?",
        context="Python is a programming language.",
    )

    assert llm.last_prompt is not None
    assert "What is Python?" in llm.last_prompt
    assert "Python is a programming language." in llm.last_prompt


def test_answer_generator_rejects_empty_question():
    llm = FakeLLM()

    generator = AnswerGenerator(llm=llm)

    try:
        generator.generate(
            question="",
            context="Some context",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_answer_generator_rejects_empty_context():
    llm = FakeLLM()

    generator = AnswerGenerator(llm=llm)

    try:
        generator.generate(
            question="What is Python?",
            context="",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass