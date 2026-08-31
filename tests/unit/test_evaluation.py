from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)
from app.evaluation.evaluator import RetrievalEvaluator


def test_precision_at_k():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = {"doc-1", "doc-3"}

    score = precision_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=3,
    )

    assert score == 2 / 3


def test_recall_at_k():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = {"doc-1", "doc-3", "doc-4"}

    score = recall_at_k(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=3,
    )

    assert score == 2 / 3


def test_reciprocal_rank():
    retrieved = [
        "doc-5",
        "doc-3",
        "doc-1",
    ]

    relevant = {"doc-1"}

    score = reciprocal_rank(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
    )

    assert score == 1 / 3


def test_reciprocal_rank_returns_zero_when_no_match():
    retrieved = ["doc-5", "doc-6"]
    relevant = {"doc-1"}

    score = reciprocal_rank(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
    )

    assert score == 0.0


def test_ndcg_at_k():
    retrieved = [
        "doc-1",
        "doc-2",
        "doc-3",
    ]

    relevant = {
        "doc-1": 3,
        "doc-2": 2,
        "doc-3": 0,
    }

    score = ndcg_at_k(
        retrieved_ids=retrieved,
        relevance_scores=relevant,
        k=3,
    )

    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_metrics_handle_empty_results():
    assert (
        precision_at_k(
            retrieved_ids=[],
            relevant_ids={"doc-1"},
            k=5,
        )
        == 0.0
    )

    assert (
        recall_at_k(
            retrieved_ids=[],
            relevant_ids={"doc-1"},
            k=5,
        )
        == 0.0
    )

    assert (
        reciprocal_rank(
            retrieved_ids=[],
            relevant_ids={"doc-1"},
        )
        == 0.0
    )


def test_metrics_reject_invalid_k():
    try:
        precision_at_k(
            retrieved_ids=["doc-1"],
            relevant_ids={"doc-1"},
            k=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        recall_at_k(
            retrieved_ids=["doc-1"],
            relevant_ids={"doc-1"},
            k=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_retrieval_evaluator_evaluates_results():
    evaluator = RetrievalEvaluator()

    retrieved = [
        "doc-1",
        "doc-2",
        "doc-3",
    ]

    relevant = {
        "doc-1",
        "doc-3",
    }

    result = evaluator.evaluate(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=3,
    )

    assert result.precision == 2 / 3
    assert result.recall == 1.0
    assert result.mrr == 1.0
    assert 0.0 <= result.ndcg <= 1.0


def test_retrieval_evaluator_returns_metrics_for_multiple_k():
    evaluator = RetrievalEvaluator()

    retrieved = [
        "doc-1",
        "doc-2",
        "doc-3",
        "doc-4",
    ]

    relevant = {
        "doc-1",
        "doc-4",
    }

    results = evaluator.evaluate_at_k_values(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k_values=[1, 2, 4],
    )

    assert set(results.keys()) == {1, 2, 4}

    assert results[1].precision == 1.0
    assert results[1].recall == 0.5

    assert results[2].precision == 0.5
    assert results[2].recall == 0.5

    assert results[4].precision == 0.5
    assert results[4].recall == 1.0