"""
Unit tests for the escalation rules engine. Uses fabricated result dicts, no live model calls, no Qdrant, no network. Fast, deterministic.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from monitor.rules import evaluate, THRESHOLDS


def make_result(rerank_score=8.0, answer_relevance_score=0.9,
                 lexical_match_score=5.0, semantic_consistency_score=0.85,
                 is_refusal=False):
    """Builds a fabricated pipeline result dict for testing evaluate()
    in isolation, without running the actual pipeline."""
    return {
        "rerank_score": rerank_score,
        "answer_relevance_score": answer_relevance_score,
        "lexical_match_score": lexical_match_score,
        "semantic_consistency_score": semantic_consistency_score,
        "is_refusal": is_refusal,
    }


def test_healthy_result_is_served_normally():
    result = make_result()
    decision = evaluate(result)
    assert decision["decision"] == "served_normally"
    assert decision["failed_checks"] == []


def test_low_rerank_score_triggers_flag():
    result = make_result(rerank_score=THRESHOLDS["rerank_score"] - 1)
    decision = evaluate(result)
    assert decision["decision"] == "flagged_for_review"
    assert "rerank_score" in decision["failed_checks"]


def test_low_answer_relevance_triggers_flag():
    result = make_result(answer_relevance_score=THRESHOLDS["answer_relevance_score"] - 0.1)
    decision = evaluate(result)
    assert decision["decision"] == "flagged_for_review"
    assert "answer_relevance_score" in decision["failed_checks"]


def test_refusal_triggers_flag_even_with_good_scores():
    result = make_result(is_refusal=True)
    decision = evaluate(result)
    assert decision["decision"] == "flagged_for_review"
    assert "is_refusal" in decision["failed_checks"]


def test_boundary_value_at_exact_threshold_passes():
    # Values exactly AT the threshold should pass (>=, not >).
    result = make_result(
        rerank_score=THRESHOLDS["rerank_score"],
        answer_relevance_score=THRESHOLDS["answer_relevance_score"],
    )
    decision = evaluate(result)
    assert decision["decision"] == "served_normally"


def test_lexical_match_never_triggers_on_its_own():
    # lexical_match_score is logged, not a trigger, regardless of how extreme its value is.
    result = make_result(lexical_match_score=-50.0)
    decision = evaluate(result)
    assert decision["decision"] == "served_normally"
    assert "lexical_match_score" not in decision["failed_checks"]


def test_semantic_consistency_never_triggers_on_its_own():
    result = make_result(semantic_consistency_score=0.0)
    decision = evaluate(result)
    assert decision["decision"] == "served_normally"
    assert "semantic_consistency_score" not in decision["failed_checks"]


def test_informational_signals_are_present_in_output():
    result = make_result(lexical_match_score=3.3, semantic_consistency_score=0.65)
    decision = evaluate(result)
    assert decision["informational"]["lexical_match_score"] == 3.3
    assert decision["informational"]["semantic_consistency_score"] == 0.65


def test_multiple_failures_all_appear_in_failed_checks():
    result = make_result(
        rerank_score=THRESHOLDS["rerank_score"] - 1,
        answer_relevance_score=THRESHOLDS["answer_relevance_score"] - 0.1,
        is_refusal=True,
    )
    decision = evaluate(result)
    assert decision["decision"] == "flagged_for_review"
    assert len(decision["failed_checks"]) == 3
