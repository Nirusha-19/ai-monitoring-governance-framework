"""
The escalation rules engine takes one pipeline result and decides whether it needs review. Thresholds and the reasoning behind them
are explained in the README's Escalation Engine section.
"""

THRESHOLDS = {
    "rerank_score": -6.0,
    "answer_relevance_score": 0.72,
}


def evaluate(result: dict) -> dict:
    checks = {
        "rerank_score": {
            "value": result["rerank_score"],
            "threshold": THRESHOLDS["rerank_score"],
            "passed": result["rerank_score"] >= THRESHOLDS["rerank_score"],
        },
        "answer_relevance_score": {
            "value": result["answer_relevance_score"],
            "threshold": THRESHOLDS["answer_relevance_score"],
            "passed": result["answer_relevance_score"] >= THRESHOLDS["answer_relevance_score"],
        },
        "is_refusal": {
            "value": result["is_refusal"],
            "passed": not result["is_refusal"],
        },
    }

    informational = {
        "lexical_match_score": result["lexical_match_score"],
        "semantic_consistency_score": result["semantic_consistency_score"],
    }

    failed_checks = [name for name, c in checks.items() if not c["passed"]]
    decision = "flagged_for_review" if failed_checks else "served_normally"

    return {
        "decision": decision,
        "checks": checks,
        "informational": informational,
        "failed_checks": failed_checks,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from rag.generate import run_pipeline

    for q in [
        "How do I reset my email password?",
        "My VPN keeps disconnecting, how do I fix it?",
        "What's the best recipe for chocolate cake?",
    ]:
        result = run_pipeline(q)
        decision = evaluate(result)
        print(f"\n{'=' * 60}")
        print(f"Question: {q}")
        print(f"Decision: {decision['decision']}")
        for name, c in decision["checks"].items():
            status = "PASS" if c["passed"] else "FAIL"
            print(f"  [{status}] {name}: {c['value']}")
        print(f"  (informational) lexical_match: {decision['informational']['lexical_match_score']:.3f}")
        print(f"  (informational) semantic_consistency: {decision['informational']['semantic_consistency_score']:.3f}")
