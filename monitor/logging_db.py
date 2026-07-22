"""
Writes a permanent, structured trace record for every pipeline run into a local SQLite database. This is the audit trail.
"""
import os
import sys
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import TRACE_DB_PATH, OUTPUTS_DIR

import structlog
import sqlite_utils

os.makedirs(OUTPUTS_DIR, exist_ok=True)

_connection = sqlite3.connect(TRACE_DB_PATH, check_same_thread=False)
_db = sqlite_utils.Database(_connection)
_log_table = _db["execution_logs"]


def _write_to_sqlite(logger, log_method, event_dict):
    _log_table.insert(event_dict, alter=True)
    return event_dict


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        _write_to_sqlite,
        structlog.processors.JSONRenderer(),
    ]
)

_logger = structlog.get_logger()


def log_trace(pipeline_result: dict, decision_result: dict):
    _logger.info(
        "pipeline_run",
        question=pipeline_result["question"],
        tools_used=pipeline_result["tools_used"],
        retrieved_chunk_id=pipeline_result["retrieved_chunk_id"],
        rerank_score=pipeline_result["rerank_score"],
        candidates_considered=pipeline_result["candidates_considered"],
        generated_answer=pipeline_result["generated_answer"],
        lexical_match_score=pipeline_result["lexical_match_score"],
        semantic_consistency_score=pipeline_result["semantic_consistency_score"],
        answer_relevance_score=pipeline_result["answer_relevance_score"],
        is_refusal=pipeline_result["is_refusal"],
        latency_ms_total=pipeline_result["latency_ms"]["total_ms"],
        latency_ms_retrieval=pipeline_result["latency_ms"]["retrieval_ms"],
        latency_ms_generation=pipeline_result["latency_ms"]["generation_ms"],
        latency_ms_scoring=pipeline_result["latency_ms"]["scoring_ms"],
        decision=decision_result["decision"],
        failed_checks=decision_result["failed_checks"],
    )


if __name__ == "__main__":
    from rag.generate import run_pipeline
    from monitor.rules import evaluate

    for q in [
        "How do I reset my email password?",
        "What's the best recipe for chocolate cake?",
    ]:
        result = run_pipeline(q)
        decision = evaluate(result)
        log_trace(result, decision)

        print(f"\n{'=' * 70}")
        print(f"Question:  {q}")
        print(f"Answer:    {result['generated_answer']}")
        print(f"\nDecision:  {decision['decision'].upper()}")
        print(f"Rerank score:               {result['rerank_score']:.3f}")
        print(f"Semantic consistency score: {result['semantic_consistency_score']:.3f}")
        print(f"Answer relevance score:     {result['answer_relevance_score']:.3f}")
        print(f"Is refusal:                 {result['is_refusal']}")
        print(f"(Lexical match, logged only: {result['lexical_match_score']:.3f})")
        if decision["failed_checks"]:
            print(f"Failed checks:              {', '.join(decision['failed_checks'])}")
        print(f"Latency:                    {result['latency_ms']['total_ms']:.0f} ms")

    print(f"\n{'=' * 70}")
    print(f"Confirmed logged to: {TRACE_DB_PATH}")
    print(f"Total records in database: {_log_table.count}")
