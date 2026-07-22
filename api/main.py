"""
FastAPI service exposing the full pipeline as a real, callable endpoint.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from rag.generate import run_pipeline
from monitor.rules import evaluate
from monitor.logging_db import log_trace

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Monitoring & Governance Framework")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    decision: str
    failed_checks: list[str]
    rerank_score: float
    semantic_consistency_score: float
    answer_relevance_score: float
    is_refusal: bool
    lexical_match_score: float
    latency_ms: float


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = run_pipeline(request.question)
    decision = evaluate(result)
    log_trace(result, decision)

    return AskResponse(
        question=result["question"],
        answer=result["generated_answer"],
        decision=decision["decision"],
        failed_checks=decision["failed_checks"],
        rerank_score=result["rerank_score"],
        semantic_consistency_score=result["semantic_consistency_score"],
        answer_relevance_score=result["answer_relevance_score"],
        is_refusal=result["is_refusal"],
        lexical_match_score=result["lexical_match_score"],
        latency_ms=result["latency_ms"]["total_ms"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
