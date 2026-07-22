"""
Retrieve + rerank (pipeline.py) -> generate an answer -> analyze it along signals, instead of one blended "groundedness" score:

  0. Rerank score - how relevant the retrieved document is to the question (computed in pipeline.py, carried through here; this is
     the one signal computed BEFORE generation, not after)
  1. Lexical match - word overlap between answer and context (reranker,reused). Logged for diagnostic reference only - does not drive the
     escalation decision (see monitor/rules.py): calibration showed it swings widely (-10.6 to 9.9) even for known-good answers.
  2. Semantic consistency - meaning similarity, answer vs. context
  3. Answer relevance - meaning similarity, answer vs. the question
  4. Is refusal - did the model actually answer, or decline to help?

"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import BASE_LLM_MODEL
from rag.pipeline import retrieve_and_rerank, _reranker, _embedder

import ollama
import numpy as np

PROMPT_TEMPLATE = """You are an IT support assistant. Answer the user's question using \
only the context below. If the context does not contain relevant information, \
answer as best you can based on general IT support knowledge, but do not \
claim the context supports something it does not.

Context:
{context}

Question: {question}

Answer:"""

REFUSAL_MARKERS = [
    "i cannot provide", "cannot assist with", "unable to help with",
    "not able to provide", "i can't help with", "i cannot help with",
]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def generate_answer(question: str, context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = ollama.generate(
        model=BASE_LLM_MODEL,
        prompt=prompt,
        options={"temperature": 0}
    )
    return response["response"].strip()


def cosine_similarity(vec_a, vec_b) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def analyze_answer(question: str, answer: str, source_context: str) -> dict:
    lexical_match_score = float(_reranker.predict([(source_context, answer)])[0])

    answer_embedding = _embedder.encode(answer)
    context_embedding = _embedder.encode(source_context)
    semantic_consistency_score = cosine_similarity(answer_embedding, context_embedding)

    question_embedding = _embedder.encode(question)
    answer_relevance_score = cosine_similarity(answer_embedding, question_embedding)

    return {
        "lexical_match_score": lexical_match_score,
        "semantic_consistency_score": semantic_consistency_score,
        "answer_relevance_score": answer_relevance_score,
        "is_refusal": is_refusal(answer),
    }


def run_pipeline(question: str) -> dict:
    start = time.time()

    retrieval = retrieve_and_rerank(question)
    retrieval_time = time.time()

    context = retrieval.get("best_answer") or "(no relevant context found)"
    answer = generate_answer(question, context)
    generation_time = time.time()

    analysis = analyze_answer(question, answer, context)
    scoring_time = time.time()

    return {
        "question": question,
        "tools_used": ["qdrant_search", "cross_encoder_rerank", "qwen3_generate",
                        "lexical_match_scoring", "semantic_consistency_scoring",
                        "answer_relevance_scoring"],
        "retrieved_chunk_id": retrieval.get("best_chunk_id"),
        "rerank_score": retrieval.get("rerank_score"),
        "candidates_considered": retrieval.get("candidates_considered"),
        "generated_answer": answer,
        "lexical_match_score": analysis["lexical_match_score"],
        "semantic_consistency_score": analysis["semantic_consistency_score"],
        "answer_relevance_score": analysis["answer_relevance_score"],
        "is_refusal": analysis["is_refusal"],
        "latency_ms": {
            "retrieval_ms": round((retrieval_time - start) * 1000, 1),
            "generation_ms": round((generation_time - retrieval_time) * 1000, 1),
            "scoring_ms": round((scoring_time - generation_time) * 1000, 1),
            "total_ms": round((scoring_time - start) * 1000, 1),
        },
    }


if __name__ == "__main__":
    for q in [
        "How do I reset my email password?",
        "What's the best recipe for chocolate cake?",
    ]:
        result = run_pipeline(q)
        print(f"\n{'=' * 60}")
        print(f"Question: {result['question']}")
        print(f"Rerank score: {result['rerank_score']:.3f}  (candidates: {result['candidates_considered']})")
        print(f"Generated answer: {result['generated_answer']}")
        print(f"Semantic consistency score: {result['semantic_consistency_score']:.3f}")
        print(f"Answer relevance score: {result['answer_relevance_score']:.3f}")
        print(f"Is refusal: {result['is_refusal']}")
        print(f"(Lexical match score, logged only: {result['lexical_match_score']:.3f})")
        print(f"Latency: {result['latency_ms']['total_ms']} ms total")