"""
Retrieval quality benchmark: Precision@K, Recall@K, and MRR, computed against the held-out golden set.
No human-annotated relevance judgments exist for this project. Instead, a candidate is labeled relevant if it's similar enough to the
golden set's known-correct answer. This is a silver-standard approximation, not ground truth, disclosed here and in the README.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import GOLDEN_SET_PATH, OUTPUTS_DIR, TOP_K_RETRIEVE
from rag.pipeline import _embedder, _qdrant, QDRANT_COLLECTION

SAMPLE_SIZE = 150
SILVER_LABEL_THRESHOLD = 0.75
K_VALUES = [1, 3, 5, 10]
random.seed(7)


def load_golden_set():
    pairs = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def cosine_similarity(vec_a, vec_b):
    import numpy as np
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_candidates_with_silver_labels(question: str, known_correct_answer: str):
    """
    Retrieves the top candidates via bi-encoder search only, no reranking, since this benchmarks retrieval quality itself. Labels each candidate
    relevant or not against the golden answer.
    """
    query_vector = _embedder.encode(question).tolist()
    hits = _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
    ).points

    correct_answer_embedding = _embedder.encode(known_correct_answer)

    labeled = []
    for hit in hits:
        candidate_embedding = _embedder.encode(hit.payload["answer"])
        similarity = cosine_similarity(candidate_embedding, correct_answer_embedding)
        labeled.append({
            "doc_id": hit.payload["doc_id"],
            "is_relevant": similarity >= SILVER_LABEL_THRESHOLD,
        })
    return labeled


def precision_at_k(labeled_candidates, k):
    top_k = labeled_candidates[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c["is_relevant"]) / len(top_k)


def recall_at_k(labeled_candidates, k):
    total_relevant = sum(1 for c in labeled_candidates if c["is_relevant"])
    if total_relevant == 0:
        return None  
    top_k = labeled_candidates[:k]
    found = sum(1 for c in top_k if c["is_relevant"])
    return found / total_relevant


def reciprocal_rank(labeled_candidates):
    for i, c in enumerate(labeled_candidates):
        if c["is_relevant"]:
            return 1.0 / (i + 1)
    return 0.0  


def main():
    golden = load_golden_set()
    sample = random.sample(golden, min(SAMPLE_SIZE, len(golden)))
    print(f"Benchmarking retrieval on {len(sample)} golden-set questions...")
    print(f"Silver-standard relevance threshold: {SILVER_LABEL_THRESHOLD}\n")

    precision_results = {k: [] for k in K_VALUES}
    recall_results = {k: [] for k in K_VALUES}
    reciprocal_ranks = []
    zero_relevant_count = 0

    for i, pair in enumerate(sample):
        labeled = get_candidates_with_silver_labels(pair["question"], pair["answer"])

        if sum(1 for c in labeled if c["is_relevant"]) == 0:
            zero_relevant_count += 1

        for k in K_VALUES:
            precision_results[k].append(precision_at_k(labeled, k))
            r = recall_at_k(labeled, k)
            if r is not None:
                recall_results[k].append(r)

        reciprocal_ranks.append(reciprocal_rank(labeled))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)} processed...")

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY BENCHMARK (silver-standard labels)")
    print("=" * 60)
    for k in K_VALUES:
        p = sum(precision_results[k]) / len(precision_results[k])
        r = sum(recall_results[k]) / len(recall_results[k]) if recall_results[k] else float("nan")
        print(f"  Precision@{k}: {p:.3f}   Recall@{k}: {r:.3f}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\n  MRR: {mrr:.3f}")
    print(f"\n  Queries with zero silver-labeled relevant candidates: "
          f"{zero_relevant_count}/{len(sample)} "
          f"({100 * zero_relevant_count / len(sample):.1f}%)")

    out_path = os.path.join(OUTPUTS_DIR, "retrieval_benchmark.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "precision_at_k": {k: sum(v) / len(v) for k, v in precision_results.items()},
            "recall_at_k": {k: sum(v) / len(v) if v else None for k, v in recall_results.items()},
            "mrr": mrr,
            "silver_label_threshold": SILVER_LABEL_THRESHOLD,
            "sample_size": len(sample),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
