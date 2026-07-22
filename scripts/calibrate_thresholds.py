"""
Runs a sample from the golden set through the full pipeline and reports the real score distribution for known-good questions.
This is the actual calibration step, thresholds get set from these real scores, not guessed and not based on two hand-picked examples.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import GOLDEN_SET_PATH, OUTPUTS_DIR
from rag.generate import run_pipeline

SAMPLE_SIZE = 150
random.seed(7)


def load_golden_set():
    pairs = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def summarize(name, values):
    values = sorted(values)
    n = len(values)
    print(f"\n{name}:")
    print(f"  min:    {values[0]:.3f}")
    print(f"  p10:    {values[int(n * 0.10)]:.3f}")
    print(f"  median: {values[n // 2]:.3f}")
    print(f"  p90:    {values[int(n * 0.90)]:.3f}")
    print(f"  max:    {values[-1]:.3f}")


def main():
    golden = load_golden_set()
    sample = random.sample(golden, min(SAMPLE_SIZE, len(golden)))
    print(f"Running {len(sample)} golden-set questions through the full pipeline...")
    print(f"(These questions were held out and never indexed, so this is a")
    print(f" genuine test of retrieval + generation quality, not a lookup")
    print(f" of memorized content.)\n")

    results = []
    for i, pair in enumerate(sample):
        result = run_pipeline(pair["question"])
        results.append(result)
        print(f"  [{i + 1}/{len(sample)}] rerank={result['rerank_score']:.2f}  "
              f"lexical={result['lexical_match_score']:.2f}  "
              f"semantic={result['semantic_consistency_score']:.2f}  "
              f"relevance={result['answer_relevance_score']:.2f}  "
              f"source={result['source_used']}  refusal={result['is_refusal']}")

    print("\n" + "=" * 60)
    print("REAL score distributions for known-good, in-scope questions:")
    summarize("Rerank score", [r["rerank_score"] for r in results])
    summarize("Lexical match score", [r["lexical_match_score"] for r in results])
    summarize("Semantic consistency score", [r["semantic_consistency_score"] for r in results])
    summarize("Answer relevance score", [r["answer_relevance_score"] for r in results])

    n = len(results)
    general_knowledge_count = sum(1 for r in results if r["source_used"] == "general_knowledge")
    refusal_count = sum(1 for r in results if r["is_refusal"])
    print(f"\nSource used general_knowledge: {general_knowledge_count}/{n} "
          f"({100 * general_knowledge_count / n:.1f}%)")
    print(f"Flagged as refusal:            {refusal_count}/{n} "
          f"({100 * refusal_count / n:.1f}%)")

    semantic_vals = [r["semantic_consistency_score"] for r in results]
    relevance_vals = [r["answer_relevance_score"] for r in results]
    semantic_spread = max(semantic_vals) - min(semantic_vals)
    relevance_spread = max(relevance_vals) - min(relevance_vals)
    print(f"\nSemantic consistency spread (max-min): {semantic_spread:.3f}")
    print(f"Answer relevance spread (max-min):     {relevance_spread:.3f}")
    print("(A larger spread suggests a signal that discriminates more between")
    print(" queries -- useful for deciding whether both signals earn their keep.)")

    out_path = os.path.join(OUTPUTS_DIR, "golden_set_calibration.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
