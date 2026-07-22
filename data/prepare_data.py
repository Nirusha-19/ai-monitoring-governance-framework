"""
Downloads a real IT-support Q&A dataset and splits it into two non-overlapping groups: kb_documents.jsonl and golden_set.jsonl.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import (RAW_DATA_DIR, KB_DOCS_PATH, GOLDEN_SET_PATH,
                     KB_FRACTION, GOLDEN_FRACTION)

from datasets import load_dataset

random.seed(42)

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
TARGET_TOTAL_PAIRS = 30000  

def find_field(example, candidates):
    for c in candidates:
        if c in example:
            return c
    raise KeyError(f"None of {candidates} found. Available fields: {list(example.keys())}")


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(KB_DOCS_PATH), exist_ok=True)

    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  {len(ds)} total raw examples available.")

    first = ds[0]
    print(f"  Fields: {list(first.keys())}")
    # Confirmed real fields: 'subject', 'body' (the customer's message), 'answer' (the support agent's response).
    question_field = find_field(first, ["body", "question", "query"])
    answer_field = find_field(first, ["answer", "response", "resolution"])
    print(f"  Using '{question_field}' as question, '{answer_field}' as answer.")

    # Filter to English only, since the dataset is genuinely bilingual (en/de).
    if "language" in first:
        ds = ds.filter(lambda ex: ex["language"] == "en")
        print(f"  {len(ds)} examples remain after filtering to English only.")

    n = min(TARGET_TOTAL_PAIRS, len(ds))
    indices = random.sample(range(len(ds)), n)

    pairs = []
    for i in indices:
        ex = ds[i]
        q = str(ex[question_field]).strip()
        a = str(ex[answer_field]).strip()
        if q and a and q.lower() != "none" and len(a) > 10: 
            pairs.append({"id": f"doc_{i}", "question": q, "answer": a})

    random.shuffle(pairs)
    split_point = int(len(pairs) * KB_FRACTION)
    kb_pairs = pairs[:split_point]
    golden_pairs = pairs[split_point:]

    with open(KB_DOCS_PATH, "w") as f:
        for p in kb_pairs:
            f.write(json.dumps(p) + "\n")

    with open(GOLDEN_SET_PATH, "w") as f:
        for p in golden_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal usable pairs: {len(pairs)}")
    print(f"  Knowledge base (indexed):  {len(kb_pairs)} -> {KB_DOCS_PATH}")
    print(f"  Golden set (held out):     {len(golden_pairs)} -> {GOLDEN_SET_PATH}")
    print("\nConfirm these two files never overlap in 'id' before proceeding.")


if __name__ == "__main__":
    main()
