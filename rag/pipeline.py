"""
The retrieval half of the RAG pipeline. Given a question, this:

  1. Embeds it with bge-small-en-v1.5 (bi-encoder stage)
  2. Searches Qdrant for the top TOP_K_RETRIEVE candidates (fast, approximate)
  3. Reranks those candidates with a cross-encoder (slow, precise) to pick
     the single best match

Deliberately does NOT decide whether the match is "good enough" -- that judgment belongs to groundedness scoring, after generation, not here.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import (QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
                     EMBEDDING_MODEL, CROSS_ENCODER_MODEL, TOP_K_RETRIEVE)

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

_embedder = SentenceTransformer(EMBEDDING_MODEL)
_reranker = CrossEncoder(CROSS_ENCODER_MODEL)
_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def retrieve_and_rerank(query: str, top_k_retrieve: int = TOP_K_RETRIEVE):
    query_vector = _embedder.encode(query).tolist()
    hits = _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k_retrieve,
    ).points

    if not hits:
        return {
            "best_chunk_id": None,
            "best_answer": None,
            "rerank_score": 0.0,
            "candidates_considered": 0,
        }

    pairs = [(query, hit.payload["question"]) for hit in hits]
    rerank_scores = _reranker.predict(pairs)

    best_idx = int(rerank_scores.argmax())
    best_hit = hits[best_idx]

    return {
        "best_chunk_id": best_hit.payload["doc_id"],
        "best_question": best_hit.payload["question"],
        "best_answer": best_hit.payload["answer"],
        "rerank_score": float(rerank_scores[best_idx]),
        "candidates_considered": len(hits),
        "bi_encoder_top_score": float(hits[0].score),
    }


if __name__ == "__main__":
    test_query = "My VPN keeps disconnecting, how do I fix it?"
    result = retrieve_and_rerank(test_query)
    print(f"Query: {test_query}")
    print(f"Best match (doc {result['best_chunk_id']}, rerank score {result['rerank_score']:.3f}):")
    print(f"  Matched question: {result.get('best_question')}")
    print(f"  Retrieved answer: {result.get('best_answer')}")
