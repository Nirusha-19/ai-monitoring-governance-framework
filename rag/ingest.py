"""
Reads kb_documents.jsonl, embeds every document with bge-small-en-v1.5, and uploads all of them into a Qdrant collection. This builds the actual
searchable vector index the retrieval pipeline queries against.

Run this once, after prepare_data.py, and before the RAG pipeline or API.
Re-run it if the knowledge base changes.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import (KB_DOCS_PATH, QDRANT_HOST, QDRANT_PORT,
                     QDRANT_COLLECTION, EMBEDDING_MODEL)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

BATCH_SIZE = 256


def load_documents():
    docs = []
    with open(KB_DOCS_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    print(f"Loading documents from {KB_DOCS_PATH}...")
    docs = load_documents()
    print(f"  {len(docs)} documents to embed and index.")

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    print(f"  Collection '{QDRANT_COLLECTION}' created.")

    print("Embedding and uploading in batches...")
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        texts = [d["question"] for d in batch]
        vectors = model.encode(texts, show_progress_bar=False)

        points = [
            PointStruct(
                id=i + start,
                vector=vectors[i].tolist(),
                payload={
                    "doc_id": batch[i]["id"],
                    "question": batch[i]["question"],
                    "answer": batch[i]["answer"],
                },
            )
            for i in range(len(batch))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  Indexed {min(start + BATCH_SIZE, len(docs))}/{len(docs)}")

    count = client.count(collection_name=QDRANT_COLLECTION).count
    print(f"\nDone. Qdrant collection '{QDRANT_COLLECTION}' now holds {count} vectors.")


if __name__ == "__main__":
    main()
