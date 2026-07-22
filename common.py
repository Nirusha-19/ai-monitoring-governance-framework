import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE_LLM_MODEL = "qwen3:8b" 

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "it_support_kb"

# Data
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
KB_DOCS_PATH = os.path.join(DATA_DIR, "kb_documents.jsonl")       
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_set.jsonl")      

# Split sizes
KB_FRACTION = 0.85       # 85% of pairs become the live knowledge base
GOLDEN_FRACTION = 0.15   # 15% held out, never indexed, used only for degradation testing

# Retrieval
TOP_K_RETRIEVE = 30      # bi-encoder stage, how many candidates Qdrant returns
TOP_K_RERANK = 1         # cross-encoder stage, how many reach the LLM after reranking

# Thresholds - placeholders, replaced with real values derived from the golden set (see scripts/calibrate_thresholds.py).
GROUNDEDNESS_THRESHOLD = None
LATENCY_P95_THRESHOLD_MS = None
HALLUCINATION_RATE_THRESHOLD = None

# Storage
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
TRACE_DB_PATH = os.path.join(OUTPUTS_DIR, "traces.db")
