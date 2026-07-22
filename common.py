import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE_LLM_MODEL = "qwen3:8b"  # served via Ollama, local

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "it_support_kb"

# Data
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
KB_DOCS_PATH = os.path.join(DATA_DIR, "kb_documents.jsonl")       # indexed into Qdrant
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_set.jsonl")      # held out, NEVER indexed

# Split sizes
KB_FRACTION = 0.85       # 85% of pairs become the live knowledge base
GOLDEN_FRACTION = 0.15   # 15% held out, never indexed, used only for degradation testing

# Retrieval
TOP_K_RETRIEVE = 30      # bi-encoder stage: how many candidates Qdrant returns
TOP_K_RERANK = 1         # cross-encoder stage: how many survive reranking to reach the LLM

# Thresholds -- placeholders only. These get replaced with real, empirically
# derived values after running the golden set once and observing the actual
# score distribution (see scripts/calibrate_thresholds.py).
GROUNDEDNESS_THRESHOLD = None
LATENCY_P95_THRESHOLD_MS = None
HALLUCINATION_RATE_THRESHOLD = None

# Storage
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
TRACE_DB_PATH = os.path.join(OUTPUTS_DIR, "traces.db")
