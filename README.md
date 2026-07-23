# RAG Monitoring and Governance Framework

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-teal) ![Qdrant](https://img.shields.io/badge/Qdrant-red) ![Grafana](https://img.shields.io/badge/Grafana-orange) ![Ollama](https://img.shields.io/badge/Ollama-black)

Wraps a standard RAG pipeline in a multi-signal analysis layer and an escalation engine that catches silent failure modes (weak retrieval, off-topic answers, refusals) a RAG pipeline never surfaces on its own, then makes that visible in a live Grafana dashboard.

## 🎯 What Is This?

A RAG pipeline, by itself, returns an answer with no indication of whether that answer is well grounded, actually addresses the question, or is a refusal. There's no built-in way to know if it's wrong. This project builds that layer. Five signals scored on every response, an escalation engine that flags the risky ones, and a live Grafana dashboard with three panels, decisions over time, flag rate, and latency by stage, demonstrated on a self hosted IT support RAG agent running entirely on local hardware.

## 🔍 What It Does?

- Runs a full local RAG pipeline, using bi-encoder retrieval, cross-encoder reranking, and Qwen3 8B generation, with no cloud APIs
- Scores every generated answer on five signals, covering how relevant the retrieved source is, how relevant the answer is to the question, whether the model refused to answer, word overlap with the source, and meaning consistency with the source
- Flags low-confidence or risky answers automatically, using thresholds derived from real test data, not guessed
- Logs every request, decision, and score permanently to SQLite, visualized live in Grafana
- Exposes the whole pipeline as a real FastAPI service, with interactive docs

## 📊 Dataset

`Tobi-Bueck/customer-support-tickets`, from Hugging Face.

- English-filtered subset (the raw dataset is bilingual, English and German)
- 24,010 knowledge-base documents indexed in Qdrant
- 4,238-question golden set, held out and never indexed, used for retrieval benchmarking and threshold calibration
- ~20% of the knowledge base relates to healthcare-compliance and medical-data-security contexts (HIPAA, EMR/PACS, patient data), alongside general IT tickets, reflecting the kind of support queue a healthcare organization, or any company handling health-adjacent data, would actually field.

## 🖥️ How It Works, End to End

**Retrieval:** A query is embedded with `bge-small-en-v1.5` (bi-encoder) and searched against Qdrant for the top 30 candidates. `ms-marco-MiniLM-L6-v2` (cross-encoder) reranks those candidates for precision, producing `rerank_score`.

**Generation:** The best matched candidate and the question are passed to Qwen3 8B, running locally via Ollama, with `temperature=0` for reproducible answers.

**Scoring:** Every generated answer is analyzed on five signals, `rerank_score`, `answer_relevance_score`, `is_refusal`, `lexical_match_score`, and `semantic_consistency_score`, before any decision is made.

**Escalation:** The rules engine checks three of those five signals against calibrated thresholds and classifies the response as `served_normally` or `flagged_for_review`.

**Logging:** Every field from every request, structlog formatted, is written to a SQLite `execution_logs` table.

**Dashboarding:** Grafana reads that same SQLite file live, showing decisions over time, flag rate, and latency by pipeline stage.

## 📈 The Multi-Signal Analysis Layer

| Signal | Role | What it measures |
|---|---|---|
| `rerank_score` | Active trigger | Cross-encoder relevance between the query and the retrieved chunk |
| `answer_relevance_score` | Active trigger | Embedding similarity between the answer and the original question |
| `is_refusal` | Active trigger | Whether the model declined to answer, found via specific phrases |
| `lexical_match_score` | Logged only | Word overlap between answer and source, via the reranker |
| `semantic_consistency_score` | Logged only | Embedding similarity between the answer and the retrieved context |

Three signals are active escalation triggers; two are logged but don't independently flag a response. This split is the result of testing, not the starting design. Two earlier single score groundedness approaches, reusing the reranker alone and a dedicated NLI entailment model, were tried and set aside after each produced uninformative scores on real test cases.

## ⚙️ Escalation Engine

Responses are classified as `served_normally` or `flagged_for_review`, based on the following.

- `rerank_score` at or above **-6.0**
- `answer_relevance_score` at or above **0.72**
- `is_refusal` must be `False`

These thresholds were derived twice. Initial calibration ran against the 150-question golden set, producing a first set of thresholds (`rerank_score` at or above 5.96). Those thresholds flagged 100% of questions in a live test-harness run using naturally-phrased questions, since golden-set questions are historical ticket text copied word for word, which retrieves more strongly than freshly-worded questions asking about the same topics. Thresholds were re-derived from the live-traffic results instead, and `semantic_consistency_score` was dropped as a trigger after the same test showed it didn't separate good and bad answers in practice.

## 🎯 Retrieval Quality Benchmark

Precision@K, Recall@K, and MRR, measured against 150 golden-set questions using silver-standard relevance labels (embedding similarity to the golden set's own known-correct answer, a substitute for human annotation at scale).

| Metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| Precision | 0.967 | 0.913 | 0.856 | 0.818 |
| Recall | 0.065 | 0.154 | 0.221 | 0.389 |

MRR is 0.976. Precision falling and recall rising as K increases is the expected shape for these metrics. None of the 150 test questions had zero relevant candidates in the retrieved pool.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Embeddings | `bge-small-en-v1.5` (bi-encoder, local) |
| Reranking | `ms-marco-MiniLM-L6-v2` (cross-encoder, local) |
| Vector store | Qdrant (Docker) |
| Generation | Qwen3 8B via Ollama, local |
| API | FastAPI |
| Logging | structlog + SQLite |
| Dashboarding | Grafana + `frser-sqlite-datasource` |
| Testing | pytest, GitHub Actions |
| Language | Python |

## Running Everything Locally

Every model runs on this machine, covering retrieval, reranking, and generation. No external API calls, no cloud LLM costs, no data leaving the machine. The tradeoff is latency. Each request takes 15 to 30 seconds end to end, since Qwen3 8B runs on CPU only in this setup, with no GPU used. A faster, user facing deployment would need a smaller model, GPU inference, or both, either a lighter weight model that generates answers more quickly, or running the same model on graphics hardware instead of a regular processor, which is significantly faster for this kind of computation.

## 📁 Project Structure

```
ai-monitoring-framework/
├── README.md
├── requirements.txt
├── docker-compose.yml               ← one-command startup for Qdrant and Grafana
├── pyproject.toml                   ← makes monitor, rag, and api pip-installable
├── .gitignore                       
├── common.py                        ← shared config, models, paths, thresholds
├── .github/
│   └── workflows/
│       └── tests.yml                ← runs the unit tests on every push
├── data/
│   ├── prepare_data.py              ← downloads and splits the dataset
│   ├── kb_documents.jsonl           ← knowledge base, indexed into Qdrant
│   └── golden_set.jsonl             ← held-out set, never indexed
├── rag/
│   ├── ingest.py                    ← embeds and uploads kb_documents.jsonl into Qdrant
│   ├── pipeline.py                  ← retrieval and reranking
│   └── generate.py                  ← generation and five-signal scoring
├── monitor/
│   ├── rules.py                     ← escalation engine and thresholds
│   └── logging_db.py                ← structlog + SQLite persistence
├── api/
│   └── main.py                      ← FastAPI /ask and /health endpoints
├── scripts/
│   ├── calibrate_thresholds.py      ← derives thresholds from the golden set
│   ├── retrieval_benchmark.py       ← Precision@K, Recall@K, MRR
│   └── test_harness.py              ← sends test traffic to the live API
├── tests/
│   └── test_rules.py                ← unit tests for the escalation engine
└── outputs/                         ← generated at runtime; not tracked
    ├── traces.db                    ← SQLite execution log, every /ask call
    ├── golden_set_calibration.json  ← raw results from calibrate_thresholds.py
    └── retrieval_benchmark.json     ← raw results from retrieval_benchmark.py
```

An empty `__init__.py` also exists inside `rag/`, `monitor/`, `scripts/`, and `tests/`, marking each as an importable Python package. Left out of the tree above to keep it focused on files with actual content.

## 🚀 Run Locally

Prerequisites are Python 3.10 or higher, Docker Desktop, Ollama with `qwen3:8b` pulled, and about 8GB free RAM.

```bash
pip install -r requirements.txt
```

Start Qdrant.

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Build the knowledge base index.

```bash
python data/prepare_data.py
python rag/ingest.py
```

Start the API.

```bash
python -m uvicorn api.main:app --reload
```

Test it through the interactive docs at `http://127.0.0.1:8000/docs`.

Set up Grafana.

```bash
docker run -d -p 3000:3000 --name grafana -v $(pwd)/outputs:/data grafana/grafana
```

Open `http://localhost:3000` (default login `admin` / `admin`, then set a new password), install the `frser-sqlite-datasource` plugin, add a SQLite data source pointed at `/data/traces.db`, then build three panels: Decisions Over Time, Flag Rate Percent, and Pipeline Latency by Stage.

Run the test harness.

```bash
python scripts/test_harness.py
```

Run the unit tests.

```bash
python -m pytest tests/test_rules.py -v
```

Run the calibration and benchmarking scripts.

```bash
python scripts/calibrate_thresholds.py
python scripts/retrieval_benchmark.py
```

## 💡 What Makes This Different

Most of the engineering here happened after the pipeline already worked. Retrieval and generation were the easy part. Scoring the output honestly was not. Three single score groundedness approaches were tried and discarded before landing on the current five signal design, and the first set of escalation thresholds, calibrated cleanly on a golden set, turned out to flag 100 percent of real, naturally phrased traffic. Fixing that with live evidence, rather than trusting the offline number, is the part of this project worth calling out directly.

## Limitations

- Automated test coverage is limited to the escalation engine. `tests/test_rules.py` covers the decision logic in isolation (9 tests). Retrieval, generation, and the API endpoint are validated manually, via the test harness script and direct interactive testing, not by automated tests.
- Retrieval benchmark relevance labels are silver-standard, not human-verified.
- No authentication, rate limiting, or handling for upstream service downtime, for example Ollama or Qdrant being unreachable. The API assumes a trusted, single-user, local environment.
- Thresholds were calibrated on one dataset and one local model, and are not guaranteed to hold for a different knowledge base, embedding model, or LLM without re-running calibration.

## 🔮 Future Work

- Alerting rules in Grafana, for example a notification when flag rate exceeds a threshold for several consecutive minutes
- A dedicated toxicity or PII-leak signal in the analysis layer
- Automated tests for retrieval and the API layer
- A/B testing threshold configurations against held-out traffic before promoting new defaults

## 👩‍💻 Author

Nirusha Mantralaya Ramesh

🔗 GitHub: Nirusha-19

## 📄 License

MIT. Free to use, fork, and build upon.
