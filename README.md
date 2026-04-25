# 🔍 Hybrid RAG — Vector Search + BM25

> **Day XX of [60-days-genai](https://github.com/your-username/60-days-genai)**

A production-ready Retrieval-Augmented Generation (RAG) pipeline that combines **dense vector search** and **sparse BM25 retrieval** via **Reciprocal Rank Fusion (RRF)** for significantly better retrieval quality than either method alone.

---

## 🧠 Why Hybrid Search?

| Method | Strength | Weakness |
|---|---|---|
| **Vector Search** | Semantic similarity, handles paraphrase | Misses exact keyword matches |
| **BM25** | Exact keyword match, fast | No semantic understanding |
| **Hybrid (RRF)** | ✅ Best of both worlds | Slightly more compute |

---

## 🏗️ Architecture

```
Documents
    │
    ▼
┌──────────────────────────────┐
│        Ingestion Layer        │
│  Loader → Chunker → Embedder │
└────────────┬─────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐     ┌──────────┐
│ ChromaDB│     │  BM25    │
│ (Dense) │     │ (Sparse) │
└────┬────┘     └────┬─────┘
     │               │
     └──────┬─────────┘
            ▼
   ┌─────────────────┐
   │  RRF Re-ranker  │
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  LLM Generator  │  ← OpenAI / Anthropic
   └────────┬────────┘
            ▼
        Answer
```

---

## 📁 Project Structure

```
day-XX-hybrid-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                        # CLI entrypoint
├── config/
│   └── settings.py                # Centralised config
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py     # Load .txt, .pdf, .md files
│   │   ├── chunker.py             # Recursive text splitter
│   │   └── embedder.py            # HuggingFace sentence-transformers
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── bm25_retriever.py      # rank_bm25 wrapper
│   │   └── hybrid_retriever.py    # Orchestrates both retrievers
│   ├── reranking/
│   │   └── rrf_reranker.py        # Reciprocal Rank Fusion
│   └── generation/
│       └── rag_pipeline.py        # End-to-end RAG chain
├── data/
│   └── sample_docs/               # Drop your docs here
├── tests/
│   ├── test_retrieval.py
│   └── test_pipeline.py
└── notebooks/
    └── demo.ipynb
```

---

## ⚙️ Setup

```bash
# 1. Clone and enter
git clone https://github.com/your-username/60-days-genai.git
cd 60-days-genai/day-XX-hybrid-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

---

## 🚀 Usage

### Ingest documents
```bash
python main.py ingest --docs-dir data/sample_docs
```

### Query
```bash
python main.py query --question "What is retrieval augmented generation?"
```

### Interactive mode
```bash
python main.py chat
```

---

## 🔧 Configuration

All settings live in `config/settings.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_VECTOR` | `10` | Candidates from vector search |
| `TOP_K_BM25` | `10` | Candidates from BM25 |
| `TOP_K_FINAL` | `5` | Final docs after RRF |
| `RRF_K` | `60` | RRF ranking constant |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |

---

## 📊 How RRF Works

Each retriever ranks documents 1…N. RRF combines them:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
               retrievers
```

Higher score = better combined rank. `k=60` is a standard constant that dampens the influence of very high ranks.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Tech Stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Vector DB**: `chromadb`
- **Sparse Retrieval**: `rank-bm25`
- **LLM**: OpenAI `gpt-4o-mini` or Anthropic `claude-sonnet-4-20250514`
- **PDF Parsing**: `pypdf`

---

## 🔗 Related Days

- Day XX-1: Naive RAG baseline
- Day XX+1: Adding cross-encoder reranking

---

*Part of the [60-days-genai](https://github.com/your-username/60-days-genai) challenge.*
