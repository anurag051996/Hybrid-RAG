# Vector Databases: A Practical Overview

## What is a Vector Database?

A vector database is a specialised data store designed to index and query high-dimensional numerical vectors efficiently. Unlike traditional relational databases that query structured rows and columns, vector databases retrieve data by **semantic similarity** — finding vectors that are geometrically closest to a query vector in high-dimensional space.

## Why Vector Databases for AI?

Modern AI applications — especially those using Large Language Models — generate and consume vector embeddings constantly. A vector database enables:

- **Semantic search**: Find documents by meaning, not just keywords.
- **Recommendation systems**: Find items similar to what a user has liked.
- **Image/audio search**: Find similar media by embedding-level proximity.
- **RAG pipelines**: Retrieve relevant context chunks for LLM grounding.

---

## Popular Vector Databases

### ChromaDB
ChromaDB is an open-source, embeddable vector database designed for simplicity. It runs in-memory or with local persistence, making it ideal for development and small-to-medium deployments. ChromaDB supports cosine, L2, and IP distance metrics and integrates natively with LangChain and LlamaIndex.

### FAISS (Facebook AI Similarity Search)
FAISS is a C++ library (with Python bindings) developed by Meta Research. It is extremely fast for large-scale similarity search and supports approximate nearest-neighbour (ANN) algorithms like IVF and HNSW. FAISS is typically used as a backend rather than a standalone database.

### Pinecone
Pinecone is a fully managed, cloud-native vector database. It handles scaling, indexing, and infrastructure automatically. Pinecone supports hybrid search (dense + sparse) natively and is widely used in enterprise RAG deployments.

### Weaviate
Weaviate is an open-source vector database with a GraphQL API. It supports multi-modal data (text, images) and has built-in hybrid search combining BM25 with dense retrieval.

### Qdrant
Qdrant is a Rust-based vector database known for high performance and rich filtering capabilities. It supports payload filtering — allowing you to combine vector similarity with structured metadata filters efficiently.

---

## Indexing Algorithms

### HNSW (Hierarchical Navigable Small World)
HNSW is the most commonly used ANN algorithm. It builds a multi-layer graph where each layer is a subset of the previous, allowing logarithmic search complexity. ChromaDB uses HNSW by default.

**Pros**: Fast search, high recall, incremental updates.
**Cons**: High memory usage, slower build time.

### IVF (Inverted File Index)
IVF clusters vectors using k-means and searches only within the closest clusters. It is more memory-efficient than HNSW but requires training on a representative dataset.

### Flat (Brute-Force)
Compares the query vector against every indexed vector. Guarantees exact results. Suitable only for small corpora (< 100K vectors).

---

## Distance Metrics

The choice of distance metric affects retrieval quality:

| Metric | Formula | Use Case |
|---|---|---|
| Cosine Similarity | `A·B / (|A| |B|)` | Normalized embeddings (most common) |
| L2 (Euclidean) | `sqrt(Σ(ai-bi)²)` | When magnitude matters |
| Dot Product | `Σ(ai * bi)` | Equivalent to cosine on unit vectors |

Most modern embedding models produce **normalised vectors**, making cosine similarity and dot product equivalent. ChromaDB's cosine mode is the standard choice for RAG.

---

## Metadata Filtering

Production RAG systems frequently need to filter by metadata alongside vector similarity. For example:

- "Find passages from documents uploaded after 2024-01-01"
- "Search only within the legal documents collection"
- "Restrict results to PDF files"

ChromaDB supports `where` clauses:
```python
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    where={"file_type": {"$eq": ".pdf"}}
)
```

Qdrant and Pinecone support more advanced payload filtering with boolean combinations.

---

## Scaling Considerations

For production deployments beyond a few million vectors:

1. **Use approximate ANN** (HNSW or IVF) instead of brute-force.
2. **Shard the index** across multiple machines.
3. **Cache frequent queries** — repeated semantic searches can be expensive.
4. **Quantise embeddings** — int8 quantisation reduces memory by 4× with minimal quality loss.
5. **Monitor query latency** — p99 latency matters more than average for user-facing applications.
