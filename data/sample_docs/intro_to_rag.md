# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge at inference time. Instead of relying solely on the knowledge encoded in model weights during training, a RAG system first retrieves relevant documents from a knowledge base and then uses those documents as context for generation.

RAG was introduced in the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. (2020) from Facebook AI Research. It has since become one of the most widely adopted techniques in production LLM applications.

## Why RAG?

LLMs have several well-known limitations:
- **Knowledge cutoff**: Their training data has a fixed cutoff date.
- **Hallucination**: They can generate plausible-sounding but incorrect facts.
- **Domain specificity**: They may not know about proprietary or niche information.

RAG directly addresses all three by grounding the model's responses in retrieved, verifiable documents.

---

## Components of a RAG Pipeline

### 1. Document Ingestion
Documents are loaded, split into chunks, and converted into vector embeddings. These embeddings are stored in a vector database like ChromaDB, FAISS, or Pinecone.

### 2. Retrieval
When a user asks a question, the query is embedded using the same model. The system searches the vector database for the most semantically similar chunks. This is called **dense retrieval**.

### 3. Generation
The retrieved chunks are passed as context to the LLM along with the original question. The model generates an answer grounded in the retrieved content.

---

## Hybrid Search

Hybrid search combines two complementary retrieval approaches:

### Dense Retrieval (Vector Search)
Dense retrieval converts text into high-dimensional vector representations using transformer-based encoder models like `all-MiniLM-L6-v2` or `text-embedding-ada-002`. Similarity is computed via cosine distance or dot product. It excels at capturing **semantic meaning** — finding documents that mean the same thing even if they use different words.

### Sparse Retrieval (BM25)
BM25 (Best Match 25) is a probabilistic ranking function based on TF-IDF principles. It scores documents by the frequency and rarity of exact query terms. BM25 excels at **exact keyword matching** — it will reliably find documents that contain the precise terms in the query.

### Reciprocal Rank Fusion (RRF)
RRF is the standard method for combining ranked lists from multiple retrievers. For each document, its RRF score is computed as:

```
RRF(d) = Σ  1 / (k + rank_i(d))
        i
```

Where `k = 60` is a smoothing constant and `rank_i(d)` is the rank of document `d` in retriever `i`. Documents that appear in multiple ranked lists receive higher scores. RRF is parameter-free (aside from k), robust, and consistently outperforms linear combination of scores.

---

## Chunking Strategies

Chunking is a critical but often underestimated step. Chunk size affects retrieval quality significantly:

- **Too small**: Chunks lose context; retrieved passages are incomplete.
- **Too large**: Chunks contain irrelevant content; noise is introduced.
- **Overlapping chunks**: Adding overlap (e.g., 64 tokens) between adjacent chunks ensures that sentences split across boundaries are still retrievable.

Common strategies:
- **Fixed-size character splitting**: Simple but ignores sentence boundaries.
- **Recursive character splitting**: Tries paragraph → sentence → word splits in order.
- **Semantic chunking**: Splits based on embedding similarity between sentences.

---

## Evaluation of RAG Systems

RAG systems should be evaluated across three axes:

1. **Retrieval quality**: Are the right chunks being retrieved? Metrics: Recall@K, MRR, NDCG.
2. **Faithfulness**: Does the answer accurately reflect the retrieved context? (No hallucinations)
3. **Answer relevance**: Does the answer address the user's question?

Frameworks like RAGAS and TruLens automate RAG evaluation using LLM-as-a-judge approaches.

---

## Best Practices

- Use a **reranker** (e.g., cross-encoder) after retrieval for additional precision.
- Always store **metadata** with chunks (source, page number) for attribution.
- Use **hybrid search** in production — pure vector search misses exact keyword matches.
- Implement **query expansion** to improve recall for ambiguous queries.
- Monitor **retrieval latency** — vector search on large corpora can be slow without approximate nearest-neighbour (ANN) indexing.
