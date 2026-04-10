# RAG System Improvements

## Current Limitations

| Area | Problem |
|---|---|
| Chunking | Fixed-size `RecursiveCharacterTextSplitter` splits mid-schema, breaking semantic units |
| Retrieval | Pure vector/cosine search misses exact keyword matches (e.g. `GET /hris/employees`) |
| Context | No query rewriting — poorly phrased questions return irrelevant chunks |
| Evaluation | LLM-as-judge scores are reference-free; no regression tracking over time |
| Observability | `print()` tracing; no visibility into retrieval quality per query |

---

## Suggested Improvements

### 1. Smarter Chunking — Schema-Aware Splitting
Instead of splitting by character count, split per OpenAPI operation (one chunk = one endpoint + its parameters + response schema). This preserves semantic units and prevents context bleed between unrelated endpoints.

> Source: [LangChain — Deconstructing RAG / Chunk Size](https://blog.langchain.com/deconstructing-rag/)

### 2. Hybrid Search (Dense + Sparse)
Add BM25 keyword search alongside vector search, then merge results with **Reciprocal Rank Fusion (RRF)**. This catches exact matches like endpoint paths or HTTP method names that cosine similarity misses.

> Sources: [LangChain RAG Fusion](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb) · [Chroma hybrid search](https://docs.trychroma.com)

### 3. Query Rewriting + Expansion
Before retrieval, use the LLM to:
- **Rewrite** the user query into cleaner retrieval language (Rewrite-Retrieve-Read)
- **Expand** it into 2–3 sub-questions to cover multiple facets (multi-query retriever)

> Source: [LangChain — Query Transformations](https://blog.langchain.com/query-transformations/)

### 4. Re-ranking
After retrieval, apply a cross-encoder re-ranker (e.g. Cohere ReRank or a local `cross-encoder/ms-marco` model) to re-score and reorder the `k` chunks before synthesis. Significantly improves precision at low extra cost.

> Source: [LangChain — Post-Processing / Re-ranking](https://blog.langchain.com/deconstructing-rag/)

### 5. Metadata Filtering via Self-Query
Tag each chunk with structured metadata (`api: hris`, `method: GET`, `path: /employees`) at index time. Use a **self-query retriever** so questions like "what HRIS endpoints return a list?" pre-filter by metadata before vector search.

> Source: [LangChain — Text-to-metadata filters](https://blog.langchain.com/deconstructing-rag/)

### 6. Vectorless RAG — Page-Level Index (BM25 / Full-Text)
Instead of embedding every chunk, build a **page-level inverted index** using BM25 (or Elasticsearch / Typesense). Retrieve full OpenAPI path objects by keyword, then pass them directly to the LLM. No embedding model needed.

This works especially well for this codebase because queries tend to contain exact tokens (`/employees`, `GET`, `HRIS`) that BM25 excels at matching, whereas cosine similarity over embeddings can blur these distinctions.

**Pros:** no embedding cost or latency, deterministic ranking, zero vector DB required, great for structured/code-like content  
**Cons:** no semantic generalisation (synonyms, paraphrases missed), needs separate BM25 store alongside ChromaDB if used hybrid

> Sources: [BM25 overview — Elastic](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) · [LangChain BM25 retriever](https://python.langchain.com/docs/integrations/retrievers/bm25/)

### 7. LangGraph Agentic RAG
Replace the current linear pipeline with a **LangGraph state machine**:

```
query → rewrite → retrieve → grade relevance → [re-retrieve | generate] → answer
```

Nodes can decide whether retrieved context is sufficient and loop back to retrieval with a refined query if not. This removes the need for a hard `distance_threshold`.

**Pros:** handles multi-hop questions, graceful degradation is built-in, observable per-node  
**Cons:** more complex, adds LangGraph dependency, slower on simple queries

> Source: [LangGraph Agentic RAG tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)

### 8. LangSmith Evaluation Pipeline

Replace the current `/evaluate` endpoint with a full LangSmith eval loop covering **three tiers**:

#### a) LLM-as-Judge (offline + online)
Define reusable workspace-level evaluators in LangSmith with structured output rubrics:

| Metric | Type | When |
|---|---|---|
| Faithfulness | Continuous 0–1 | Offline (PR gate) + Online (production sampling) |
| Answer relevancy | Continuous 0–1 | Offline + Online |
| Context precision | Continuous 0–1 | Offline |
| Conciseness / toxicity | Boolean | Online (100% of traces) |

Configure online evaluators with a **sampling rate** (e.g. 10% of prod traffic) to control cost. Attach them to the tracing project via the LangSmith UI — they fire automatically on every matching run.

> Sources: [LangSmith LLM-as-Judge](https://docs.langchain.com/langsmith/llm-as-judge) · [Online evaluators](https://docs.langchain.com/langsmith/online-evaluations-llm-as-judge)

#### b) Audit Traces + Observability
Instrument the app with LangSmith tracing so every request records a full **trace** (run tree):

```
/chat request
  └── get_relevant_chunks   [query, distances, k]
  └── create_prompt         [context_length]
  └── llm.get_completion    [model, tokens, latency]
```

Tag runs with metadata (`api_category`, `query_length`, `n_chunks_retrieved`) for filtering. This turns `print()` statements into a queryable audit log — every retrieval miss or hallucination is inspectable.

> Source: [LangSmith Observability Concepts](https://docs.langchain.com/langsmith/observability-concepts)

#### c) Human Review via Annotation Queues
Set up an automation rule: runs where online evaluator `faithfulness < 0.7` are auto-routed to a **single-run annotation queue**. Human reviewers score against a rubric, correct outputs, and export them to a dataset — closing the loop for future offline evals.

For comparing chunking/prompt strategies, use **pairwise annotation queues** to do side-by-side A/B reviews.

> Source: [LangSmith Annotation Queues](https://docs.langchain.com/langsmith/annotation-queues)

---

## Recommended Roadmap

```
Phase 1 (quick wins)   →  Schema-aware chunking + metadata tags + re-ranking
Phase 2 (quality)      →  Hybrid search + query rewriting
Phase 3 (production)   →  LangGraph orchestration + LangSmith eval pipeline
```
