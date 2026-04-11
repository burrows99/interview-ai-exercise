# RAG System Improvements

## Current Limitations

| Area | Problem |
|---|---|
| Chunking | Fixed-size `RecursiveCharacterTextSplitter` splits mid-schema, breaking semantic units |
| Retrieval | Pure vector/cosine search misses exact keyword matches (e.g. `GET /hris/employees`) |
| Context | No query rewriting — poorly phrased questions return irrelevant chunks |
| Evaluation | LLM-as-judge scores are reference-free; no regression tracking over time |
| Observability | No visibility into retrieval quality per query; scores not tracked over time |

---

## Suggested Improvements

### 1. Smarter Chunking — Schema-Aware Splitting
**Current drawback:** `RecursiveCharacterTextSplitter` cuts at character boundaries, so a single endpoint's parameters, request body, and response schemas are split across multiple chunks. The LLM sees incomplete context and cannot reason about the full contract.

**How it helps:** Splitting per OpenAPI operation keeps all the information the LLM needs in a single unit — one complete, self-contained operation is retrieved instead of several partial fragments, eliminating the need to infer missing parts.

Instead, split per OpenAPI operation (one chunk = one endpoint + its parameters + response schema). This preserves semantic units and prevents context bleed between unrelated endpoints.

> Source: [LangChain — Deconstructing RAG / Chunk Size](https://blog.langchain.com/deconstructing-rag/)

### 2. Hybrid Search (Dense + Sparse)
**Current drawback:** Pure cosine similarity over embeddings blurs exact tokens. A query for `GET /hris/employees` might rank a semantically similar but unrelated endpoint higher than the exact match because embeddings capture meaning, not verbatim strings.

**How it helps:** BM25 scores documents by exact term frequency, so `/hris/employees` or `ConnectSessionCreate` score highest when those tokens appear literally. Merging BM25 and vector results via RRF means both strategies vote, and the true best result rises to the top regardless of whether the match is semantic or lexical.

Add BM25 keyword search alongside vector search, then merge results with **Reciprocal Rank Fusion (RRF)**. This catches exact matches like endpoint paths or HTTP method names that cosine similarity misses.

> Sources: [LangChain RAG Fusion](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb) · [Chroma hybrid search](https://docs.trychroma.com)

### 3. Query Rewriting + Expansion
**Current drawback:** User queries go directly into vector search. Vague or conversational queries (e.g. "how do I add a person?") don't match the technical language in the OpenAPI docs (`POST /hris/employees`), causing retrieval misses even when the answer exists.

**How it helps:** Rewriting translates natural language into retrieval-friendly technical terms before the search runs, closing the vocabulary gap. Expansion generates multiple phrasings of the same intent so a single vague query triggers several targeted retrievals, dramatically increasing recall.

Before retrieval, use the LLM to:
- **Rewrite** the user query into cleaner retrieval language (Rewrite-Retrieve-Read)
- **Expand** it into 2–3 sub-questions to cover multiple facets (multi-query retriever)

> Source: [LangChain — Query Transformations](https://blog.langchain.com/query-transformations/)

### 4. Re-ranking
**Current drawback:** Chroma returns the top-k chunks by embedding distance alone. Distance is a shallow proxy for relevance — a chunk can be close in embedding space but still not contain the answer. The top result is often noisy.

**How it helps:** A cross-encoder reads the query and each candidate chunk together (not as independent embeddings), producing a much more accurate relevance score. Reordering chunks before synthesis ensures the most directly useful one appears first in the prompt, reducing hallucination caused by noisy leading context.

After retrieval, apply a cross-encoder re-ranker (e.g. Cohere ReRank or a local `cross-encoder/ms-marco` model) to re-score and reorder the `k` chunks before synthesis. Significantly improves precision at low extra cost.

> Source: [LangChain — Post-Processing / Re-ranking](https://blog.langchain.com/deconstructing-rag/)

### 5. Metadata Filtering via Self-Query
**Current drawback:** Every query searches across all 7 APIs and all component types. A question about HRIS endpoints will retrieve schema chunks from CRM or ATS, adding noise and consuming context window tokens with irrelevant content.

**How it helps:** The self-query retriever parses the user's question to extract structural intent (e.g. "HRIS", "GET") and converts it into a metadata pre-filter before vector search runs. This constrains the search space to only the relevant API and method, drastically reducing noise in the retrieved context.

Tag each chunk with structured metadata (`api: hris`, `method: GET`, `path: /employees`) at index time. Use a **self-query retriever** so questions like "what HRIS endpoints return a list?" pre-filter by metadata before vector search.

> Source: [LangChain — Text-to-metadata filters](https://blog.langchain.com/deconstructing-rag/)

### 6. Vectorless RAG — Page-Level Index (BM25 / Full-Text)
**Current drawback:** Embedding models add latency and cost at both index and query time. For structured content like OpenAPI specs, where queries often contain exact tokens (`/employees`, `GET`, `ConnectSessionCreate`), embedding-based retrieval is overkill and introduces unnecessary semantic blurring.

**How it helps:** BM25 indexes raw tokens without any model inference. A query for `ConnectSessionCreate` returns the exact schema document in milliseconds at zero model cost — particularly effective since developers typically reference exact schema names, endpoint paths, or operation IDs rather than paraphrasing them.

Instead of embedding every chunk, build a **page-level inverted index** using BM25 (or Elasticsearch / Typesense). Retrieve full OpenAPI path objects by keyword, then pass them directly to the LLM. No embedding model needed.

**Pros:** no embedding cost or latency, deterministic ranking, zero vector DB required, great for structured/code-like content  
**Cons:** no semantic generalisation (synonyms, paraphrases missed), needs separate BM25 store alongside ChromaDB if used hybrid

> Sources: [BM25 overview — Elastic](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) · [LangChain BM25 retriever](https://python.langchain.com/docs/integrations/retrievers/bm25/)

### 7. LangGraph Agentic RAG
**Current drawback:** The pipeline is a fixed linear sequence: retrieve → prompt → generate. If retrieval returns nothing useful (distance threshold filters all chunks), the system immediately falls back to "I don't know" with no retry. There is no mechanism to rephrase the query or fetch from a different source.

**How it helps:** The LangGraph state machine adds a relevance-grading node after retrieval. If chunks score below threshold, the graph loops back and generates a rewritten query before retrying — mimicking how a researcher rephrases and searches again. Hard-coded fallbacks are replaced by intelligent, observable retry logic.

Replace the current linear pipeline with a **LangGraph state machine**:

```
query → rewrite → retrieve → grade relevance → [re-retrieve | generate] → answer
```

Nodes can decide whether retrieved context is sufficient and loop back to retrieval with a refined query if not. This removes the need for a static `score_threshold` on the retriever.

**Pros:** handles multi-hop questions, graceful degradation is built-in, observable per-node  
**Cons:** more complex, adds LangGraph dependency, slower on simple queries

> Source: [LangGraph Agentic RAG tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)

### 8. LangSmith Evaluation Pipeline
**Current drawback:** The `/evaluate` endpoint scores individual queries in isolation using LLM-as-judge, but scores are never tracked over time. There is no way to know if a prompt change improved or regressed quality, no production monitoring, and no human review workflow for bad outputs.

**How it helps:** LangSmith persists every score against a versioned dataset, making quality trends visible across experiments. Offline evals can gate PRs automatically (regression = block merge). Online evaluators catch production degradation in real-time at configurable sampling rates. Annotation queues close the loop by converting bad production traces into labelled examples for future evals.

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
  └── retriever.invoke          [query, score_threshold, k]
  └── RAGChatPrompts.answer     [context_length, n_chunks]
  └── llm_provider.get_completion  [model, tokens, latency]
```

Tag runs with metadata (`api_category`, `query_length`, `n_chunks_retrieved`) for filtering. This turns structured logs into a queryable audit trail — every retrieval miss or hallucination is inspectable.

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

---

## Sample Evaluation Questions

Use these with `/chat` or `/evaluate` to test retrieval quality across different query types.

### Schema / Component Lookups
| Query | What it tests |
|---|---|
| `What fields does ConnectSessionCreate have?` | Schema property retrieval |
| `What is required to create a connect session?` | Required field extraction |
| `What does ConnectSessionTokenAuthLink return?` | Response schema retrieval |
| `What are the properties of LinkedAccount?` | Cross-spec schema lookup |
| `What fields are in the Employee schema?` | HRIS schema retrieval |

### Endpoint Discovery
| Query | What it tests |
|---|---|
| `How do I list all linked accounts?` | Path + method resolution |
| `What endpoint creates a connect session?` | POST endpoint retrieval |
| `How do I delete an account?` | DELETE method retrieval |
| `What HRIS endpoints are available for employees?` | Multi-result path retrieval |
| `How do I get a single job posting in ATS?` | Cross-spec endpoint retrieval |

### Parameters & Auth
| Query | What it tests |
|---|---|
| `What query parameters does the list accounts endpoint accept?` | Parameter extraction |
| `How do I paginate results in the HRIS API?` | Pagination pattern retrieval |
| `What authentication does the StackOne API use?` | Security scheme retrieval |
| `What does the x-account-id header do?` | Header parameter retrieval |

### Error Handling
| Query | What it tests |
|---|---|
| `What does a 429 response mean?` | Error schema retrieval |
| `What fields does BadRequestResponse contain?` | Error schema property lookup |
| `How does the API handle rate limiting?` | Retry/backoff pattern retrieval |

### Out-of-Scope (should gracefully decline)
| Query | What it tests |
|---|---|
| `What is the capital of France?` | Graceful limitation response |
| `How do I configure a Stripe webhook?` | Out-of-domain detection |
| `What is the price of the StackOne enterprise plan?` | Non-documentation query |
