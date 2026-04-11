# RAG System Improvements

## Current Limitations

| Area | Problem |
|---|---|
| Retrieval | Pure vector/cosine search misses exact keyword matches (e.g. `GET /hris/employees`) |
| Evaluation | LLM-as-judge scores are reference-free; no regression tracking over time |
| Observability | No visibility into retrieval quality per query; scores not tracked over time |

---

## Suggested Improvements

### 1. ⭐ Vectorless RAG — Page-Level Index (BM25 / Full-Text) *(recommended first step)*

**Current drawback:** The system splits OpenAPI operations into fixed-size text chunks, embeds each chunk as a dense vector, and retrieves by cosine similarity. This creates three compounding problems:

1. **Chunking artifacts degrade response accuracy.** `RecursiveCharacterTextSplitter` cuts at character boundaries, not semantic ones. A single endpoint's summary, parameters, request body, and response schemas are split across multiple chunks. The LLM only ever sees a fragment of the contract — it cannot reason about required fields, response shapes, or error codes from a truncated chunk. LangChain's own RAG analysis found chunk size to be one of the highest-leverage tuning parameters precisely because incomplete chunks directly cause incomplete or incorrect answers. ([LangChain — Deconstructing RAG, Indexing](https://blog.langchain.com/deconstructing-rag/))

2. **Cosine similarity blurs exact tokens.** Embedding models encode *meaning*, not verbatim strings — a useful property for natural language that actively hurts retrieval of structured technical content. A query for `ConnectSessionCreate` or `GET /hris/employees` may rank a semantically *nearby* but wrong endpoint above the literal match, because the embedding space merges similar concepts. The LLM is then synthesising an answer from the wrong document.

3. **Chunking forces over-retrieval to compensate.** Because each chunk is partial, the system must retrieve multiple chunks (controlled by `k_neighbors`) and hope they collectively form a complete picture. This fills the context window with redundant or tangentially-related fragments, increasing the chance of hallucination and diluting the signal that was actually relevant.

**How BM25 fixes all three:**

BM25 (Okapi BM25) is a probabilistic ranking function that scores documents by term frequency (TF) and inverse document frequency (IDF) — no embedding model, no vector DB required. ([Robertson & Zaragoza, 2009](https://dl.acm.org/doi/10.1561/1500000019); [Elasticsearch BM25 reference](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables))

- **Response accuracy improves because documents are indexed whole, not chunked.** Each OpenAPI operation (path + method + parameters + request body + response schemas) is stored as one document. When retrieved, the LLM receives the *complete contract* for an endpoint, not a fragment — so answers about required fields, response shapes, and error codes are grounded in full context.

- **Exact token recall is deterministic.** BM25 ranks by how often query terms literally appear in a document, weighted by how rare those terms are across the corpus (IDF). A query for `ConnectSessionCreate` scores the exact schema document at the top — not a semantically adjacent one. This is particularly effective for OpenAPI content, where users naturally query with exact endpoint paths, HTTP methods, and schema names. ([Wikipedia — Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25))

- **Context quality increases because retrieval precision is higher.** With whole-document retrieval and exact matching, a single returned document typically contains the full answer. The LLM is not reasoning across several partial fragments — reducing hallucination caused by incomplete or conflicting context in the prompt.

- **No embedding latency or cost.** BM25 is a statistical function over an inverted index. There is zero inference cost at index time and zero model call at query time — making it faster and cheaper than the current embed-then-query pipeline.

Instead of embedding every chunk, build a **page-level inverted index** using BM25 (via `rank_bm25` or Elasticsearch / Typesense), where each document is one full OpenAPI operation object. Retrieve by keyword, then pass the complete document directly to the LLM.

**Pros:** complete context per retrieved document, deterministic ranking, exact token matching, no embedding cost or latency, no chunking step, zero vector DB required  
**Cons:** no semantic generalisation — synonyms and paraphrases that share no tokens with the document will miss (e.g. "how do I add a person?" won't match `POST /hris/employees`); can be partially mitigated by LangGraph retry with rewritten queries (Improvement #2)

> Sources:
> - Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389. <https://dl.acm.org/doi/10.1561/1500000019>
> - Connelly, S. (2018). *Practical BM25 — Part 2: The BM25 Algorithm and its Variables*. Elastic Blog. <https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables>
> - LangChain. *BM25 Retriever*. <https://python.langchain.com/docs/integrations/retrievers/bm25/>
> - LangChain. *Deconstructing RAG — Indexing / Chunk Size*. <https://blog.langchain.com/deconstructing-rag/>
> - Wikipedia. *Okapi BM25*. <https://en.wikipedia.org/wiki/Okapi_BM25>

### 2. LangGraph Agentic RAG
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

### 3. LangSmith Evaluation Pipeline
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
Phase 1  →  BM25 / full-text index
Phase 2  →  LangGraph for retry/rewrite logic on low-confidence retrievals
Phase 3  →  LangSmith eval pipeline
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
