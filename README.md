# AI Exercise - Retrieval

> simple RAG example

## Requirements

- [Docker Engine](https://docs.docker.com/engine/install/)
- [Ollama Desktop](https://ollama.com/download)

## Hardware

The default setup uses `gpt-oss:120b-cloud` via Ollama Desktop (cloud-hosted, no local GPU required) and `nomic-embed-text` for embeddings.

If you'd prefer to use OpenAI instead, set the following in `.env`:

```env
PROVIDER=openai
OPENAI_API_KEY=<your-key>
```

## Ollama Desktop Setup

1. Download and install [Ollama Desktop](https://ollama.com/download)
2. Sign in to your Ollama account in the app
3. Pull the embeddings model:

```bash
ollama pull nomic-embed-text
```

The chat model (`gpt-oss:120b-cloud`) is cloud-hosted and does not need to be pulled separately — it will be used automatically once you're signed in.

## Setup

Start the API:

```bash
docker compose up api -d
```

The API connects to your local Ollama Desktop instance automatically.

## Usage

Open [http://localhost/docs](http://localhost/docs) in your browser to access the interactive API docs.

- Use the `/chat` endpoint to query the RAG system
- Use the `/evaluate` endpoint to evaluate retrieval quality

## Get Started

Have a look in `ai_exercise/constants.py`. Then check out the server routes in `ai_exercise/main.py`.

1. Load some documents by calling the `/load` endpoint. Does the system work as intended? Are there any issues?

2. Find some method of evaluating the quality of the retrieval system.

3. See how you can improve the retrieval system. Some ideas:
- Play with the chunking logic
- Try different embeddings models
- Other types of models which may be relevant
- How else could you store the data for better retrieval?

## Running Evaluations

The project includes a batch evaluation CLI (`evals.py`) that runs a test dataset through the full RAG pipeline and scores each response. Results are printed to the terminal and saved as a timestamped CSV.

### Dataset format

CSV files in `evals/datasets/` with two columns:

| Column | Required | Description |
|---|---|---|
| `query` | yes | The question to ask the RAG system |
| `grading_notes` | no | Reference criteria; enables a pass/fail correctness score |

### Docker

```bash
docker compose up evals
```

Results are written back to `evals/experiments/experiment_<timestamp>.csv` on the host via the bind mount.

### Output

- **Terminal** — formatted score table with per-question and aggregate metrics  
- **CSV** — `evals/experiments/experiment_<timestamp>.csv` with columns: `query`, `answer`, `faithfulness`, `answer_relevancy`, `context_relevancy`, `correctness`, `retrieved_chunks`

---

## Evaluation Metrics

This project uses a **reference-free, LLM-as-judge** evaluation system inspired by the [RAGAS framework](https://arxiv.org/abs/2309.15217) — the most widely adopted approach for evaluating RAG pipelines without requiring labelled ground truth.

All scores are 0–1, higher is better. They are computed both by the `/evaluate` API endpoint (single-query, on-demand) and by `evals.py` (batch, regression testing).

### Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Every claim in the answer is grounded in the retrieved context — no hallucinations |
| **Answer Relevancy** | The answer actually addresses the question asked |
| **Context Relevancy** | The retrieved chunks are relevant to the question |
| **Correctness** | Pass/fail: does the answer cover the grading notes? (batch eval only, requires `grading_notes` column) |

### How it works

Each metric is scored by a second LLM call ("LLM-as-judge"): the model receives a strict prompt and returns a single float. The correctness metric uses a discrete pass/fail prompt comparing the answer against reference grading notes, equivalent to RAGAS's `DiscreteMetric` pattern.

This approach:
- requires **no labelled ground truth** at index time
- scales to arbitrary question sets
- produces a **regression-trackable CSV** per run, enabling quality trends to be observed across pipeline changes

### Citations

- **RAGAS** — defined the faithfulness, answer relevancy, and context relevancy metrics used here:  
  Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217. <https://arxiv.org/abs/2309.15217>

- **LLM-as-judge** — the general paradigm of using an LLM to score another LLM's output:  
  Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685. <https://arxiv.org/abs/2306.05685>

- **RAGAS LangChain integration** — the evaluation loop pattern (`EvaluationDataset`, `DiscreteMetric`, experiment CSV export) this project follows:  
  <https://docs.ragas.io/en/stable/howtos/integrations/langchain/>

## Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for a detailed write-up of current limitations and what would be done next in a production-ready version.
