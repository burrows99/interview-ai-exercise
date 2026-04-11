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
docker-compose up -d
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

## Evaluation Metrics

The `/evaluate` endpoint scores each response using three **reference-free, LLM-as-judge** metrics (all scores are 0–1, higher is better). They were originally defined by the RAGAS framework and are widely used for production RAG evaluation.

| Metric | What it measures | Why it matters |
|---|---|---|
| **Faithfulness** | Every claim in the answer is grounded in the retrieved context — no hallucinations | Ensures the model isn't introducing facts outside the docs |
| **Answer Relevancy** | The answer actually addresses the question asked | Catches responses that are on-topic but evasive or incomplete |
| **Context Relevancy** | The retrieved chunks are relevant to the question | Measures retrieval quality independently of generation |

### Method

Each metric is scored by a second LLM call ("LLM-as-judge"). The model is given a strict prompt and asked to return a single float. This approach requires no labelled ground truth and scales to arbitrary question sets.

### Sources

- **RAGAS** — the framework that defined these three metrics:  
  Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217. <https://arxiv.org/abs/2309.15217>

- **LLM-as-judge** — the general paradigm of using an LLM to evaluate another LLM's output:  
  Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685. <https://arxiv.org/abs/2306.05685>

## Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for a detailed write-up of current limitations and what would be done next in a production-ready version.
