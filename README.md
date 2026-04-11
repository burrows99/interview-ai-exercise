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

## Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for a detailed write-up of current limitations and what would be done next in a production-ready version.
