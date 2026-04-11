# AI Exercise - Retrieval

> simple RAG example

## Requirements

- [Docker Engine](https://docs.docker.com/engine/install/)

## Hardware

The default setup runs `llama3.2` (3B, ~2GB) and `nomic-embed-text` via Ollama, which fits comfortably within a standard M3 MacBook Pro with 16GB RAM.

If you'd prefer to use OpenAI instead (e.g. for a more capable model or on machines with less RAM), set the following in `.env`:

```env
PROVIDER=openai
OPENAI_API_KEY=<your-key>
```

The `ollama` container will still start but won't be used.

## Setup

Start everything (Ollama + model downloads + API):

```bash
docker-compose up -d
```

That's it. The API and Ollama container will start, models will be pulled automatically, and the API will be ready once models are loaded.

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
