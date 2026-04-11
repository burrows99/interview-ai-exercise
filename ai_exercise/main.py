"""FastAPI app creation, main API routes."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_chroma import Chroma

from ai_exercise.constants import SETTINGS, chroma_client, llm_provider
from ai_exercise.evaluation.evaluator import RAGEvaluator
from ai_exercise.llm.rag_chat_prompts import RAGChatPrompts
from ai_exercise.loading.openapi_spec_loader import build_and_add_documents
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    EvaluateQuery,
    EvaluationResult,
    HealthRouteOutput,
    LoadDocumentsOutput,
)
from ai_exercise.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

vector_store: Chroma | None = None


def _reload_documents() -> Chroma:
    """Embed fresh OpenAPI docs into a new vector store and return it."""
    logger.info("Resetting vector store and reloading documents...")
    store = ChromaVectorStore(
        chroma_client, llm_provider.embeddings, SETTINGS.collection_name
    ).reset()
    count = build_and_add_documents(store)
    logger.info("Done. %d documents in vector store.", count)
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all documents into the vector store on startup."""
    logger.info("Starting up...")
    global vector_store
    vector_store = _reload_documents()
    logger.info("Startup complete.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Reload documents into the vector store."""
    logger.info("Manual document reload triggered.")
    global vector_store
    vector_store = _reload_documents()
    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    if vector_store is None:
        raise RuntimeError("Vector store not initialised")
    logger.info("Chat query: %s", chat_query.query)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": SETTINGS.k_neighbors, "score_threshold": SETTINGS.score_threshold},
    )
    relevant_chunks = [doc.page_content for doc in retriever.invoke(chat_query.query)]
    logger.debug("Retrieved %d chunks.", len(relevant_chunks))
    prompt = RAGChatPrompts.answer(query=chat_query.query, context=relevant_chunks)
    logger.debug("Prompt: %s", prompt)
    result = llm_provider.get_completion(prompt)
    return ChatOutput(message=result)


@app.post("/evaluate")
def evaluate_route(eval_query: EvaluateQuery) -> EvaluationResult:
    """Run the RAG pipeline and evaluate the result with LLM-as-judge metrics."""
    if vector_store is None:
        raise RuntimeError("Vector store not initialised")
    logger.info("Evaluate query: %s", eval_query.query)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": SETTINGS.k_neighbors, "score_threshold": SETTINGS.score_threshold},
    )
    relevant_chunks = [doc.page_content for doc in retriever.invoke(eval_query.query)]
    logger.debug("Retrieved %d chunks.", len(relevant_chunks))
    prompt = RAGChatPrompts.answer(query=eval_query.query, context=relevant_chunks)
    answer = llm_provider.get_completion(prompt)

    scores = RAGEvaluator(llm_provider).evaluate(
        query=eval_query.query,
        context=relevant_chunks,
        answer=answer,
    )

    logger.info("Evaluation scores for '%s': %s", eval_query.query, scores)

    return EvaluationResult(
        query=eval_query.query,
        answer=answer,
        **scores,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_exercise.main:app", host="0.0.0.0", port=80, reload=True)
