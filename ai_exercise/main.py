"""FastAPI app creation, main API routes."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, llm_provider
from ai_exercise.llm.completions import create_prompt
from ai_exercise.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data,
    split_docs,
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
)
from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.retrieval.vector_store import create_collection

collection = create_collection(
    chroma_client, llm_provider.embedding_function, SETTINGS.collection_name
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all documents into the vector store on startup."""
    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} documents. Skipping load.")
    else:
        print("Loading documents from StackOne OpenAPI specs...")
        json_data = get_json_data()
        print(f"Fetched {len(json_data)} specs. Building docs...")
        documents = build_docs(json_data)
        print(f"Built {len(documents)} documents. Splitting...")
        documents = split_docs(documents)
        print(f"Split into {len(documents)} chunks. Adding to vector store...")
        add_documents(collection, documents)
        print(f"Done. Number of documents in collection: {collection.count()}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store."""
    json_data = get_json_data()
    documents = build_docs(json_data)

    # split docs
    documents = split_docs(documents)

    # load documents into vector store
    add_documents(collection, documents)

    # check the number of documents in the collection
    print(f"Number of documents in collection: {collection.count()}")

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Get relevant chunks from the collection
    relevant_chunks = get_relevant_chunks(
        collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors
    )

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=relevant_chunks)

    print(f"Prompt: {prompt}")

    # Get completion from LLM provider
    result = llm_provider.get_completion(prompt)

    return ChatOutput(message=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
