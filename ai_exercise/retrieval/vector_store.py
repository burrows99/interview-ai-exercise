"""LangChain Chroma vector store manager."""

import contextlib
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Use cosine distance so scores are in [0, 1] and our threshold is meaningful.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class ChromaVectorStore:
    """Manages creation and reset of a LangChain Chroma vector store."""

    def __init__(
        self, client: chromadb.ClientAPI, embeddings: Embeddings, name: str
    ) -> None:
        """Initialise with a chromadb client, embedding function, and collection name."""
        self._client = client
        self._embeddings = embeddings
        self._name = name

    def _build(self) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=self._name,
            embedding_function=self._embeddings,
            collection_metadata=_COLLECTION_METADATA,
        )

    def get(self) -> Chroma:
        """Return a Chroma vector store, creating the collection if needed."""
        return self._build()

    def reset(self) -> Chroma:
        """Delete and recreate the collection, then return a fresh vector store."""
        logger.info("Resetting collection '%s'.", self._name)
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=self._name)
        return self._build()
