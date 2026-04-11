"""Document loader for OpenAPI specs, following LangChain's BaseLoader interface."""

import json
import logging
from collections.abc import Iterator

import requests
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS

logger = logging.getLogger(__name__)

_ADD_BATCH_SIZE = 50


class OpenAPISpecLoader(BaseLoader):
    """LangChain loader that fetches and reduces OpenAPI specs into Documents."""

    def __init__(self, urls: list[str]) -> None:
        """Initialise with a list of OpenAPI spec URLs."""
        self._urls = urls

    def lazy_load(self) -> Iterator[Document]:
        """Yield one Document per endpoint across all specs."""
        for url in self._urls:
            logger.info("Loading OpenAPI spec from %s...", url)
            yield from self._load_spec(url)

    def _load_spec(self, url: str) -> Iterator[Document]:
        """Fetch and reduce a single spec URL into Documents."""
        raw = requests.get(url, timeout=30).json()
        reduced = reduce_openapi_spec(raw, dereference=False)
        for name, description, endpoint_docs in reduced.endpoints:
            yield self._endpoint_to_document(name, description, endpoint_docs)

    @staticmethod
    def _endpoint_to_document(
        name: str, description: str | None, endpoint_docs: dict
    ) -> Document:
        """Convert a single reduced endpoint into a LangChain Document."""
        method, path = name.split(" ", 1)
        parts = [f"Path: {name}"]
        if description:
            parts.append(f"Description: {description}")
        parts.append(json.dumps(endpoint_docs))
        return Document(
            page_content="\n".join(parts),
            metadata={"source": "paths", "path": path, "method": method},
        )


def add_documents(vector_store: Chroma, docs: list[Document]) -> None:
    """Add documents to the vector store in batches."""
    for start in range(0, len(docs), _ADD_BATCH_SIZE):
        vector_store.add_documents(docs[start : start + _ADD_BATCH_SIZE])


def build_and_add_documents(vector_store: Chroma) -> int:
    """Load, split, and add all OpenAPI spec documents. Returns document count."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""],
        chunk_size=SETTINGS.chunk_size,
    )
    docs = OpenAPISpecLoader(SETTINGS.docs_urls).load_and_split(splitter)
    logger.info("Loaded and split into %d chunks.", len(docs))
    add_documents(vector_store, docs)
    return len(docs)
