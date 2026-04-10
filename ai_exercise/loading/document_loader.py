"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.models import Document

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def get_json_data() -> list[dict[str, Any]]:
    # Send a GET request to each URL specified in SETTINGS.docs_urls
    results = []
    for url in SETTINGS.docs_urls:
        print(f"Fetching {url}...")
        response = requests.get(url)
        response.raise_for_status()
        results.append(response.json())
    return results


def _format_path_doc(path: str, method: str, operation: dict[str, Any]) -> str:
    """Produce a human-readable + JSON document for one HTTP method on a path."""
    parts = [
        f"Path: {method.upper()} {path}",
        f"OperationId: {operation.get('operationId', '')}",
        f"Summary: {operation.get('summary', '')}",
    ]
    if tags := operation.get("tags"):
        parts.append(f"Tags: {', '.join(tags)}")
    if description := operation.get("description"):
        parts.append(f"Description: {description}")
    parts.append(json.dumps({method: operation}))
    return "\n".join(parts)


def _format_schema_doc(name: str, schema: dict[str, Any]) -> str:
    """Produce a human-readable + JSON document for a single schema."""
    parts = [f"Schema: {name}"]
    if desc := schema.get("description"):
        parts.append(f"Description: {desc}")
    if props := schema.get("properties"):
        parts.append(f"Properties: {', '.join(props.keys())}")
    if req := schema.get("required"):
        parts.append(f"Required: {', '.join(req)}")
    parts.append(json.dumps({name: schema}))
    return "\n".join(parts)


def build_docs(data: list[dict[str, Any]]) -> list[Document]:
    """Chunk and convert the JSON data into a list of Document objects."""
    docs = []
    for spec in data:
        # paths: one chunk per HTTP method per path
        for path, path_item in spec.get("paths", {}).items():
            if not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method not in _HTTP_METHODS or not isinstance(operation, dict):
                    continue
                docs.append(Document(
                    page_content=_format_path_doc(path, method, operation),
                    metadata={"source": "paths", "path": path, "method": method.upper()},
                ))

        # webhooks: one chunk per webhook
        for name, webhook_item in spec.get("webhooks", {}).items():
            docs.append(Document(
                page_content=json.dumps({name: webhook_item}),
                metadata={"source": "webhooks", "name": name},
            ))

        # components: one chunk per item, with readable header for schemas
        for component_type, component_items in spec.get("components", {}).items():
            if not isinstance(component_items, dict):
                continue
            for item_name, definition in component_items.items():
                if component_type == "schemas" and isinstance(definition, dict):
                    content = _format_schema_doc(item_name, definition)
                else:
                    content = json.dumps({item_name: definition})
                docs.append(Document(
                    page_content=content,
                    metadata={"source": f"components/{component_type}", "name": item_name},
                ))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(collection: chromadb.Collection, docs: list[Document]) -> None:
    """Add documents to the collection"""
    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata or {} for doc in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
