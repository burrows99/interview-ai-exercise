"""Create a vector store."""

from typing import Any

import chromadb

# Use cosine distance so scores are in [0, 1] and our threshold is meaningful.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def create_collection(
    client: chromadb.Client, embedding_fn: Any, name: str
) -> chromadb.Collection:
    """Create and return a Chroma collection, or get existing one if it exists"""
    return client.get_or_create_collection(
        name=name, embedding_function=embedding_fn, metadata=_COLLECTION_METADATA
    )


def reset_collection(
    client: chromadb.Client, embedding_fn: Any, name: str
) -> chromadb.Collection:
    """Delete and recreate a collection, wiping all existing documents."""
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.create_collection(
        name=name, embedding_function=embedding_fn, metadata=_COLLECTION_METADATA
    )
