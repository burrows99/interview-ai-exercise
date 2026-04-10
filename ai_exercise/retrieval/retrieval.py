"""Retrieve relevant chunks from a vector store."""

import chromadb

from ai_exercise.constants import SETTINGS


def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list[str]:
    """Retrieve k most relevant chunks for the query, filtered by distance threshold."""
    results = collection.query(query_texts=[query], n_results=k, include=["documents", "distances"])

    documents = results["documents"][0]
    distances = results["distances"][0]

    return [
        doc
        for doc, dist in zip(documents, distances)
        if dist <= SETTINGS.distance_threshold
    ]
