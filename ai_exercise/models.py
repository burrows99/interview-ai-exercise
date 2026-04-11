"""Types for the API."""

from langchain_core.documents import Document as Document
from pydantic import BaseModel


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str


class ChatOutput(BaseModel):
    """Model for the chat route output."""

    message: str


class EvaluateQuery(BaseModel):
    """Input for the evaluate route."""

    query: str


class EvaluationResult(BaseModel):
    """Scores returned by the evaluation route."""

    query: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
