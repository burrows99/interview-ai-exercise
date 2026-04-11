"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Return a text completion for the given prompt."""
        ...

    @property
    @abstractmethod
    def embeddings(self) -> Embeddings:
        """Return a LangChain-compatible Embeddings object."""
        ...
