"""LLM Provider abstraction supporting OpenAI and Ollama backends."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Return a text completion for the given prompt."""
        ...

    @property
    @abstractmethod
    def embedding_function(self) -> Any:
        """Return a ChromaDB-compatible embedding function."""
        ...


class OpenAIProvider(LLMProvider):
    """LLM provider backed by OpenAI."""

    def __init__(self, api_key: str, model: str, embeddings_model: str) -> None:
        from openai import OpenAI
        import chromadb.utils.embedding_functions as ef

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._embedding_function = ef.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embeddings_model,
        )

    def get_completion(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content  # type: ignore[return-value]

    @property
    def embedding_function(self) -> Any:
        return self._embedding_function


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance."""

    def __init__(self, base_url: str, model: str, embeddings_model: str) -> None:
        from openai import OpenAI
        import chromadb.utils.embedding_functions as ef

        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model
        self._embedding_function = ef.OllamaEmbeddingFunction(
            url=f"{base_url}/api/embeddings",
            model_name=embeddings_model,
        )

    def get_completion(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content  # type: ignore[return-value]

    @property
    def embedding_function(self) -> Any:
        return self._embedding_function


def create_provider(
    provider: str,
    openai_api_key: str,
    openai_model: str,
    embeddings_model: str,
    ollama_base_url: str,
    ollama_model: str,
    ollama_embeddings_model: str,
) -> LLMProvider:
    """Factory: return the configured LLMProvider."""
    if provider == "ollama":
        return OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
            embeddings_model=ollama_embeddings_model,
        )
    if provider == "openai":
        return OpenAIProvider(
            api_key=openai_api_key,
            model=openai_model,
            embeddings_model=embeddings_model,
        )
    raise ValueError(f"Unknown provider '{provider}'. Supported: 'openai', 'ollama'.")
