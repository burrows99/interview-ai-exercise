"""Ollama-backed LLM provider."""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI

from ai_exercise.llm.providers.base import LLMProvider

import logging

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance."""

    def __init__(self, base_url: str, model: str, embeddings_model: str) -> None:
        """Initialise the Ollama client and embedding function."""
        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model
        self._embeddings = OllamaEmbeddings(
            base_url=base_url, model=embeddings_model
        )

    def get_completion(self, prompt: str) -> str:
        """Return a text completion for the given prompt."""
        logger.debug("Requesting completion from model '%s'.", self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    @property
    def embeddings(self) -> Embeddings:
        """Return the LangChain-compatible Embeddings object."""
        return self._embeddings
