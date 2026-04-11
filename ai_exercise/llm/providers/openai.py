"""OpenAI-backed LLM provider."""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import SecretStr

from ai_exercise.llm.providers.base import LLMProvider

import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider backed by OpenAI."""

    def __init__(self, api_key: str, model: str, embeddings_model: str) -> None:
        """Initialise the OpenAI client and embedding function."""
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=SecretStr(api_key), model=embeddings_model
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
