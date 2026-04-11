"""Factory function for creating LLM providers."""

from ai_exercise.llm.providers.base import LLMProvider
from ai_exercise.llm.providers.ollama import OllamaProvider
from ai_exercise.llm.providers.openai import OpenAIProvider


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
