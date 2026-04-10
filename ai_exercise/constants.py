"""Set up some constants for the project."""

import chromadb
from pydantic import SecretStr
from pydantic_settings import BaseSettings

from ai_exercise.llm.provider import LLMProvider, create_provider


class Settings(BaseSettings):
    """Settings for the demo app.

    Reads from environment variables.
    You can create the .env file from the .env_example file.

    !!! SecretStr is a pydantic type that hides the value in logs.
    If you want to use the real value, you should do:
    SETTINGS.<variable>.get_secret_value()
    """

    class Config:
        """Config for the settings."""

        env_file = ".env"

    # Provider selection: "openai" or "ollama"
    provider: str = "ollama"

    # OpenAI settings
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_embeddings_model: str = "nomic-embed-text"

    collection_name: str = "documents"
    chunk_size: int = 1000
    k_neighbors: int = 5

    # All StackOne OpenAPI specs to load
    docs_urls: list[str] = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json",
    ]


SETTINGS = Settings()  # type: ignore

llm_provider: LLMProvider = create_provider(
    provider=SETTINGS.provider,
    openai_api_key=SETTINGS.openai_api_key.get_secret_value(),
    openai_model=SETTINGS.openai_model,
    embeddings_model=SETTINGS.embeddings_model,
    ollama_base_url=SETTINGS.ollama_base_url,
    ollama_model=SETTINGS.ollama_model,
    ollama_embeddings_model=SETTINGS.ollama_embeddings_model,
)

chroma_client = chromadb.PersistentClient(path="./.chroma_db")
