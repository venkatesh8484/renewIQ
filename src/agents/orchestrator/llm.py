"""
LLM backend factory — single function that returns the configured LLM.

Switch via LLM_BACKEND env var:
  - "ollama"      → local Ollama (free, for development)
  - "databricks"  → DBRX via Databricks AI Gateway (production/demo)
"""

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.api.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the configured LLM. Cached — instantiated once per process."""
    if settings.llm_backend == "databricks":
        logger.info(f"LLM: Databricks DBRX via {settings.databricks_host}")
        from databricks_langchain import ChatDatabricks
        return ChatDatabricks(
            endpoint="databricks-dbrx-instruct",
            temperature=0,
            max_tokens=4096,
        )

    # Default: local Ollama
    logger.info(f"LLM: Ollama {settings.ollama_llm_model} at {settings.ollama_base_url}")
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the configured embedding model. Always Ollama for now (Phase 3)."""
    logger.info(f"Embeddings: Ollama {settings.ollama_embed_model}")
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
