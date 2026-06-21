from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM backend
    llm_backend: str = "ollama"  # "ollama" | "databricks"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # Databricks
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_serving_endpoint: str = "renewiq-agent-endpoint"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # API
    api_key: str = "dev-secret"
    log_level: str = "INFO"

    # Mock mode
    use_mock_endpoint: bool = False


settings = Settings()
