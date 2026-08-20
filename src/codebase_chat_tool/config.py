from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="codebase_chat_tool", alias="QDRANT_COLLECTION")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")

    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_k: int = Field(default=30, alias="RETRIEVAL_CANDIDATE_K")

    index_dir: str = Field(default=".codebase_chat_tool", alias="INDEX_DIR")

    def index_path(self, repo_root: Path) -> Path:
        return repo_root / self.index_dir


def get_settings() -> Settings:
    return Settings()
