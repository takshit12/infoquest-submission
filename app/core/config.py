"""Application configuration via environment variables (pydantic-settings)."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalWeights(BaseSettings):
    """Deterministic, config-controlled weights for the ranking signals.

    These are CONSTANTS, not LLM-emitted. The LLM identifies which constraints
    are hard vs. soft and which signals apply; magnitudes are tuned offline
    against the golden-query eval set.
    """

    model_config = SettingsConfigDict(env_prefix="WEIGHT_", env_file=".env", extra="ignore")

    industry: float = 0.25
    function: float = 0.20
    seniority: float = 0.20
    skill_category: float = 0.10
    recency: float = 0.10
    dense: float = 0.10
    bm25: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "industry_match": self.industry,
            "function_match": self.function,
            "seniority_match": self.seniority,
            "skill_category_match": self.skill_category,
            "recency_decay": self.recency,
            "dense_cosine": self.dense,
            "bm25_score": self.bm25,
        }


class Settings(BaseSettings):
    """Top-level app settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- source Postgres (no default: must be set via env; see .env.example) ----
    database_url: str = Field(alias="DATABASE_URL")

    # ---- OpenRouter LLM ----
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="anthropic/claude-haiku-4.5", alias="OPENROUTER_MODEL"
    )
    openrouter_timeout: float = Field(default=30.0, alias="OPENROUTER_TIMEOUT")
    openrouter_max_retries: int = Field(default=2, alias="OPENROUTER_MAX_RETRIES")

    # ---- embeddings ----
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ---- vector store ----
    chroma_dir: str = Field(default="./chroma_data", alias="CHROMA_DIR")
    chroma_collection: str = Field(default="roles_v1", alias="CHROMA_COLLECTION")

    # ---- sparse ----
    bm25_index_path: str = Field(default="./bm25_index.pkl", alias="BM25_INDEX_PATH")

    # ---- sessions ----
    sessions_db: str = Field(default="./sessions.db", alias="SESSIONS_DB")

    # ---- retrieval / ranking ----
    retrieval_top_k_dense: int = Field(default=100, alias="RETRIEVAL_TOP_K_DENSE")
    retrieval_top_k_sparse: int = Field(default=100, alias="RETRIEVAL_TOP_K_SPARSE")
    rrf_k: int = Field(default=60, alias="RRF_K")
    rerank_top_k: int = Field(default=50, alias="RERANK_TOP_K")
    final_top_k: int = Field(default=5, alias="FINAL_TOP_K")
    why_not_k: int = Field(default=5, alias="WHY_NOT_K")
    mmr_lambda: float = Field(default=0.7, alias="MMR_LAMBDA")
    prior_shortlist_boost: float = Field(default=0.05, alias="PRIOR_SHORTLIST_BOOST")

    # ---- MaxP ----
    maxp_multi_role_bonus: float = Field(default=0.05, alias="MAXP_MULTI_ROLE_BONUS")
    maxp_multi_role_cap: float = Field(default=0.15, alias="MAXP_MULTI_ROLE_CAP")

    # ---- logging ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # ---- server ----
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    @property
    def weights(self) -> SignalWeights:
        return SignalWeights()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
