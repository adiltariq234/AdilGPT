"""
Centralized configuration management using Pydantic Settings.
Provides type-safe, validated configuration with environment variable support.
"""

import os
from typing import List, Optional, Set
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Allow extra env vars without errors
    )

    # ── Application ──
    app_name: str = Field(default="AdilGPT", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    env: str = Field(default="development", description="Environment: development, staging, production")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")

    @field_validator("env")
    def validate_env(cls, v):
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            raise ValueError(f"env must be one of: {allowed}")
        return v.lower()

    # ── Security ──
    secret_key: str = Field(default="change-me-in-production", min_length=16, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT token expiration in hours")
    cors_origins: str = Field(default="*", description="Comma-separated CORS origins")
    allowed_hosts: str = Field(default="*", description="Comma-separated allowed hosts")

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def allowed_hosts_list(self) -> List[str]:
        return [host.strip() for host in self.allowed_hosts.split(",")]

    # ── Database ──
    database_url: str = Field(default="sqlite:///data/chatbot_memory.db", description="Database URL")
    db_pool_size: int = Field(default=5, ge=1, le=100, description="DB connection pool size")
    db_max_overflow: int = Field(default=10, ge=0, description="DB max overflow connections")
    db_pool_recycle: int = Field(default=3600, ge=60, description="DB connection recycle time (seconds)")

    # ── LangGraph ──
    langgraph_db_path: str = Field(default="data/langgraph_checkpoints.sqlite", description="LangGraph SQLite path")

    # ── LLM API Keys ──
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")

    # ── Default Model ──
    default_model: str = Field(default="llama-3.3-70b-versatile", description="Default LLM model")

    # ── Tavily Search ──
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")

    # ── RAG Configuration ──
    rag_chunk_size: int = Field(default=1000, ge=100, le=10000, description="RAG chunk size")
    rag_chunk_overlap: int = Field(default=200, ge=0, le=1000, description="RAG chunk overlap")
    rag_similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="RAG similarity threshold")
    embedding_model: str = Field(default="gemini-embedding-001", description="Embedding model name")

    # ── Upload Limits ──
    max_upload_size_mb: int = Field(default=50, ge=1, le=500, description="Max upload size in MB")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # ── Rate Limiting ──
    rate_limit_requests_per_minute: int = Field(default=30, ge=1, description="Rate limit per minute")

    # ── Logging ──
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    def validate_log_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of: {allowed}")
        return v.upper()

    # ── Derived Properties ──
    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def show_docs(self) -> bool:
        return self.is_development

    @property
    def reload(self) -> bool:
        return self.is_development


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export for easy access
settings = get_settings()
