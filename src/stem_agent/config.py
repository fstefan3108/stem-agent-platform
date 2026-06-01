"""StemConfig: pydantic-settings configuration for the StemAgent platform."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StemConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key.")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model name used for all pipeline calls.")

    db_url: str = Field(..., description="PostgreSQL DSN, e.g. postgresql://user:pass@localhost/stem_agent.")

    system_context: str = Field(
        "You are a helpful AI assistant.",
        description="Base system prompt injected into every pipeline call. Consuming projects override this to give the agent its domain identity.",
    )
    agent_name: str = Field("StemAgent", description="Human-readable name returned in the A2A agent card.")
    log_level: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR.")
