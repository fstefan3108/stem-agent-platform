"""Tests for StemConfig — validation, defaults, and required fields."""

import pytest
from pydantic import ValidationError

from stem_agent.config import StemConfig


def test_valid_config_minimal():
    config = StemConfig(openai_api_key="sk-test", db_url="postgresql://localhost/test")
    assert config.openai_model == "gpt-4o-mini"
    assert config.agent_name == "StemAgent"
    assert config.log_level == "INFO"
    assert "helpful" in config.system_context


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        StemConfig(db_url="postgresql://localhost/test")


def test_missing_db_url_raises(monkeypatch):
    monkeypatch.delenv("DB_URL", raising=False)
    with pytest.raises(ValidationError):
        StemConfig(openai_api_key="sk-test")


def test_custom_model_override():
    config = StemConfig(openai_api_key="sk-test", db_url="postgresql://localhost/test", openai_model="gpt-4o")
    assert config.openai_model == "gpt-4o"


def test_custom_system_context():
    config = StemConfig(
        openai_api_key="sk-test",
        db_url="postgresql://localhost/test",
        system_context="You are an ACME Corp assistant.",
    )
    assert config.system_context == "You are an ACME Corp assistant."


def test_custom_agent_name():
    config = StemConfig(openai_api_key="sk-test", db_url="postgresql://localhost/test", agent_name="AcmeBot")
    assert config.agent_name == "AcmeBot"
