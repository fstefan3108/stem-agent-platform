"""Tests for StemConfig — validation, defaults, and required fields."""

import pytest
from pydantic import ValidationError

from stem_agent.config import StemConfig


def test_valid_config_minimal():
    config = StemConfig(openai_api_key="sk-test")
    assert config.openai_model == "gpt-4o-mini"
    assert config.agent_name == "StemAgent"
    assert config.log_level == "INFO"
    assert "helpful" in config.system_context


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        StemConfig()


def test_db_path_default():
    config = StemConfig(openai_api_key="sk-test")
    assert config.db_path == ".stem_agent.db"


def test_db_path_override():
    config = StemConfig(openai_api_key="sk-test", db_path="/tmp/custom.db")
    assert config.db_path == "/tmp/custom.db"


def test_custom_model_override():
    config = StemConfig(openai_api_key="sk-test", openai_model="gpt-4o")
    assert config.openai_model == "gpt-4o"


def test_custom_system_context():
    config = StemConfig(
        openai_api_key="sk-test",
        system_context="You are an ACME Corp assistant.",
    )
    assert config.system_context == "You are an ACME Corp assistant."


def test_custom_agent_name():
    config = StemConfig(openai_api_key="sk-test", agent_name="AcmeBot")
    assert config.agent_name == "AcmeBot"
