"""Tests for Phase 1 — perceive(): structured OpenAI call, no live API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stem_agent.agent_core.perceive import _Entity, _PerceptionResponse, perceive
from stem_agent.shared.schemas import AgentMessage, PerceptionResult


def _make_message(content: str = "Hello, can you help me?") -> AgentMessage:
    return AgentMessage(caller_id="test-caller", content=content, protocol="rest")


def _mock_parse_response(result: PerceptionResult):
    """Build a mock that looks like client.beta.chat.completions.parse(...)."""
    parsed_msg = MagicMock()
    parsed_msg.parsed = result

    choice = MagicMock()
    choice.message = parsed_msg

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_perceive_returns_perception_result():
    expected = PerceptionResult(intent="question", complexity="simple", sentiment="neutral")

    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_mock_parse_response(expected)
        )

        result = await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    assert isinstance(result, PerceptionResult)
    assert result.intent == "question"


@pytest.mark.asyncio
async def test_perceive_returns_values_from_parsed_not_raw_response():
    """Must return a PerceptionResult built from the parsed fields, not the raw response."""
    parsed = _PerceptionResponse(intent="task", complexity="medium", urgency=True)

    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_mock_parse_response(parsed)
        )

        result = await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    assert isinstance(result, PerceptionResult)
    assert result.intent == "task"
    assert result.complexity == "medium"
    assert result.urgency is True


@pytest.mark.asyncio
async def test_perceive_converts_entity_list_to_dict():
    """The wire model returns entities as a list of name/value pairs; perceive flattens to a dict."""
    parsed = _PerceptionResponse(
        intent="search",
        entities=[_Entity(name="location", value="Belgrade")],
    )

    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_mock_parse_response(parsed)
        )

        result = await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    assert result.entities == {"location": "Belgrade"}


# ---------------------------------------------------------------------------
# API call structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_perceive_passes_api_key_to_client():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )

        await perceive(_make_message(), api_key="sk-secret-key", model="gpt-4o")

    mock_cls.assert_called_once_with(api_key="sk-secret-key")


@pytest.mark.asyncio
async def test_perceive_passes_model_to_parse():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        await perceive(_make_message(), api_key="sk-test", model="gpt-4o")

    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_perceive_uses_temperature_zero():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["temperature"] == 0


@pytest.mark.asyncio
async def test_perceive_sends_user_message_content():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        msg = _make_message(content="Fix the login bug ASAP!")
        await perceive(msg, api_key="sk-test", model="gpt-4o-mini")

    messages = parse_mock.call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert user_msg["content"] == "Fix the login bug ASAP!"


@pytest.mark.asyncio
async def test_perceive_includes_system_message():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    messages = parse_mock.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert "system" in roles


@pytest.mark.asyncio
async def test_perceive_uses_perception_result_as_response_format():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["response_format"] is _PerceptionResponse
    assert issubclass(_PerceptionResponse, PerceptionResult)


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_perceive_system_prompt_is_not_empty():
    with patch("stem_agent.agent_core.perceive.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        parse_mock = AsyncMock(
            return_value=_mock_parse_response(PerceptionResult(intent="unknown"))
        )
        mock_client.beta.chat.completions.parse = parse_mock

        await perceive(_make_message(), api_key="sk-test", model="gpt-4o-mini")

    messages = parse_mock.call_args.kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] and len(system_msg["content"]) > 50
