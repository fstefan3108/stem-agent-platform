"""Tests for shared Pydantic schemas — field validation, defaults, and constraints."""

import pytest
from pydantic import ValidationError

from stem_agent.shared.schemas import (
    AgentMessage,
    AgentResponse,
    BehaviorParameters,
    CallerProfile,
    Episode,
    ExecutionPlan,
    ExecutionResult,
    PerceptionResult,
    ReasoningResult,
    SkillRecord,
    StyleDimensions,
    ToolCall,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# StyleDimensions
# ---------------------------------------------------------------------------

def test_style_dimensions_defaults():
    s = StyleDimensions()
    assert s.verbosity == 0.5
    assert s.formality == 0.5
    assert s.technical_depth == 0.5
    assert s.use_emoji is False


def test_style_dimensions_rejects_above_range():
    with pytest.raises(ValidationError):
        StyleDimensions(verbosity=1.1)


def test_style_dimensions_rejects_below_range():
    with pytest.raises(ValidationError):
        StyleDimensions(formality=-0.1)


# ---------------------------------------------------------------------------
# BehaviorParameters
# ---------------------------------------------------------------------------

def test_behavior_parameters_defaults():
    b = BehaviorParameters()
    assert b.reasoning_depth == 3
    assert b.creativity == 0.4
    assert b.max_plan_steps == 10
    assert b.proactive_suggestions is True


def test_behavior_parameters_depth_too_low():
    with pytest.raises(ValidationError):
        BehaviorParameters(reasoning_depth=0)


def test_behavior_parameters_depth_too_high():
    with pytest.raises(ValidationError):
        BehaviorParameters(reasoning_depth=6)


def test_behavior_parameters_creativity_out_of_range():
    with pytest.raises(ValidationError):
        BehaviorParameters(creativity=1.5)


# ---------------------------------------------------------------------------
# PerceptionResult
# ---------------------------------------------------------------------------

def test_perception_result_defaults():
    p = PerceptionResult(intent="question")
    assert p.complexity == "simple"
    assert p.urgency is False
    assert p.sentiment == "neutral"
    assert p.entities == {}


def test_perception_result_rejects_invalid_complexity():
    with pytest.raises(ValidationError):
        PerceptionResult(intent="question", complexity="impossible")


def test_perception_result_rejects_invalid_sentiment():
    with pytest.raises(ValidationError):
        PerceptionResult(intent="question", sentiment="happy")


def test_perception_result_valid_values():
    p = PerceptionResult(
        intent="task",
        complexity="complex",
        urgency=True,
        sentiment="negative",
        entities={"customer_id": "ACME-123"},
    )
    assert p.urgency is True
    assert p.entities["customer_id"] == "ACME-123"


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------

def test_agent_message_auto_generates_unique_ids():
    m1 = AgentMessage(caller_id="alice", content="hello", protocol="rest")
    m2 = AgentMessage(caller_id="alice", content="hello", protocol="rest")
    assert m1.message_id != m2.message_id


def test_agent_message_rejects_invalid_protocol():
    with pytest.raises(ValidationError):
        AgentMessage(caller_id="alice", content="hello", protocol="grpc")


def test_agent_message_accepts_a2a_protocol():
    m = AgentMessage(caller_id="agent-b", content="do X", protocol="a2a")
    assert m.protocol == "a2a"


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

def test_agent_response_links_to_message():
    msg = AgentMessage(caller_id="alice", content="hello", protocol="rest")
    resp = AgentResponse(
        in_response_to=msg.message_id,
        caller_id="alice",
        content="Hello back.",
        protocol="rest",
    )
    assert resp.in_response_to == msg.message_id
    assert resp.response_id != msg.message_id


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------

def test_tool_definition_empty_parameters_default():
    t = ToolDefinition(name="my_tool", description="Does something.")
    assert t.parameters == {}


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

def test_tool_call_defaults():
    tc = ToolCall(tool_name="get_tickets")
    assert tc.call_id is None
    assert tc.arguments == {}


def test_tool_call_with_arguments():
    tc = ToolCall(tool_name="get_tickets", arguments={"customer_id": "ACME-123"}, call_id="call_abc")
    assert tc.arguments["customer_id"] == "ACME-123"
    assert tc.call_id == "call_abc"


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------

def test_execution_plan_default_empty():
    plan = ExecutionPlan()
    assert plan.steps == []


def test_execution_plan_with_steps():
    plan = ExecutionPlan(steps=[ToolCall(tool_name="get_tickets"), ToolCall(tool_name="get_user")])
    assert len(plan.steps) == 2


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

def test_execution_result_defaults():
    result = ExecutionResult()
    assert result.outputs == {}
    assert result.errors == []


def test_execution_result_with_data():
    result = ExecutionResult(outputs={"get_tickets": [1, 2, 3]}, errors=["other_tool: timeout"])
    assert result.outputs["get_tickets"] == [1, 2, 3]
    assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# SkillRecord
# ---------------------------------------------------------------------------

def test_skill_record_defaults():
    skill = SkillRecord(intent_pattern="data")
    assert skill.maturity == "progenitor"
    assert skill.activation_count == 1
    assert skill.success_count == 0
    assert skill.tool_sequence == []


def test_skill_record_rejects_invalid_maturity():
    with pytest.raises(ValidationError):
        SkillRecord(intent_pattern="data", maturity="legendary")


def test_skill_record_auto_generates_id():
    s1 = SkillRecord(intent_pattern="data")
    s2 = SkillRecord(intent_pattern="data")
    assert s1.skill_id != s2.skill_id


# ---------------------------------------------------------------------------
# ReasoningResult
# ---------------------------------------------------------------------------

def test_reasoning_result_valid_strategies():
    for strategy in ("chain_of_thought", "react", "reflexion"):
        r = ReasoningResult(strategy=strategy)
        assert r.strategy == strategy


def test_reasoning_result_rejects_invalid_strategy():
    with pytest.raises(ValidationError):
        ReasoningResult(strategy="magic")


def test_reasoning_result_defaults():
    r = ReasoningResult(strategy="chain_of_thought")
    assert r.thoughts == ""
    assert r.initial_tool_calls == []


# ---------------------------------------------------------------------------
# CallerProfile
# ---------------------------------------------------------------------------

def test_caller_profile_defaults():
    p = CallerProfile(caller_id="bob")
    assert p.interaction_count == 0
    assert p.preferences == {}
    assert isinstance(p.style, StyleDimensions)


def test_caller_profile_rejects_negative_interaction_count():
    with pytest.raises(ValidationError):
        CallerProfile(caller_id="bob", interaction_count=-1)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def test_episode_defaults():
    e = Episode(caller_id="alice", user_message="hi", agent_response="hello")
    assert e.tools_used == []
    assert e.metadata == {}


def test_episode_auto_generates_unique_ids():
    e1 = Episode(caller_id="alice", user_message="hi", agent_response="hello")
    e2 = Episode(caller_id="alice", user_message="hi", agent_response="hello")
    assert e1.episode_id != e2.episode_id
