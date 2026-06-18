"""Tests for Phase 2 — adapt(): derive BehaviorParameters from perception + profile."""

import pytest

from stem_agent.agent_core.adapt import adapt
from stem_agent.shared.schemas import (
    BehaviorParameters,
    CallerProfile,
    PerceptionResult,
    StyleDimensions,
)


def _perception(
    intent: str = "question",
    complexity: str = "medium",
    urgency: bool = False,
    sentiment: str = "neutral",
) -> PerceptionResult:
    return PerceptionResult(
        intent=intent, complexity=complexity, urgency=urgency, sentiment=sentiment
    )


def _profile(verbosity: float = 0.5, formality: float = 0.5, technical_depth: float = 0.5) -> CallerProfile:
    return CallerProfile(
        caller_id="test",
        style=StyleDimensions(
            verbosity=verbosity, formality=formality, technical_depth=technical_depth
        ),
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_adapt_returns_behavior_parameters():
    result = adapt(_perception(), _profile())
    assert isinstance(result, BehaviorParameters)


# ---------------------------------------------------------------------------
# reasoning_depth
# ---------------------------------------------------------------------------

def test_chitchat_forces_shallow_reasoning_even_when_complex():
    result = adapt(_perception(intent="chitchat", complexity="complex"), _profile())
    assert result.reasoning_depth == 1


def test_complex_task_gets_deep_reasoning():
    result = adapt(_perception(intent="task", complexity="complex"), _profile())
    assert result.reasoning_depth == 5


def test_simple_task_gets_shallow_reasoning():
    result = adapt(_perception(intent="task", complexity="simple"), _profile())
    assert result.reasoning_depth == 1


def test_medium_task_gets_default_reasoning():
    result = adapt(_perception(intent="task", complexity="medium"), _profile())
    assert result.reasoning_depth == 3


# ---------------------------------------------------------------------------
# Intent-driven dispositions (dict lookups)
# ---------------------------------------------------------------------------

def test_intent_sets_tool_use_preference():
    result = adapt(_perception(intent="search", complexity="medium"), _profile())
    assert result.tool_use_preference == pytest.approx(0.9)


def test_intent_sets_creativity():
    result = adapt(_perception(intent="creative", complexity="medium"), _profile())
    assert result.creativity == pytest.approx(0.8)


def test_intent_sets_confidence_threshold():
    result = adapt(_perception(intent="config", complexity="medium"), _profile())
    assert result.confidence_threshold == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Complexity effects (caps, reflection, nudges)
# ---------------------------------------------------------------------------

def test_complex_sets_caps_and_reflection():
    result = adapt(_perception(intent="task", complexity="complex"), _profile())
    assert result.max_plan_steps == 15
    assert result.self_reflection_frequency == 3


def test_simple_sets_caps_and_reflection():
    result = adapt(_perception(intent="task", complexity="simple"), _profile())
    assert result.max_plan_steps == 5
    assert result.self_reflection_frequency == 8


def test_medium_leaves_caps_at_defaults():
    result = adapt(_perception(intent="task", complexity="medium"), _profile())
    assert result.max_plan_steps == 10
    assert result.self_reflection_frequency == 5


def test_complex_nudges_tool_use_up_and_confidence_down():
    result = adapt(_perception(intent="task", complexity="complex"), _profile())
    assert result.tool_use_preference == pytest.approx(0.8)
    assert result.confidence_threshold == pytest.approx(0.7)


def test_simple_nudges_tool_use_down_and_confidence_up():
    result = adapt(_perception(intent="task", complexity="simple"), _profile())
    assert result.tool_use_preference == pytest.approx(0.6)
    assert result.confidence_threshold == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Style mirroring from profile
# ---------------------------------------------------------------------------

def test_style_dimensions_mirror_profile():
    profile = _profile(verbosity=0.8, formality=0.2, technical_depth=0.9)
    result = adapt(_perception(), profile)
    assert result.verbosity == pytest.approx(0.8)
    assert result.formality == pytest.approx(0.2)
    assert result.technical_depth == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Urgency
# ---------------------------------------------------------------------------

def test_urgency_reduces_verbosity_and_disables_suggestions():
    profile = _profile(verbosity=0.8)
    result = adapt(_perception(urgency=True), profile)
    assert result.verbosity == pytest.approx(0.5)
    assert result.proactive_suggestions is False


# ---------------------------------------------------------------------------
# Clamping (edge cases)
# ---------------------------------------------------------------------------

def test_verbosity_clamped_to_zero_for_terse_caller_under_urgency():
    profile = _profile(verbosity=0.1)
    result = adapt(_perception(urgency=True), profile)
    assert result.verbosity == 0.0


def test_tool_use_clamped_to_one():
    result = adapt(_perception(intent="search", complexity="complex"), _profile())
    assert result.tool_use_preference == 1.0


def test_all_clamped_fields_stay_within_bounds():
    for intent in [
        "question", "task", "analysis", "creative", "debug",
        "search", "data", "config", "chitchat", "unknown",
    ]:
        for complexity in ["simple", "medium", "complex"]:
            result = adapt(_perception(intent=intent, complexity=complexity), _profile(verbosity=0.0))
            assert 0.0 <= result.verbosity <= 1.0
            assert 0.0 <= result.tool_use_preference <= 1.0
            assert 0.0 <= result.confidence_threshold <= 1.0


# ---------------------------------------------------------------------------
# Proactive suggestions
# ---------------------------------------------------------------------------

def test_proactive_suggestions_on_by_default():
    result = adapt(_perception(sentiment="neutral"), _profile(verbosity=0.5))
    assert result.proactive_suggestions is True


def test_negative_sentiment_disables_suggestions():
    result = adapt(_perception(sentiment="negative"), _profile(verbosity=0.5))
    assert result.proactive_suggestions is False


def test_terse_caller_disables_suggestions():
    result = adapt(_perception(), _profile(verbosity=0.2))
    assert result.proactive_suggestions is False
