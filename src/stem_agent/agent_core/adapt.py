"""Phase 2 — Adapt: load caller profile and derive behavior parameters."""

from stem_agent.shared.schemas import BehaviorParameters, CallerProfile, PerceptionResult

_TOOL_USE_BY_INTENT: dict[str, float] = {
    "search": 0.9,
    "data": 0.9,
    "task": 0.7,
    "config": 0.7,
    "debug": 0.6,
    "analysis": 0.5,
    "question": 0.4,
    "unknown": 0.4,
    "creative": 0.1,
    "chitchat": 0.0,
}

_TEMPERATURE_BY_INTENT: dict[str, float] = {
    "creative": 0.8,
    "chitchat": 0.7,
    "question": 0.4,
    "unknown": 0.4,
    "task": 0.3,
    "search": 0.2,
    "analysis": 0.2,
    "debug": 0.1,
    "data": 0.1,
    "config": 0.1,
}

_CONFIDENCE_BY_INTENT: dict[str, float] = {
    "config": 0.85,
    "task": 0.8,
    "data": 0.8,
    "unknown": 0.75,
    "debug": 0.7,
    "analysis": 0.65,
    "search": 0.6,
    "question": 0.55,
    "creative": 0.4,
    "chitchat": 0.3,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def adapt(perception: PerceptionResult, caller_profile: CallerProfile) -> BehaviorParameters:
    style = caller_profile.style
    params = BehaviorParameters()

    params.verbosity = style.verbosity
    params.formality = style.formality
    params.technical_depth = style.technical_depth

    params.tool_use_preference = _TOOL_USE_BY_INTENT.get(perception.intent, 0.4)
    params.creativity = _TEMPERATURE_BY_INTENT.get(perception.intent, 0.4)
    params.confidence_threshold = _CONFIDENCE_BY_INTENT.get(perception.intent, 0.7)

    if perception.intent == "chitchat":
        params.reasoning_depth = 1
    elif perception.complexity == "complex":
        params.reasoning_depth = 5
    elif perception.complexity == "simple":
        params.reasoning_depth = 1
    else:
        params.reasoning_depth = 3

    if perception.complexity == "complex":
        params.tool_use_preference += 0.1
        params.confidence_threshold -= 0.1
        params.self_reflection_frequency = 3
        params.max_plan_steps = 15
    elif perception.complexity == "simple":
        params.tool_use_preference -= 0.1
        params.confidence_threshold += 0.1
        params.self_reflection_frequency = 8
        params.max_plan_steps = 5

    if perception.urgency:
        params.verbosity -= 0.3
        params.proactive_suggestions = False

    if perception.sentiment == "negative":
        params.proactive_suggestions = False

    if style.verbosity < 0.3:
        params.proactive_suggestions = False

    params.verbosity = _clamp(params.verbosity)
    params.tool_use_preference = _clamp(params.tool_use_preference)
    params.confidence_threshold = _clamp(params.confidence_threshold)

    return params
