from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# Supported protocols. Extend this Literal as new gateways are added.
Protocol = Literal["rest", "a2a"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class StyleDimensions(BaseModel):
    """Style preferences for adapting responses to a caller.

    Each dimension is a continuous value in [0.0, 1.0]. Defaults are
    neutral mid-points; the caller profiler nudges these over time using
    EMA based on observed interactions.
    """

    verbosity: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = terse, 1 = verbose")
    formality: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = casual, 1 = formal")
    technical_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = layperson, 1 = expert")
    use_emoji: bool = Field(default=False, description="Whether responses may include emoji")


class CallerProfile(BaseModel):
    """Accumulated knowledge about a caller, per caller_id.

    Persists across sessions. Updated during the Learn phase of the pipeline.
    """

    caller_id: str
    style: StyleDimensions = Field(default_factory=StyleDimensions)
    interaction_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="Open extension point for caller-specific data",
    )


class AgentMessage(BaseModel):
    """An incoming message from a caller, normalized across protocols.

    The gateway is responsible for translating protocol-specific requests
    into this universal shape before handing them to the agent core.
    """

    message_id: str = Field(default_factory=_new_id)
    caller_id: str
    content: str
    protocol: Protocol
    timestamp: datetime = Field(default_factory=_utcnow)


class AgentResponse(BaseModel):
    """An outgoing response to a caller, normalized across protocols.

    The gateway serializes this back into a protocol-specific response.
    """

    response_id: str = Field(default_factory=_new_id)
    in_response_to: str = Field(description="The AgentMessage.message_id this response answers")
    caller_id: str
    content: str
    protocol: Protocol
    timestamp: datetime = Field(default_factory=_utcnow)


class PerceptionResult(BaseModel):
    """Structured output of the Perceive phase.

    Captures what the agent understood about the incoming message
    before reasoning begins.
    """

    intent: str = Field(description="Classified intent, one of the 10 intent categories.")
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted named entities, e.g. {'location': 'Belgrade'}",
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        default="simple",
        description="Estimated task complexity, used by Adapt to set reasoning depth.",
    )
    urgency: bool = Field(
        default=False,
        description="Whether the caller's message signals time pressure.",
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default="neutral",
        description="Detected sentiment of the incoming message.",
    )


class BehaviorParameters(BaseModel):
    """Tunable behavioral parameters computed by the Adapt phase.

    Derived from the caller's learned profile blended with signals from
    the current PerceptionResult. Passed through the pipeline to shape
    how Reason, Execute, and Format behave for this specific request.
    """

    reasoning_depth: int = Field(default=3, ge=1, le=5, description="How many reasoning steps to allow. 1 = shallow, 5 = deep.")
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0, description="Target response length. Mirrors StyleDimensions.verbosity.")
    formality: float = Field(default=0.5, ge=0.0, le=1.0, description="Response tone. Mirrors StyleDimensions.formality.")
    technical_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="How technical the response language should be.")
    tool_use_preference: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = prefer reasoning alone, 1 = prefer using tools.")
    creativity: float = Field(default=0.4, ge=0.0, le=1.0, description="Maps to LLM temperature in Reason and Format calls.")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence before the agent acts without clarifying.")
    proactive_suggestions: bool = Field(default=True, description="Whether the agent may volunteer related suggestions unprompted.")
    self_reflection_frequency: int = Field(default=5, ge=1, description="How many steps between self-critique passes in Reflexion strategy.")
    max_plan_steps: int = Field(default=10, ge=1, description="Hard cap on the number of tool calls in a single ExecutionPlan.")


class ToolDefinition(BaseModel):
    """Describes a single tool available to the agent.

    Stored in the ToolRegistry. The parameters field is a JSON Schema object
    that OpenAI's function-calling API uses to understand what arguments the
    tool accepts. It is auto-generated from the registered function's type hints.
    """

    name: str = Field(description="Unique tool identifier, e.g. 'query_crm'.")
    description: str = Field(description="Plain-English description of what the tool does. This is what the LLM reads to decide whether to use it.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object describing the tool's arguments.",
    )


class ToolCall(BaseModel):
    """A single tool invocation decided by the Reason phase.

    Produced by the Plan phase and consumed by the Execute phase.
    call_id is the identifier OpenAI assigns to each function call in its
    response — we carry it through so we can match results back correctly
    in multi-step tool loops.
    """

    tool_name: str = Field(description="Must match a ToolDefinition.name in the ToolRegistry.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Parsed arguments to pass to the tool function.")
    call_id: str | None = Field(default=None, description="OpenAI tool call ID. None when the call is constructed internally.")


class ExecutionPlan(BaseModel):
    """An ordered sequence of tool calls produced by the Plan phase.

    The Execute phase works through steps sequentially. An empty steps list
    means the request requires no tools — the Format phase will produce the
    response from reasoning alone.
    """

    steps: list[ToolCall] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """The collected outputs and errors from the Execute phase.

    outputs maps tool_name to whatever the tool function returned.
    errors carries human-readable failure descriptions for any steps
    that exhausted their retries — the Format phase handles these
    gracefully rather than crashing.
    """

    outputs: dict[str, Any] = Field(default_factory=dict, description="tool_name → return value for each successful tool call.")
    errors: list[str] = Field(default_factory=list, description="Descriptions of any tool calls that failed after all retries.")


class SkillRecord(BaseModel):
    """A crystallised interaction pattern stored in procedural memory.

    Created by the Learn phase when the same intent + tool sequence repeats.
    Progresses through a maturity lifecycle:
        progenitor (1-3 activations) → committed (4-9) → mature (10+)
    A mature skill causes the pipeline to skip Reason and Plan entirely.
    """

    skill_id: str = Field(default_factory=_new_id)
    intent_pattern: str = Field(description="The intent label this skill was crystallised from, e.g. 'task'.")
    entities_pattern: dict[str, Any] = Field(default_factory=dict, description="Entity shape that triggered crystallisation. Used for matching.")
    tool_sequence: list[str] = Field(default_factory=list, description="Ordered list of tool names this skill executes.")
    activation_count: int = Field(default=1, ge=1)
    success_count: int = Field(default=0, ge=0)
    maturity: Literal["progenitor", "committed", "mature"] = Field(default="progenitor")
    created_at: datetime = Field(default_factory=_utcnow)


class ReasoningResult(BaseModel):
    """Output of the Reason phase — the strategy chosen and the thoughts produced.

    initial_tool_calls carries any tool invocations OpenAI decided on during
    reasoning. The Plan phase converts these into a typed ExecutionPlan.
    Empty when strategy is chain_of_thought (no tools needed).
    """

    strategy: Literal["chain_of_thought", "react", "reflexion"] = Field(
        description="The reasoning strategy selected for this request."
    )
    thoughts: str = Field(
        default="",
        description="The raw reasoning text produced by the LLM.",
    )
    initial_tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw tool call dicts from the OpenAI response. Converted to ToolCall objects by the Plan phase.",
    )


class SemanticFact(BaseModel):
    """A distilled knowledge triple extracted from interactions.

    Stored in semantic memory. Unlike episodes, facts are not tied to a
    specific caller or timestamp — they represent global knowledge the
    agent has learned about the world it operates in.
    """

    fact_id: str = Field(default_factory=_new_id)
    subject: str = Field(description="The entity this fact is about, e.g. 'ACME Corp'.")
    predicate: str = Field(description="The relationship or property, e.g. 'uses', 'located_in'.")
    object: str = Field(description="The value or related entity, e.g. 'Salesforce'.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="How certain the agent is about this fact.")
    created_at: datetime = Field(default_factory=_utcnow)


class Episode(BaseModel):
    """One recorded interaction, stored in episodic memory.

    Written during the Learn phase after a response is produced. Read
    during the Adapt or Reason phases when historical context matters.
    """

    episode_id: str = Field(default_factory=_new_id)
    caller_id: str
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=_utcnow)
    tools_used: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """Assembled memory context passed to the Reason phase.

    Built by MemoryManager.get_context() and injected into the pipeline
    so the LLM has relevant history and knowledge before reasoning begins.
    """

    recent_episodes: list[Episode] = Field(default_factory=list)
    semantic_facts: list[SemanticFact] = Field(default_factory=list)