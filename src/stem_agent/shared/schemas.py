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

    intent: str = Field(description="Extracted intent label, e.g. 'weather_query'")
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities, e.g. {'location': 'Belgrade'}",
    )


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