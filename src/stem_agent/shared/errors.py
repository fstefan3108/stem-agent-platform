"""Exception hierarchy for the stem_agent platform.

All custom exceptions inherit from StemAgentError so callers can catch the
entire family with a single `except StemAgentError`. Specific subclasses
indicate which subsystem raised the error.
"""

class StemAgentError(Exception):
    """Base exception for all stem_agent errors."""


class ValidationError(StemAgentError):
    """Raised when input validation fails at a stem_agent boundary.

    Distinct from pydantic.ValidationError, which is raised by Pydantic
    model parsing. This one is for our domain-level validation rules.
    """


class PipelinePhaseError(StemAgentError):
    """Raised when one of the cognitive pipeline phases fails.

    The 8 phases: Perceive, Adapt, MatchSkills, Reason, Plan, Execute,
    Learn, Respond.
    """


class MemorySubsystemError(StemAgentError):
    """Raised when a memory operation fails (episodic, semantic, procedural,
    or user-context store).

    Named to avoid collision with Python's builtin MemoryError.
    """


class ToolError(StemAgentError):
    """Raised when a tool invocation fails (MCP call, function execution, etc.)."""


class ProtocolError(StemAgentError):
    """Raised when a protocol handler fails (REST, A2A, etc.)."""