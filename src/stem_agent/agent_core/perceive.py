"""Phase 1 — Perceive: classify intent, extract entities, estimate complexity."""

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from stem_agent.prompts.percieve import build_percieve_prompt
from stem_agent.shared.schemas import AgentMessage, PerceptionResult


class _Entity(BaseModel):
    name: str
    value: str


class _PerceptionResponse(PerceptionResult):
    entities: list[_Entity] = Field(default_factory=list)


async def perceive(message: AgentMessage, api_key: str, model: str) -> PerceptionResult:
    client = AsyncOpenAI(api_key=api_key)
    response = await client.beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": build_percieve_prompt()},
            {"role": "user", "content": message.content},
        ],
        response_format=_PerceptionResponse,
    )
    parsed = response.choices[0].message.parsed
    return PerceptionResult(
        intent=parsed.intent,
        entities={e.name: e.value for e in parsed.entities},
        complexity=parsed.complexity,
        urgency=parsed.urgency,
        sentiment=parsed.sentiment,
    )
