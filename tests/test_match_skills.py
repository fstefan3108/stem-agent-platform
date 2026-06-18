"""Tests for Phase 3 — match_skills(): look up a crystallised skill for the request."""

import pytest

from stem_agent.agent_core.match_skills import match_skills
from stem_agent.shared.schemas import PerceptionResult, SkillRecord


class _FakeMemory:
    """Stands in for MemoryManager — records the find_skill call and returns a preset result."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def find_skill(self, intent, entities):
        self.calls.append((intent, entities))
        return self._result


def _perception(intent="task", entities=None) -> PerceptionResult:
    return PerceptionResult(intent=intent, entities=entities or {})


@pytest.mark.asyncio
async def test_match_skills_returns_skill_from_memory():
    skill = SkillRecord(intent_pattern="task", tool_sequence=["delete_account"], maturity="mature")
    memory = _FakeMemory(skill)

    result = await match_skills(_perception(), memory)

    assert result is skill


@pytest.mark.asyncio
async def test_match_skills_returns_none_when_no_match():
    memory = _FakeMemory(None)

    result = await match_skills(_perception(), memory)

    assert result is None


@pytest.mark.asyncio
async def test_match_skills_forwards_intent_and_entities():
    memory = _FakeMemory(None)
    perception = _perception(intent="search", entities={"location": "Belgrade"})

    await match_skills(perception, memory)

    assert memory.calls == [("search", {"location": "Belgrade"})]
