"""Phase 3 — Match Skills: check if a crystallised skill covers the current request."""

from stem_agent.memory.manager import MemoryManager
from stem_agent.shared.schemas import PerceptionResult, SkillRecord


async def match_skills(perception: PerceptionResult, memory: MemoryManager) -> SkillRecord | None:
    return await memory.find_skill(perception.intent, perception.entities)
