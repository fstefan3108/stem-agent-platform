"""MemoryManager: facade over episodic, semantic, and procedural memory stores."""

import aiosqlite

from stem_agent.memory.episodic import EpisodicMemory
from stem_agent.memory.procedural import ProceduralMemory
from stem_agent.memory.semantic import SemanticMemory
from stem_agent.shared.schemas import Episode, MemoryContext, SemanticFact


class MemoryManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db = None             

        self.episodic: EpisodicMemory | None = None
        self.semantic: SemanticMemory | None = None
        self.procedural: ProceduralMemory | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        self.episodic = EpisodicMemory(self._db)
        self.semantic = SemanticMemory(self._db)
        self.procedural = ProceduralMemory(self._db)

        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()

    async def get_context(self, caller_id: str, subject: str | None = None) -> MemoryContext:
        recent_episodes = await self.episodic.get_recent(caller_id)
        semantic_facts = await self.semantic.retrieve(subject) if subject else []

        return MemoryContext(
            recent_episodes=recent_episodes,
            semantic_facts=semantic_facts,
        )

    async def store_episode(self, episode: Episode) -> None:
        await self.episodic.save(episode)

    async def store_semantic_fact(self, fact: SemanticFact) -> None:
        await self.semantic.save(fact)

    async def record_skill_activation(self, skill_id: str, success: bool) -> None:
        await self.procedural.record_activation(skill_id, success)

    async def promote_or_create_skill(self, intent: str, entities: dict, tool_sequence: list[str]) -> None:
        await self.procedural.promote_or_create(intent, entities, tool_sequence)
