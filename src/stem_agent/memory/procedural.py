"""Procedural memory: skill registry with maturity lifecycle in SQLite."""

import json
from aiosqlite import Connection
from stem_agent.shared.schemas import SkillRecord

_MATURITY_THRESHOLDS = {"progenitor": 4, "committed": 10}


def _compute_maturity(activation_count: int) -> str:
    if activation_count >= _MATURITY_THRESHOLDS["committed"]:
        return "mature"
    if activation_count >= _MATURITY_THRESHOLDS["progenitor"]:
        return "committed"
    return "progenitor"


class ProceduralMemory:
    def __init__(self, db: Connection):
        self._db = db

    async def initialize(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id         TEXT PRIMARY KEY,
                intent_pattern   TEXT NOT NULL,
                entities_pattern TEXT NOT NULL,
                tool_sequence    TEXT NOT NULL,
                activation_count INTEGER NOT NULL DEFAULT 1,
                success_count    INTEGER NOT NULL DEFAULT 0,
                maturity         TEXT NOT NULL DEFAULT 'progenitor',
                created_at       TEXT NOT NULL
            )
        """)
        await self._db.commit()

    async def find_match(self, intent: str, entities: dict) -> SkillRecord | None:
        async with self._db.execute(
            """
            SELECT * FROM skills
            WHERE intent_pattern = ? AND maturity = 'mature'
            ORDER BY success_count DESC
            """,
            (intent,),
        ) as cursor:
            rows = await cursor.fetchall()

        request_keys = set(entities.keys())
        for row in rows:
            entities_pattern = json.loads(row["entities_pattern"])
            if set(entities_pattern.keys()) != request_keys:
                continue
            data = dict(row)
            data["entities_pattern"] = entities_pattern
            data["tool_sequence"] = json.loads(data["tool_sequence"])
            return SkillRecord(**data)

        return None

    async def record_activation(self, skill_id: str, success: bool) -> None:
        async with self._db.execute(
            "SELECT activation_count, success_count FROM skills WHERE skill_id = ?",
            (skill_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return

        new_activation_count = row["activation_count"] + 1
        new_success_count = row["success_count"] + (1 if success else 0)
        new_maturity = _compute_maturity(new_activation_count)

        await self._db.execute(
            """
            UPDATE skills
            SET activation_count = ?,
                success_count    = ?,
                maturity         = ?
            WHERE skill_id = ?
            """,
            (new_activation_count, new_success_count, new_maturity, skill_id),
        )
        await self._db.commit()

    async def promote_or_create(self, intent: str, entities: dict, tool_sequence: list[str]) -> None:
        async with self._db.execute(
            "SELECT skill_id FROM skills WHERE intent_pattern = ?",
            (intent,),
        ) as cursor:
            existing = await cursor.fetchone()

        if existing is not None:
            await self.record_activation(existing["skill_id"], success=True)
            return

        skill = SkillRecord(
            intent_pattern=intent,
            entities_pattern=entities,
            tool_sequence=tool_sequence,
        )
        
        await self._db.execute(
            """
            INSERT INTO skills
                (skill_id, intent_pattern, entities_pattern, tool_sequence,
                 activation_count, success_count, maturity, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill.skill_id,
                skill.intent_pattern,
                json.dumps(skill.entities_pattern),
                json.dumps(skill.tool_sequence),
                skill.activation_count,
                skill.success_count,
                skill.maturity,
                skill.created_at.isoformat(),
            ),
        )
        await self._db.commit()
