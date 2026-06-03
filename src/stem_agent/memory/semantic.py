"""Semantic memory: subject/predicate/object knowledge triples in SQLite."""

from aiosqlite import Connection
from stem_agent.shared.schemas import SemanticFact


class SemanticMemory:
    def __init__(self, db: Connection) -> None:
        self._db = db

    async def initialize(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id    TEXT PRIMARY KEY,
                subject    TEXT NOT NULL,
                predicate  TEXT NOT NULL,
                object     TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                UNIQUE (subject, predicate)
            )
        """)
        await self._db.commit()

    async def save(self, fact: SemanticFact) -> None:
        await self._db.execute(
            """
            INSERT INTO facts (fact_id, subject, predicate, object, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (subject, predicate)
            DO UPDATE SET object = excluded.object,
                          confidence = excluded.confidence
            """,
            (
                fact.fact_id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def retrieve(self, subject: str) -> list[SemanticFact]:
        async with self._db.execute(
            "SELECT * FROM facts WHERE subject = ? ORDER BY created_at DESC",
            (subject,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [SemanticFact(**dict(row)) for row in rows]
