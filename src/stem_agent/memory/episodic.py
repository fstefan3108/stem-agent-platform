"""Episodic memory: stores and retrieves interaction episodes from SQLite."""

import json

from aiosqlite import Connection

from stem_agent.shared.schemas import Episode


class EpisodicMemory:
    def __init__(self, db: Connection):
        self._db = db

    async def initialize(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id   TEXT PRIMARY KEY,
                caller_id    TEXT NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                tools_used   TEXT NOT NULL,
                metadata     TEXT NOT NULL
            )
        """)
        await self._db.commit()

    async def save(self, episode: Episode) -> None:
        await self._db.execute(
            """
            INSERT INTO episodes
                (episode_id, caller_id, user_message, agent_response, timestamp, tools_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.caller_id,
                episode.user_message,
                episode.agent_response,
                episode.timestamp.isoformat(),
                json.dumps(episode.tools_used),
                json.dumps(episode.metadata),
            ),
        )
        await self._db.commit()

    async def get_recent(self, caller_id: str, limit: int = 10) -> list[Episode]:
        async with self._db.execute(
            "SELECT * FROM episodes WHERE caller_id = ? ORDER BY timestamp DESC LIMIT ?",
            (caller_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        result = []
        for row in rows:
            data = dict(row)
            data["tools_used"] = json.loads(data["tools_used"])
            data["metadata"] = json.loads(data["metadata"])
            result.append(Episode(**data))
        return result
