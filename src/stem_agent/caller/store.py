"""CallerStore: load/save caller profiles with EMA-based learning in SQLite."""

import json
from aiosqlite import Connection
from stem_agent.shared.schemas import CallerProfile, StyleDimensions

_EMA_ALPHA = 0.1


class CallerStore:
    def __init__(self, db: Connection):
        self._db = db

    async def initialize(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS caller_profiles (
                caller_id         TEXT PRIMARY KEY,
                style             TEXT NOT NULL,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                created_at        TEXT NOT NULL,
                last_seen         TEXT NOT NULL,
                preferences       TEXT NOT NULL
            )
        """)
        await self._db.commit()

    async def load(self, caller_id: str) -> CallerProfile:
        async with self._db.execute(
            "SELECT * FROM caller_profiles WHERE caller_id = ?", (caller_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            profile = CallerProfile(caller_id=caller_id)
            await self.save(profile)
            return profile

        data = dict(row)
        data["style"] = StyleDimensions(**json.loads(data["style"]))
        data["preferences"] = json.loads(data["preferences"])
        return CallerProfile(**data)

    async def save(self, profile: CallerProfile) -> None:
        await self._db.execute(
            """
            INSERT INTO caller_profiles
                (caller_id, style, interaction_count, created_at, last_seen, preferences)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (caller_id) DO UPDATE SET
                style             = excluded.style,
                interaction_count = excluded.interaction_count,
                last_seen         = excluded.last_seen,
                preferences       = excluded.preferences
            """,
            (
                profile.caller_id,
                json.dumps(profile.style.model_dump()),
                profile.interaction_count,
                profile.created_at.isoformat(),
                profile.last_seen.isoformat(),
                json.dumps(profile.preferences),
            ),
        )
        await self._db.commit()

    async def update_from_interaction(self, caller_id: str, signals: dict) -> None:
        profile = await self.load(caller_id)

        style = profile.style
        for field, signal_value in signals.items():
            if field == "use_emoji":
                setattr(style, field, signal_value)
                continue
            if hasattr(style, field):
                old_value = getattr(style, field)
                new_value = (1 - _EMA_ALPHA) * old_value + _EMA_ALPHA * signal_value
                setattr(style, field, round(new_value, 4))

        profile.style = style
        profile.interaction_count += 1
        await self.save(profile)
