"""Tests for CallerStore: profile load/save and EMA-based style learning."""

import pytest
import pytest_asyncio
import aiosqlite

from stem_agent.caller.store import CallerStore, _EMA_ALPHA
from stem_agent.shared.schemas import CallerProfile, StyleDimensions


# ---------------------------------------------------------------------------
# Fixture: minimal memory stub so CallerStore can borrow _db
# ---------------------------------------------------------------------------

class _FakeMemory:
    def __init__(self, db):
        self._db = db


@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    cs = CallerStore(_FakeMemory(conn))
    await cs.initialize()
    yield cs
    await conn.close()


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_creates_default_profile_for_new_caller(store):
    profile = await store.load("alice")
    assert profile.caller_id == "alice"
    assert profile.interaction_count == 0
    assert isinstance(profile.style, StyleDimensions)


@pytest.mark.asyncio
async def test_load_returns_same_profile_on_second_call(store):
    """Loading the same caller twice should return the persisted profile, not create a duplicate."""
    first = await store.load("alice")
    second = await store.load("alice")
    assert first.caller_id == second.caller_id
    assert first.created_at == second.created_at


@pytest.mark.asyncio
async def test_load_unknown_caller_inserts_row(store):
    await store.load("bob")
    async with store._db.execute(
        "SELECT COUNT(*) as cnt FROM caller_profiles WHERE caller_id = 'bob'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row["cnt"] == 1


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_persists_profile(store):
    profile = CallerProfile(caller_id="alice")
    profile.interaction_count = 5
    await store.save(profile)

    loaded = await store.load("alice")
    assert loaded.interaction_count == 5


@pytest.mark.asyncio
async def test_save_upserts_existing_profile(store):
    """Saving the same caller_id twice should update, not insert a second row."""
    profile = CallerProfile(caller_id="alice")
    await store.save(profile)
    profile.interaction_count = 99
    await store.save(profile)

    async with store._db.execute(
        "SELECT COUNT(*) as cnt FROM caller_profiles WHERE caller_id = 'alice'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row["cnt"] == 1

    loaded = await store.load("alice")
    assert loaded.interaction_count == 99


@pytest.mark.asyncio
async def test_save_roundtrips_style_dimensions(store):
    profile = CallerProfile(caller_id="alice")
    profile.style.verbosity = 0.9
    profile.style.formality = 0.1
    profile.style.use_emoji = True
    await store.save(profile)

    loaded = await store.load("alice")
    assert loaded.style.verbosity == 0.9
    assert loaded.style.formality == 0.1
    assert loaded.style.use_emoji is True


@pytest.mark.asyncio
async def test_save_roundtrips_preferences(store):
    profile = CallerProfile(caller_id="alice", preferences={"theme": "dark", "lang": "en"})
    await store.save(profile)

    loaded = await store.load("alice")
    assert loaded.preferences == {"theme": "dark", "lang": "en"}


# ---------------------------------------------------------------------------
# update_from_interaction (EMA logic)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_increments_interaction_count(store):
    await store.load("alice")
    await store.update_from_interaction("alice", {})
    profile = await store.load("alice")
    assert profile.interaction_count == 1


@pytest.mark.asyncio
async def test_update_applies_ema_to_verbosity(store):
    await store.load("alice")  # default verbosity = 0.5
    await store.update_from_interaction("alice", {"verbosity": 1.0})

    profile = await store.load("alice")
    expected = round((1 - _EMA_ALPHA) * 0.5 + _EMA_ALPHA * 1.0, 4)
    assert profile.style.verbosity == expected


@pytest.mark.asyncio
async def test_update_applies_ema_to_multiple_fields(store):
    await store.load("alice")
    await store.update_from_interaction("alice", {"verbosity": 1.0, "formality": 0.0})

    profile = await store.load("alice")
    assert profile.style.verbosity == round(0.9 * 0.5 + 0.1 * 1.0, 4)
    assert profile.style.formality == round(0.9 * 0.5 + 0.1 * 0.0, 4)


@pytest.mark.asyncio
async def test_update_sets_use_emoji_directly(store):
    """use_emoji is boolean — it should be set directly, not averaged."""
    await store.load("alice")
    await store.update_from_interaction("alice", {"use_emoji": True})

    profile = await store.load("alice")
    assert profile.style.use_emoji is True


@pytest.mark.asyncio
async def test_update_ignores_unknown_signal_keys(store):
    """Signals with keys that don't exist on StyleDimensions should be silently ignored."""
    await store.load("alice")
    await store.update_from_interaction("alice", {"nonexistent_field": 0.99})

    profile = await store.load("alice")
    assert profile.style.verbosity == 0.5  # unchanged


@pytest.mark.asyncio
async def test_update_accumulates_across_multiple_interactions(store):
    """Repeated high-verbosity signals should gradually push verbosity up."""
    await store.load("alice")

    for _ in range(5):
        await store.update_from_interaction("alice", {"verbosity": 1.0})

    profile = await store.load("alice")
    assert profile.style.verbosity > 0.5
    assert profile.interaction_count == 5
