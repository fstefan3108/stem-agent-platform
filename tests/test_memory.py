"""Tests for the Phase 1 memory system: episodic, semantic, procedural, and manager."""

import pytest
import pytest_asyncio
import aiosqlite

from stem_agent.memory.episodic import EpisodicMemory
from stem_agent.memory.semantic import SemanticMemory
from stem_agent.memory.procedural import ProceduralMemory, _compute_maturity
from stem_agent.memory.manager import MemoryManager
from stem_agent.shared.schemas import Episode, SemanticFact, MemoryContext


# ---------------------------------------------------------------------------
# Shared fixture: in-memory SQLite connection (fast, isolated per test)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


# ---------------------------------------------------------------------------
# _compute_maturity (pure function — no DB needed)
# ---------------------------------------------------------------------------

def test_compute_maturity_progenitor():
    assert _compute_maturity(1) == "progenitor"
    assert _compute_maturity(3) == "progenitor"


def test_compute_maturity_committed():
    assert _compute_maturity(4) == "committed"
    assert _compute_maturity(9) == "committed"


def test_compute_maturity_mature():
    assert _compute_maturity(10) == "mature"
    assert _compute_maturity(100) == "mature"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def episodic(db):
    mem = EpisodicMemory(db)
    await mem.initialize()
    return mem


@pytest.mark.asyncio
async def test_episodic_save_and_retrieve(episodic):
    episode = Episode(
        caller_id="alice",
        user_message="Hello",
        agent_response="Hi there",
    )
    await episodic.save(episode)

    results = await episodic.get_recent("alice")
    assert len(results) == 1
    assert results[0].caller_id == "alice"
    assert results[0].user_message == "Hello"
    assert results[0].agent_response == "Hi there"


@pytest.mark.asyncio
async def test_episodic_get_recent_scoped_to_caller(episodic):
    """get_recent must only return episodes for the requested caller_id."""
    await episodic.save(Episode(caller_id="alice", user_message="A", agent_response="B"))
    await episodic.save(Episode(caller_id="bob", user_message="C", agent_response="D"))

    alice_episodes = await episodic.get_recent("alice")
    bob_episodes = await episodic.get_recent("bob")

    assert len(alice_episodes) == 1
    assert alice_episodes[0].caller_id == "alice"
    assert len(bob_episodes) == 1
    assert bob_episodes[0].caller_id == "bob"


@pytest.mark.asyncio
async def test_episodic_get_recent_limit(episodic):
    for i in range(15):
        await episodic.save(Episode(caller_id="alice", user_message=f"msg{i}", agent_response="ok"))

    results = await episodic.get_recent("alice", limit=5)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_episodic_get_recent_empty(episodic):
    results = await episodic.get_recent("nobody")
    assert results == []


@pytest.mark.asyncio
async def test_episodic_tools_used_roundtrip(episodic):
    episode = Episode(
        caller_id="alice",
        user_message="run it",
        agent_response="done",
        tools_used=["query_crm", "send_email"],
    )
    await episodic.save(episode)

    results = await episodic.get_recent("alice")
    assert results[0].tools_used == ["query_crm", "send_email"]


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def semantic(db):
    mem = SemanticMemory(db)
    await mem.initialize()
    return mem


@pytest.mark.asyncio
async def test_semantic_save_and_retrieve(semantic):
    fact = SemanticFact(subject="ACME", predicate="uses", object="Salesforce")
    await semantic.save(fact)

    results = await semantic.retrieve("ACME")
    assert len(results) == 1
    assert results[0].subject == "ACME"
    assert results[0].predicate == "uses"
    assert results[0].object == "Salesforce"


@pytest.mark.asyncio
async def test_semantic_upsert_updates_object(semantic):
    """Saving the same subject+predicate a second time should update the object, not insert a duplicate."""
    await semantic.save(SemanticFact(subject="ACME", predicate="uses", object="Salesforce"))
    await semantic.save(SemanticFact(subject="ACME", predicate="uses", object="HubSpot"))

    results = await semantic.retrieve("ACME")
    assert len(results) == 1
    assert results[0].object == "HubSpot"


@pytest.mark.asyncio
async def test_semantic_retrieve_unknown_subject(semantic):
    results = await semantic.retrieve("unknown")
    assert results == []


@pytest.mark.asyncio
async def test_semantic_multiple_predicates(semantic):
    await semantic.save(SemanticFact(subject="ACME", predicate="uses", object="Salesforce"))
    await semantic.save(SemanticFact(subject="ACME", predicate="located_in", object="Belgrade"))

    results = await semantic.retrieve("ACME")
    predicates = {r.predicate for r in results}
    assert predicates == {"uses", "located_in"}


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def procedural(db):
    mem = ProceduralMemory(db)
    await mem.initialize()
    return mem


@pytest.mark.asyncio
async def test_procedural_promote_or_create_inserts_progenitor(procedural):
    await procedural.promote_or_create("task", {}, ["query_crm"])

    # Not mature yet, so find_match should return None
    result = await procedural.find_match("task", {})
    assert result is None


@pytest.mark.asyncio
async def test_procedural_skill_reaches_committed(procedural):
    """After 4 activations the skill should be committed (not yet mature)."""
    await procedural.promote_or_create("task", {}, ["query_crm"])

    async with procedural._db.execute(
        "SELECT skill_id FROM skills WHERE intent_pattern = 'task'"
    ) as cursor:
        row = await cursor.fetchone()
    skill_id = row["skill_id"]

    # promote_or_create already recorded 1 activation (the INSERT).
    # record_activation 3 more times to reach 4 total.
    for _ in range(3):
        await procedural.record_activation(skill_id, success=True)

    async with procedural._db.execute(
        "SELECT maturity FROM skills WHERE skill_id = ?", (skill_id,)
    ) as cursor:
        row = await cursor.fetchone()

    assert row["maturity"] == "committed"


@pytest.mark.asyncio
async def test_procedural_skill_reaches_mature(procedural):
    """After 10 activations the skill should be mature and find_match should return it."""
    await procedural.promote_or_create("task", {}, ["query_crm"])

    async with procedural._db.execute(
        "SELECT skill_id FROM skills WHERE intent_pattern = 'task'"
    ) as cursor:
        row = await cursor.fetchone()
    skill_id = row["skill_id"]

    # 9 more activations → total 10
    for _ in range(9):
        await procedural.record_activation(skill_id, success=True)

    result = await procedural.find_match("task", {})
    assert result is not None
    assert result.intent_pattern == "task"
    assert result.maturity == "mature"
    assert result.tool_sequence == ["query_crm"]


@pytest.mark.asyncio
async def test_procedural_promote_or_create_increments_existing(procedural):
    """Calling promote_or_create twice for the same intent increments the existing skill."""
    await procedural.promote_or_create("task", {}, ["query_crm"])
    await procedural.promote_or_create("task", {}, ["query_crm"])

    async with procedural._db.execute(
        "SELECT activation_count FROM skills WHERE intent_pattern = 'task'"
    ) as cursor:
        row = await cursor.fetchone()

    assert row["activation_count"] == 2


@pytest.mark.asyncio
async def test_procedural_find_match_no_mature_skill(procedural):
    result = await procedural.find_match("unknown_intent", {})
    assert result is None


# ---------------------------------------------------------------------------
# MemoryManager (integration — uses a temp file DB)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def manager(tmp_path):
    db_file = str(tmp_path / "test.db")
    mgr = MemoryManager(db_file)
    await mgr.initialize()
    yield mgr
    await mgr._db.close()


@pytest.mark.asyncio
async def test_manager_get_context_empty(manager):
    ctx = await manager.get_context("alice")
    assert isinstance(ctx, MemoryContext)
    assert ctx.recent_episodes == []
    assert ctx.semantic_facts == []


@pytest.mark.asyncio
async def test_manager_store_and_retrieve_episode(manager):
    episode = Episode(caller_id="alice", user_message="Hey", agent_response="Hello")
    await manager.store_episode(episode)

    ctx = await manager.get_context("alice")
    assert len(ctx.recent_episodes) == 1
    assert ctx.recent_episodes[0].user_message == "Hey"


@pytest.mark.asyncio
async def test_manager_get_context_with_subject(manager):
    fact = SemanticFact(subject="ACME", predicate="uses", object="Salesforce")
    await manager.store_semantic_fact(fact)

    ctx = await manager.get_context("alice", subject="ACME")
    assert len(ctx.semantic_facts) == 1
    assert ctx.semantic_facts[0].object == "Salesforce"


@pytest.mark.asyncio
async def test_manager_get_context_no_subject_skips_semantic(manager):
    fact = SemanticFact(subject="ACME", predicate="uses", object="Salesforce")
    await manager.store_semantic_fact(fact)

    ctx = await manager.get_context("alice")  # no subject
    assert ctx.semantic_facts == []
