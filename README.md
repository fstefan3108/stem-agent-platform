# StemAgent Platform

Inspired by the [STEM Agent paper](https://arxiv.org/html/2603.22359v1) _(Shen et al., 2025)_.

---

## What is StemAgent?

StemAgent is a **general-purpose AI agent core** designed to be the foundation layer for building specialised agentic systems. It provides everything an agent needs to function — memory, user profiling, reasoning, tool execution, and multi-protocol communication — with no domain logic baked in.

The name comes from biology: a **stem cell** is undifferentiated but capable of specialising into any cell type. StemAgent works the same way. It has no opinion about what your agent _does_; it only defines _how_ an agent thinks, remembers, adapts, communicates and eventually learns. Your own domain logic sits on top of it.

**S.T.E.M.** — Self-adapting · Tool-enabled · Extensible · Multi-agent

### The mental model

```
User's project:
  ├── domain tools (query_crm, create_ticket, run_report …)
  ├── system context ("You are an assistant for ACME internal ops")
  └── FastAPI app (or any ASGI app)

StemAgent provides:
  ├── 8-phase cognitive pipeline (Perceive → Learn)
  ├── per-caller memory & profile learning
  ├── tool registry + OpenAI function-calling
  ├── REST endpoint (for human callers)
  └── A2A endpoint (for agent-to-agent calls)
```

---

## Pipeline Execution Flow

Every message — whether from a human via REST or from another agent via A2A — runs through the same 8-phase pipeline.

```mermaid
flowchart TD
    A([Incoming Message\nREST or A2A]) --> GW

    subgraph GW["Gateway Layer"]
        direction TB
        G1[Normalise to AgentMessage\ncaller_id · content · protocol · timestamp]
    end

    GW --> P1

    subgraph CORE["Agent Core — 8-Phase Pipeline"]
        direction TB

        P1["①  PERCEIVE\nClassify intent · extract entities\nestimate complexity · detect urgency"]
        P2["②  ADAPT\nLoad CallerProfile · compute\n10 BehaviorParameters"]
        P3{"③  MATCH SKILLS\nMature skill found?"}
        P4["④  REASON\nSelect strategy: CoT · ReAct · Reflexion\nOpenAI call → thoughts + tool plan"]
        P5["⑤  PLAN\nBuild ExecutionPlan from tool calls\n(sequential steps)"]
        P6["⑥  EXECUTE\nRun tools · retry ×2\ncircuit breaker at 3 failures"]
        P7["⑦  FORMAT\nApply caller style\n(verbosity · formality · depth)"]
        P8(["⑧  LEARN  🔄 async\nUpdate CallerProfile via EMA\nSave episode · attempt skill crystallisation"])

        P1 --> P2 --> P3
        P3 -- "Yes → skip reasoning" --> P6
        P3 -- No --> P4 --> P5 --> P6 --> P7
        P7 -.->|fire-and-forget| P8
    end

    CORE --> GW2

    subgraph GW2["Gateway Layer"]
        direction TB
        G2[Serialise AgentResponse\nback to protocol format]
    end

    GW2 --> Z([Response to Caller])

    style P8 fill:#f0f4ff,stroke:#a0a8d0,stroke-dasharray:4 4
    style P3 fill:#fff8e1,stroke:#c8a000
    style GW fill:#f5f5f5,stroke:#bbb
    style GW2 fill:#f5f5f5,stroke:#bbb
    style CORE fill:#fafafa,stroke:#ccc
```

### Phase-by-phase breakdown

| # | Phase | What it does |
|---|-------|--------------|
| 1 | **Perceive** | OpenAI call (JSON mode) — classifies intent into 10 categories, extracts entities, estimates complexity (`simple`/`medium`/`complex`), detects urgency and sentiment. |
| 2 | **Adapt** | Loads the caller's `CallerProfile` from PostgreSQL. Derives 10 `BehaviorParameters` (reasoning depth, verbosity, creativity, …) blended from the profile and the current perception. |
| 3 | **Match Skills** | Queries the `SkillRegistry` for a crystallised skill matching the current intent + entities. If a `mature` skill is found the pipeline skips Reason and Plan and jumps straight to Execute. |
| 4 | **Reason** | Picks a reasoning strategy based on complexity and available tools: `chain_of_thought` (default), `react` (tools needed), `reflexion` (complex analysis). Makes an OpenAI call to generate thoughts and an initial tool plan. |
| 5 | **Plan** | Converts the tool call decisions from Reason into a concrete `ExecutionPlan` (ordered list of `ToolCall` objects). Validates that each requested tool is registered. |
| 6 | **Execute** | Runs each step of the plan via `ToolExecutor`. Retries failed tool calls up to 2×. A circuit breaker aborts the pipeline after 3 consecutive failures. |
| 7 | **Format** | Builds the final response. Injects reasoning, tool outputs, and memory context into an OpenAI prompt styled by the caller's `StyleDimensions` (verbosity, formality, technical depth). |
| 8 | **Learn** _(async)_ | Fires as a background task — never blocks the response. Updates the `CallerProfile` via EMA (α = 0.1), saves the `Episode` to episodic memory, and checks whether the interaction pattern should be crystallised into a new skill. |

---

## Folder Structure

```
src/stem_agent/
├── __init__.py              ← public API: StemAgent, StemConfig
├── stem_agent.py            ← StemAgent — top-level class
├── config.py                ← StemConfig (pydantic-settings)
│
├── shared/
│   ├── errors.py            ← exception hierarchy
│   ├── logger.py            ← centralised logging
│   └── schemas.py           ← all Pydantic models
│
├── agent_core/
│   ├── pipeline.py          ← AgentCore: orchestrates phases 1–8
│   ├── perceive.py          ← phase 1
│   ├── adapt.py             ← phase 2
│   ├── match_skills.py      ← phase 3
│   ├── reason.py            ← phase 4
│   ├── plan.py              ← phase 5
│   ├── execute.py           ← phase 6
│   ├── format.py            ← phase 7
│   └── learn.py             ← phase 8
│
├── memory/
│   ├── episodic.py          ← interaction episodes (PostgreSQL)
│   ├── semantic.py          ← knowledge triples (PostgreSQL)
│   ├── procedural.py        ← skill registry with maturity lifecycle
│   └── manager.py           ← MemoryManager facade
│
├── caller/
│   └── store.py             ← CallerStore: profile load/save + EMA learning
│
├── tools/
│   ├── registry.py          ← ToolRegistry: register & expose tools
│   ├── executor.py          ← ToolExecutor: invoke tools by name
│   └── decorators.py        ← @agent.tool() decorator
│
└── gateway/
    ├── rest.py              ← create_rest_router(agent) → FastAPI APIRouter
    └── a2a.py               ← create_a2a_router(agent) → FastAPI APIRouter
```

---

## Protocols

StemAgent exposes two protocol adapters that a consuming project mounts on its own FastAPI app.

### REST (human → agent)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message. Body: `{"caller_id": str, "message": str}` |
| `GET` | `/callers/{caller_id}/profile` | Retrieve the caller's learned profile |
| `DELETE` | `/callers/{caller_id}` | Erase all memory for a caller (GDPR forget-me) |

### A2A — Agent-to-Agent 

| Method | JSON-RPC method | Description |
|--------|-----------------|-------------|
| `POST` | `tasks/send` | Delegate a task from another agent |
| `POST` | `tasks/get` | Poll task status |
| `POST` | `tasks/cancel` | Cancel a running task |
| `GET` | `/.well-known/agent.json` | Agent discovery card |

---

## Memory System

Four complementary stores, all backed by PostgreSQL, managed by a single `MemoryManager` facade.

| Store | Purpose | Table |
|-------|---------|-------|
| **Episodic** | Records every interaction with timestamp, intent, and tools used. Retrieved as context for future reasoning. | `episodes` |
| **Semantic** | Subject/predicate/object knowledge triples extracted from interactions. | `facts` |
| **Procedural** | Skill registry — crystallised interaction patterns that bypass the full reasoning pipeline. | `skills` |
| **User Context** | Per-caller profiles with style dimensions and learned behavior parameters. | `callers` |

### Skill crystallisation (Procedural memory)

When the same intent + tool pattern repeats, it progresses through a maturity lifecycle:

```
progenitor  (1–3 activations)
     ↓
 committed  (4–9 activations)
     ↓
   mature   (10+ activations) ← pipeline shortcut activated
```

A `mature` skill causes the pipeline to skip Phases 4–5 (Reason + Plan) and execute the cached tool sequence directly — reducing latency and LLM cost on repeated patterns.

---

## Caller Profiling & Adaptation

Every caller gets a `CallerProfile` that evolves over interactions via **Exponential Moving Average**:

Profile confidence grows with interaction count:

Below 5 interactions the agent relies on current-message signals; as confidence grows the learned profile is blended in proportionally.

**Profile dimensions tracked:**

- **Style** — verbosity, formality, technical depth, use of emoji
- **Philosophy** — pragmatism, risk tolerance, innovation orientation
- **Habits** — session length, iteration tendency, peak hours _(future)_

---

## How to Use (as a library)

```python
from stem_agent import StemAgent, StemConfig
from stem_agent.gateway import create_rest_router, create_a2a_router
from fastapi import FastAPI

config = StemConfig(
    openai_api_key="sk-...",
    openai_model="gpt-4o-mini",
    system_context="You are an assistant for ACME Corp internal ops.",
    db_url="postgresql://user:pass@localhost/stem_agent",
)

agent = StemAgent(config)

# Register your domain tools
@agent.tool(name="query_crm", description="Query the CRM for customer info by ID.")
async def query_crm(customer_id: str) -> dict:
    return crm_client.get(customer_id)

@agent.tool(name="create_ticket", description="Create a support ticket.")
async def create_ticket(title: str, body: str) -> str:
    return ticket_system.create(title, body)

# Mount on your own FastAPI app
app = FastAPI()
app.include_router(create_rest_router(agent), prefix="/agent")
app.include_router(create_a2a_router(agent), prefix="/a2a")

@app.on_event("startup")
async def startup() -> None:
    await agent.initialize()   # runs DB migrations once
```

---

## Installation & Running

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ running locally (or any accessible instance)
- An OpenAI API key

### 1. Clone and install

```bash
git clone <repo-url>
cd stem_agent_platform

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# install the package in editable mode with dev extras
pip install -e ".[dev]"
```

### 2. Set up PostgreSQL

```bash
# create the database (adjust credentials as needed)
createdb stem_agent

# or via psql:
psql -U postgres -c "CREATE DATABASE stem_agent;"
```

StemAgent runs its own migrations on startup — no manual schema setup required.

### 3. Configure

Either pass values directly to `StemConfig`, or use environment variables / a `.env` file:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
SYSTEM_CONTEXT=You are a helpful assistant.
DB_URL=postgresql://postgres:postgres@localhost/stem_agent
LOG_LEVEL=INFO
```

### 4. Run the example

```bash
cd examples
uvicorn simple_assistant:app --reload --port 8000
```

### 5. Test it

```bash
# REST — chat
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"caller_id": "alice", "message": "What time is it?"}'

# REST — check alice's learned profile after a few messages
curl http://localhost:8000/agent/callers/alice/profile

# A2A — agent-to-agent task delegation
curl -X POST http://localhost:8000/a2a/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/send",
    "params": {"caller_id": "agent-b", "message": "Summarise the last 5 tickets."},
    "id": 1
  }'
```

### 6. Run tests

```bash
pytest
```

---

## Development Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Foundation — schemas, config, StemAgent shell, dependencies 
| 1 | Memory system — episodic, semantic, procedural, MemoryManager 
| 2 | Caller profile system — CallerStore, EMA learning
| 3 | Cognitive pipeline — all 8 phases in `agent_core/`
| 4 | Tools system — ToolRegistry, ToolExecutor, `@agent.tool()`
| 5 | Gateway — REST adapter, A2A adapter (JSON-RPC 2.0)
| 6 | Integration, tests, example project

---

## Out of Scope (Prototype)

The following are natural Phase 2 additions once the prototype proves useful:

- Vector embeddings / pgvector (episodic similarity search uses `LIKE` for now)
- AG-UI streaming protocol (Server-Sent Events for pipeline progress)
- UCP / AP2 (commerce and payments protocols from the paper)
- Framework adapters (AutoGen, CrewAI, LangGraph, OpenAI Agents SDK)
- Security / IAM plugins (JWT, OAuth2) — the consuming app handles auth
- Full Reflexion and Internal Debate reasoning strategies

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Data validation | Pydantic 2 + pydantic-settings |
| LLM | OpenAI API (`gpt-4o-mini` default) |
| Web framework | FastAPI + Uvicorn |
| Database | PostgreSQL via `asyncpg` |
| Testing | pytest + pytest-asyncio + httpx |

---

## References

- Shen et al. (2025) — _STEM Agent: A Self-adapting, Tool-enabled, Extensible, Multi-agent Framework_ — [arxiv.org/html/2603.22359v1](https://arxiv.org/html/2603.22359v1)
- [Linux Foundation A2A Protocol v0.3.0](https://github.com/google-a2a/A2A)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
