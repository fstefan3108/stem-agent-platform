"""StemAgent: top-level class — the single entry point for consuming projects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from stem_agent.config import StemConfig
from stem_agent.shared.logger import configure_logging, get_logger
from stem_agent.shared.schemas import AgentMessage, AgentResponse

if TYPE_CHECKING:
    from stem_agent.agent_core.pipeline import AgentCore
    from stem_agent.caller.store import CallerStore
    from stem_agent.memory.manager import MemoryManager
    from stem_agent.tools.executor import ToolExecutor
    from stem_agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class StemAgent:
    """General-purpose agent core — import, configure, extend, run.

    Consuming projects instantiate this class, register their domain tools
    via @agent.tool(), mount the gateway routers on their FastAPI app, and
    call initialize() once at startup. After that, every incoming message
    flows through handle().

    Note: subsystem imports are deferred to __init__ so that importing this
    module at the top of a file does not fail while subsystems are still stubs.
    """

    def __init__(self, config: StemConfig) -> None:
        from stem_agent.agent_core.pipeline import AgentCore
        from stem_agent.caller.store import CallerStore
        from stem_agent.memory.manager import MemoryManager
        from stem_agent.tools.executor import ToolExecutor
        from stem_agent.tools.registry import ToolRegistry

        self.config = config

        self._memory: MemoryManager = MemoryManager(db_path=config.db_path)
        self._caller_store: CallerStore = CallerStore(self._memory)
        self._tool_registry: ToolRegistry = ToolRegistry()
        self._tool_executor: ToolExecutor = ToolExecutor(registry=self._tool_registry)
        self._core: AgentCore = AgentCore(
            openai_api_key=config.openai_api_key,
            openai_model=config.openai_model,
            system_context=config.system_context,
            memory=self._memory,
            caller_store=self._caller_store,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
        )

    async def initialize(self) -> None:
        """Run DB migrations and configure logging. Call once at app startup."""
        configure_logging(self.config.log_level)
        await self._memory.initialize()
        await self._caller_store.initialize()
        logger.info("StemAgent '%s' ready.", self.config.agent_name)

    async def handle(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message through the full 8-phase pipeline."""
        return await self._core.run(message)

    def tool(self, name: str, description: str) -> Callable:
        """Decorator — register a sync or async Python function as an agent tool.

        Usage:
            @agent.tool(name="get_tickets", description="Fetch support tickets.")
            async def get_tickets(customer_id: str) -> list:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            self._tool_registry.register(fn, name=name, description=description)
            return fn
        return decorator
