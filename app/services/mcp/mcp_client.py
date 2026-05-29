"""
MCP Client Manager
Connects to external MCP servers via SSE transport
Official Anthropic MCP Python client

External servers:
    filesystem : http://localhost:8001/sse
    websearch  : http://localhost:8002/sse
"""

import asyncio
import concurrent.futures
import logging
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class MCPTool:
    """Represents a discovered MCP tool."""
    name:        str
    description: str
    parameters:  Dict[str, Any]
    server:      str
    server_url:  str


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    url:  str


# ============================================================
# MCP CLIENT MANAGER
# ============================================================

class MCPClientManager:
    """
    Connects to external MCP servers via SSE transport.

    Manages:
        - Server connections (filesystem, websearch)
        - Tool discovery
        - Tool execution
        - Health checking
    """

    DEFAULT_SERVERS = {
        "filesystem": "http://localhost:8001/sse",
        "websearch":  "http://localhost:8002/sse",
    }

    def __init__(
        self,
        filesystem_url: str = "http://localhost:8001/sse",
        websearch_url:  str = "http://localhost:8002/sse",
        timeout:        int = 30,
    ):
        self.servers: Dict[str, MCPServerConfig] = {
            "filesystem": MCPServerConfig(
                name="filesystem",
                url=filesystem_url,
            ),
            "websearch": MCPServerConfig(
                name="websearch",
                url=websearch_url,
            ),
        }
        self.timeout = timeout
        self._tools:  Dict[str, MCPTool] = {}
        self._health: Dict[str, bool] = {
            "filesystem": False,
            "websearch":  False,
        }
        self._ready = False

        logger.info("=" * 60)
        logger.info("✅ MCPClientManager initialized")
        logger.info(f"   FileSystem : {filesystem_url}")
        logger.info(f"   WebSearch  : {websearch_url}")
        logger.info("=" * 60)

    # ============================================================
    # ASYNC HELPER
    # ============================================================

    def _run_async(self, coro) -> Any:
        """
        Run an async coroutine safely whether or not
        an event loop is already running.
        """
        try:
            asyncio.get_running_loop()
            # Running loop exists (FastAPI) — run in a thread pool
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run() directly
            return asyncio.run(coro)

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def check_server_health(self, server_name: str) -> bool:
        """TCP socket health check — avoids hanging on SSE stream."""
        config = self.servers.get(server_name)
        if not config:
            return False

        parsed = urlparse(config.url)
        host   = parsed.hostname
        port   = parsed.port or 80

        try:
            with socket.create_connection((host, port), timeout=5):
                self._health[server_name] = True
                return True
        except Exception:
            self._health[server_name] = False
            return False

    def check_all_health(self) -> Dict[str, bool]:
        """Check health of all configured servers."""
        for name in self.servers:
            alive  = self.check_server_health(name)
            status = "✅ online" if alive else "❌ offline"
            logger.info(f"   MCP {name}: {status}")
        return self._health.copy()

    def is_server_online(self, server_name: str) -> bool:
        return self._health.get(server_name, False)

    def is_ready(self) -> bool:
        return self._ready

    def set_ready(self, value: bool):
        self._ready = value

    # ============================================================
    # TOOL EXECUTION
    # ============================================================

    def call_tool(
        self,
        server_name: str,
        tool_name:   str,
        arguments:   Dict[str, Any],
    ) -> str:
        """
        Call a tool on an external MCP server via SSE.

        Args:
            server_name: "filesystem" or "websearch"
            tool_name:   Tool to call (e.g. "read_file")
            arguments:   Tool arguments dict

        Returns:
            Tool result as string
        """
        config = self.servers.get(server_name)
        if not config:
            return f"Unknown server: '{server_name}'"

        logger.info(f"🔧 MCP call: {server_name}.{tool_name}({arguments})")

        try:
            result = self._run_async(
                self._call_tool_async(
                    url=config.url,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )
            logger.info(
                f"✅ MCP {server_name}.{tool_name}: "
                f"{len(str(result))} chars"
            )
            return result

        except Exception as e:
            logger.error(
                f"❌ MCP {server_name}.{tool_name} error: {e}",
                exc_info=True,
            )
            return f"MCP tool error: {str(e)}"

    async def _call_tool_async(
        self,
        url:       str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Async MCP tool call via SSE transport."""
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                )

                if result.content:
                    parts = []
                    for c in result.content:
                        if hasattr(c, "text"):
                            parts.append(c.text)
                        else:
                            parts.append(str(c))
                    return "\n".join(parts)

                return "No result returned"

    # ============================================================
    # TOOL DISCOVERY
    # ============================================================

    def discover_tools(self, server_name: str) -> List[MCPTool]:
        """
        Discover available tools from an MCP server.

        Args:
            server_name: Server to query

        Returns:
            List of discovered MCPTool instances
        """
        config = self.servers.get(server_name)
        if not config:
            return []

        try:
            tools = self._run_async(
                self._discover_tools_async(
                    server_name=server_name,
                    url=config.url,
                )
            )
            for t in tools:
                self._tools[t.name] = t
            return tools

        except Exception as e:
            logger.warning(
                f"⚠️  Tool discovery failed for '{server_name}': {e}"
            )
            return []

    async def _discover_tools_async(
        self,
        server_name: str,
        url:         str,
    ) -> List[MCPTool]:
        """Async tool discovery from MCP server."""
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                response = await session.list_tools()
                tools    = []

                for tool in response.tools:
                    params = {}
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        props  = tool.inputSchema.get("properties", {})
                        params = {
                            k: v.get("type", "string")
                            for k, v in props.items()
                        }

                    mcp_tool = MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=params,
                        server=server_name,
                        server_url=url,
                    )
                    tools.append(mcp_tool)
                    logger.info(f"   🔧 Discovered: {server_name}.{tool.name}")

                return tools

    def discover_all_tools(self) -> Dict[str, List[MCPTool]]:
        """Discover tools from all online servers."""
        discovered = {}
        for name in self.servers:
            if self.is_server_online(name):
                tools = self.discover_tools(name)
                discovered[name] = tools
                logger.info(f"   {name}: {len(tools)} tools discovered")
            else:
                logger.warning(f"   {name}: offline — skipping discovery")
                discovered[name] = []
        return discovered

    # ============================================================
    # CONVENIENCE WRAPPERS
    # ============================================================

    def read_file(self, file_path: str) -> str:
        """Read a file via MCP filesystem server."""
        return self.call_tool(
            server_name="filesystem",
            tool_name="read_file",
            arguments={"file_path": file_path},
        )

    def list_files(self, directory: str = "source") -> str:
        """List files via MCP filesystem server."""
        return self.call_tool(
            server_name="filesystem",
            tool_name="list_files",
            arguments={"directory": directory},
        )

    def search_files(self, pattern: str) -> str:
        """Search files via MCP filesystem server."""
        return self.call_tool(
            server_name="filesystem",
            tool_name="search_files",
            arguments={"pattern": pattern},
        )

    def get_file_info(self, file_path: str) -> str:
        """Get file info via MCP filesystem server."""
        return self.call_tool(
            server_name="filesystem",
            tool_name="get_file_info",
            arguments={"file_path": file_path},
        )

    def web_search(
        self,
        query:       str,
        max_results: int = 5,
    ) -> str:
        """Web search via MCP websearch server."""
        return self.call_tool(
            server_name="websearch",
            tool_name="web_search",
            arguments={
                "query":       query,
                "max_results": max_results,
            },
        )

    def web_search_news(
        self,
        query:      str,
        time_limit: str = "m",
    ) -> str:
        """News search via MCP websearch server."""
        return self.call_tool(
            server_name="websearch",
            tool_name="web_search_news",
            arguments={
                "query":      query,
                "time_limit": time_limit,
            },
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(self) -> Dict[str, Any]:
        """Full status report."""
        return {
            "ready":   self._ready,
            "servers": {
                name: {
                    "url":    cfg.url,
                    "online": self._health.get(name, False),
                }
                for name, cfg in self.servers.items()
            },
            "tools_discovered": len(self._tools),
            "tools": [
                {
                    "name":        t.name,
                    "server":      t.server,
                    "description": t.description,
                }
                for t in self._tools.values()
            ],
        }

    def __repr__(self) -> str:
        return (
            f"MCPClientManager("
            f"ready={self._ready}, "
            f"servers={list(self.servers.keys())}, "
            f"tools={list(self._tools.keys())})"
        )