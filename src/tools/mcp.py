"""MCP client for external services (高德地图) via langchain-mcp-adapters 0.2.x.

MultiServerMCPClient requires async initialization via client.get_tools().
Use module-level singleton pattern with lazy async warm-up.
"""

from typing import Any

from langgraph.prebuilt import ToolNode

from ..config import AMAP_KEY

# ── Module-level singleton ────────────────────────────────────────────────────

_mcp_tools: list[Any] = []
_mcp_initialized: bool = False


async def _init_mcp_tools() -> list[Any]:
    """Async warm-up: initialize MCP client and load tools once."""
    global _mcp_tools, _mcp_initialized
    if _mcp_initialized:
        return _mcp_tools

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        _mcp_initialized = True
        return []

    if not AMAP_KEY:
        _mcp_initialized = True
        return []

    client = MultiServerMCPClient(
        connections={
            "amap": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@amap/mcp-server", "--key", AMAP_KEY],
            }
        }
    )
    _mcp_tools = await client.get_tools()
    _mcp_initialized = True
    return _mcp_tools


def get_mcp_tools() -> list[Any]:
    """
    Return cached MCP tools (synchronous, for use in sync ToolNode).

    If not yet initialized, returns empty list — warm-up happens on first
    async call to init_mcp_tools(). In a LangGraph async node, await
    init_mcp_tools() first.
    """
    return _mcp_tools


async def init_mcp_tools() -> list[Any]:
    """Async entry point for MCP warm-up (call from async graph nodes)."""
    return await _init_mcp_tools()


def build_tool_node(tool_names: list[str] | None = None) -> ToolNode:
    """
    Build a ToolNode with MCP tools.

    Note: This is synchronous and uses the cached tool list. Call
    init_mcp_tools() in an async context before compiling the graph
    to ensure tools are pre-warmed.
    """
    tools = _mcp_tools
    if tool_names is not None:
        tools = [t for t in tools if getattr(t, "name", None) in tool_names]
    return ToolNode(tools)
