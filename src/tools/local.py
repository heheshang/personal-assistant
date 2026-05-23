"""Local tools using langchain DynamicStructuredTool.

Includes web search via Tavily API and code execution.
"""

import os
from typing import Any

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolNode
from ..config import TAVILY_API_KEY

def _get_tavily_api_key() -> str:
    """Retrieve Tavily API key from environment."""
    return TAVILY_API_KEY


@tool
def web_search(query: str) -> str:
    """Search the web using Tavily search API.

    Args:
        query: The search query string.

    Returns:
        Search results as a JSON string.
    """
    api_key = _get_tavily_api_key()
    if not api_key:
        return "Error: TAVILY_API_KEY not configured"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, max_results=5)
        return str(result)
    except Exception as e:
        return f"Error performing web search: {e}"


@tool
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code string to execute.

    Returns:
        Execution result or error message.
    """
    try:
        repl = PythonREPL()
        result = repl.run(code)
        return result
    except Exception as e:
        return f"Error executing code: {e}"


def get_local_tools() -> list[Any]:
    """Return all available local tools.

    Returns:
        List of DynamicStructuredTool instances.
    """
    return [web_search, code_executor]


def build_tool_node(tool_names: list[str] | None = None) -> ToolNode:
    """Build a ToolNode with specified local tools.

    Args:
        tool_names: Optional list of tool names to include.
                   If None, all local tools are included.

    Returns:
        ToolNode instance for the local tools.
    """
    if tool_names is None:
        tools = get_local_tools()
    else:
        tools = [t for t in get_local_tools() if t.name in tool_names]

    return ToolNode(tools)
