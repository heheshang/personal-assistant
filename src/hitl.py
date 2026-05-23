"""
Human-in-the-Loop approval module (langgraph 1.2.1 compatible).

No Command(suspend=True) in langgraph 1.2.1 — uses Redis as coordinator between
two separate /chat calls:

  Call 1: sensitive tool detected → hitl writes pending to Redis → returns to user
  Call 2 (after approval): START conditional sees pending → hitl executes tool
    → inject result into messages → routes to agent for synthesis

Flow for Call 2 (post-approval):
  START → [pending? y] → hitl → tools (execute approved tool)
                                   → agent (synthesize) → memory_save → END
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB
from .state import State

# Redis key prefix for HitL pending records
HITL_PENDING_PREFIX = "hitl:pending:"


# ── Redis store ────────────────────────────────────────────────────────────────

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Redis client singleton for HitL store."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
    return _redis_client


def store_pending(session_id: str, record: dict[str, Any]) -> None:
    """
    Store a pending approval record in Redis.

    Args:
        session_id: Session identifier
        record: keys → tool_name, args, tool_call_id, user_input, approved (bool or None)
    """
    client = _get_redis()
    key = f"{HITL_PENDING_PREFIX}{session_id}"
    client.set(key, json.dumps(record), ex=3600)


def get_pending(session_id: str) -> dict[str, Any] | None:
    """Retrieve a pending approval record from Redis, or None."""
    client = _get_redis()
    key = f"{HITL_PENDING_PREFIX}{session_id}"
    data = client.get(key)
    return json.loads(data) if data else None


def clear_pending(session_id: str) -> None:
    """Delete the pending approval record after decision is consumed."""
    client = _get_redis()
    key = f"{HITL_PENDING_PREFIX}{session_id}"
    client.delete(key)


# ── START conditional — check Redis for pending approval ──────────────────────

def _check_pending_start(session_id: str) -> str:
    """
    Path function for the START → hitl | memory_retrieve edge.

    Checks Redis for a pending HitL approval for this session.
    Returns 'hitl' if approval record exists, 'memory_retrieve' otherwise.
    """
    if not session_id:
        return "memory_retrieve"
    record = get_pending(session_id)
    if record and record.get("approved") is not None:
        return "hitl"
    return "memory_retrieve"


# ── LangGraph HitL node ────────────────────────────────────────────────────────

def hitl_node(state: State) -> dict:
    """
    LangGraph node that handles post-approval tool execution.

    Reads the pending approval record from Redis (set by previous /chat call
    that detected a sensitive tool).

    - If approved=True: executes the tool via ToolNode, injects result into
      messages, returns needs_approval=False so next edge routes to 'agent'.
    - If approved=False: clears pending, returns needs_approval=False,
      routes to 'memory_save' (skip tool).
    - If approved=None (not yet decided): clears pending, routes to 'memory_save'.
    """
    session_id = state.get("session_id", "")
    record = get_pending(session_id)

    if not record:
        # No pending record — treat as normal continuation
        return {"needs_approval": False}

    approved = record.get("approved")
    tool_name = record.get("tool_name", "")
    tool_args = record.get("args", {})
    tool_call_id = record.get("tool_call_id", "")

    # Clear pending immediately — one-shot
    clear_pending(session_id)

    if approved is True and tool_name:
        # Execute the approved tool
        tool_result = _execute_tool(tool_name, tool_args)
        from langchain_core.messages import AIMessage, ToolMessage

        # Build the tool-result message that would normally come from ToolNode
        tool_msg = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        return {
            "needs_approval": False,
            "messages": [tool_msg],
        }
    else:
        # Rejected or not-yet-decided — skip tool, continue to memory_save
        return {"needs_approval": False}


def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a single tool by name with arguments. Returns result string."""
    try:
        from .tools import get_mcp_tools, build_local_tool_node, build_mcp_tool_node

        # Collect all available tools (same union as builder._build_tools_node)
        local_node = build_local_tool_node()
        mcp_node = build_mcp_tool_node()
        all_tools = local_node.tools + mcp_node.tools

        # Find the tool
        tool = None
        for t in all_tools:
            if getattr(t, "name", None) == tool_name:
                tool = t
                break

        if tool is None:
            return f"Error: tool '{tool_name}' not found"

        # Invoke synchronously
        result = tool.invoke(tool_args)
        return result
    except Exception as e:
        return f"Error executing tool {tool_name}: {e}"


# ── Edge path functions ────────────────────────────────────────────────────────

def _hitl_after_approval(state: State) -> str:
    """
    Path function: after hitl_node processes approval decision.

    - If needs_approval=False and tool was executed (messages updated) → 'agent'
    - Otherwise → 'memory_save'
    """
    needs_approval = state.get("needs_approval", True)
    if needs_approval:
        return "memory_save"

    messages: list = state.get("messages", [])
    last = messages[-1] if messages else None
    # If last message is a ToolMessage, tool was executed → go to agent
    if last and hasattr(last, "type") and getattr(last, "type", None) == "tool":
        return "agent"
    return "memory_save"
