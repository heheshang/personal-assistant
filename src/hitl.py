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

def _check_pending_start(state: State) -> str:
    """
    Path function: START → hitl or memory_retrieve.

    On first call: pending_approval=None → memory_retrieve (normal flow)
    On second call (post-approval): Redis has pending record with approved=True/False
      → hitl to process the approval.
    """
    session_id = state.get("session_id", "")
    if not session_id:
        return "memory_retrieve"

    # Check Redis for a pending approval record
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
        # No pending record — this node was entered in error (e.g. from agent
        # after a non-sensitive tool call). Preserve state and exit.
        return {}

    approved = record.get("approved")
    tool_name = record.get("tool_name", "")
    tool_args = record.get("args", {})
    tool_call_id = record.get("tool_call_id", "")

    if approved is None:
        # First call: pending not yet approved — keep in Redis, signal wait to user
        return {
            "needs_approval": True,
            "pending_approval": None,
        }

    if approved is True and tool_name:
        # Second call (post-approval): execute the approved tool
        tool_result = _execute_tool(tool_name, tool_args)
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        return {
            "needs_approval": False,
            "pending_approval": None,
            "messages": [tool_msg],
        }
    else:
        # Rejected — skip tool, clear pending_approval
        return {"needs_approval": False, "pending_approval": None}


def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a single tool by name with arguments. Returns result string."""
    try:
        from .tools import get_local_tools, get_mcp_tools

        # Collect all available tools (same union as builder._build_tools_node)
        all_tools = get_local_tools() + get_mcp_tools()

        # Find the tool by name
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

    After hitl_node runs:
    - If tool was executed (ToolMessage in messages) → 'memory_save' (end flow)
    - If rejected or no pending → 'memory_save' (end flow)
    """
    messages: list = state.get("messages", [])
    last = messages[-1] if messages else None

    # Tool executed → save to memory and end
    if last and hasattr(last, "type") and getattr(last, "type", None) == "tool":
        return "memory_save"

    # No tool result → save and end
    return "memory_save"