"""
Graph builder for the PersonalAssistant (langgraph 1.2.1).

Topology::

    START ──► start_check ──┬──► [hitl] ──► tools ──► agent
                             │                        ▲
                             └─► memory_retrieve ──► router ──┬──► rag_retrieve
                                                                │
                                                                └──► agent (mcp/tools/direct)

Post-approval flow (2nd /chat call):
  /chat → start_check sees Redis approved=True → hitl → tools → agent → memory_save → END

Checkpointer: InMemorySaver  (thread_id ← state.session_id)

Key conditional edges:
  1. start_check → hitl if Redis has approved pending, else memory_retrieve
  2. router      → route_to value maps to branch node names
  3. agent       → needs_approval=True → hitl; else → tools if tool_calls else memory_save
  4. hitl        → executed tool → 'agent'; skipped/rejected → 'memory_save'
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from .hitl import _check_pending_start, hitl_node
from .memory import memory_retrieve_node, memory_save_node
from .nodes import agent, router
from .rag import rag_retrieve_node
from .state import State


# ── Path functions for conditional edges ────────────────────────────────────

def _route_by(state: State | str | None) -> str:
    """
    Path function: maps router's route_to value → target node name.

    Accepts either a State dict (graph calling convention) or a bare
    route_to string (legacy direct-call interface for tests).
    """
    route_to = state.get("route_to") if isinstance(state, dict) else state
    if route_to == "rag":
        return "rag_retrieve"
    if route_to in ("tools", "mcp", "direct"):
        return "agent"
    return "agent"


def _agent_after_call(state: State) -> str:
    """
    Path function: decide what to do after the agent node runs.

    - If needs_approval=True → hitl (sensitive tool, wait for approval)
    - If last AIMessage has tool_calls → tools
    - Otherwise → memory_save
    """
    if state.get("needs_approval"):
        return "hitl"

    messages: list = state.get("messages", [])
    if not messages:
        return "memory_save"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "memory_save"


def _hitl_after_approval(state: State) -> str:
    """
    Path function: after hitl_node processes approval decision.

    After hitl_node runs, pending_approval is always None (cleared).
    - If last message is ToolMessage (tool executed post-approval) → 'agent'
    - Otherwise → 'memory_save' (don't loop)
    """
    messages: list = state.get("messages", [])
    last = messages[-1] if messages else None

    # Tool executed → synthesize result
    if last and hasattr(last, "type") and getattr(last, "type", None) == "tool":
        return "agent"
    return "memory_save"


# ── Start node ───────────────────────────────────────────────────────────────

def start_node(state: State) -> dict:
    """
    Entry node that reads Redis to decide whether to process a pending approval.

    Returns {} — all logic is in the conditional edge path function.
    """
    return {}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and return the compiled PersonalAssistant StateGraph.
    """
    checkpointer = InMemorySaver()
    builder = StateGraph(State)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    builder.add_node("start_check", start_node)
    builder.add_node("memory_retrieve", memory_retrieve_node)
    builder.add_node("router", router)
    builder.add_node("rag_retrieve", rag_retrieve_node)
    builder.add_node("agent", agent)
    builder.add_node("hitl", hitl_node)
    builder.add_node("tools", _build_tools_node())
    builder.add_node("memory_save", memory_save_node)

    # ── Edges ─────────────────────────────────────────────────────────────────

    builder.set_entry_point("start_check")

    # START conditional: check Redis pending → hitl or memory_retrieve
    builder.add_conditional_edges(
        source="start_check",
        path=_check_pending_start,
        path_map=["hitl", "memory_retrieve"],
    )

    # Normal path: memory_retrieve → router
    builder.add_edge("memory_retrieve", "router")

    # Router conditional: rag_retrieve or agent
    builder.add_conditional_edges(
        source="router",
        path=_route_by,
        path_map=["rag_retrieve", "agent"],
    )

    # rag_retrieve → agent
    builder.add_edge("rag_retrieve", "agent")

    # Agent conditional: hitl / tools / memory_save
    builder.add_conditional_edges(
        source="agent",
        path=_agent_after_call,
        path_map={"hitl": "hitl", "tools": "tools", "memory_save": "memory_save"},
    )

    # Tool result bounces back to agent
    builder.add_edge("tools", "agent")

    # HitL conditional: execute → agent, or skip → memory_save
    builder.add_conditional_edges(
        source="hitl",
        path=_hitl_after_approval,
        path_map={"agent": "agent", "memory_save": "memory_save"},
    )

    # Exit: memory_save → END
    builder.add_edge("memory_save", END)

    return builder.compile(checkpointer=checkpointer)


def _build_tools_node():
    """
    Build the ToolNode that executes tool calls.

    Cached after first call to avoid repeated ToolNode reconstructions.
    """
    global _tools_node
    if _tools_node is not None:
        return _tools_node

    from .tools import get_local_tools, get_mcp_tools

    local_tools = get_local_tools()
    mcp_tools = get_mcp_tools()

    # Union both tool lists into one ToolNode
    all_tools = local_tools + mcp_tools
    from langgraph.prebuilt import ToolNode

    _tools_node = ToolNode(all_tools)
    return _tools_node


# Module-level singletons
_graph = None
_tools_node = None


def get_graph() -> StateGraph:
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
