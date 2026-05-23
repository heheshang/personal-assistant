"""
Graph builder for the PersonalAssistant (langgraph 1.2.1).

Topology::

    START → memory_retrieve → router ──┬──► rag_retrieve ──► agent ──┬──► tools ──► agent (loop)
                                       │                               │
                                       │                          [no tool_calls]
                                       │                               │
                                       └──► agent (mcp/tools/direct) ─┴──► memory_save → END

Checkpointer: InMemorySaver  (thread_id ← state.session_id)

Key conditional edges:
  1. router     → route_to value maps to branch node names
  2. agent      → "tools" if last message has tool_calls, else "memory_save"
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage

from .memory import memory_retrieve_node, memory_save_node
from .nodes import agent, router
from .rag import rag_retrieve_node
from .state import State


# ── Path functions for conditional edges ────────────────────────────────────

def _route_by(route_to: str | None) -> str:
    """
    Path function: maps router's route_to value → target node name.

    Written to state by the router node, read by add_conditional_edges.
    """
    if route_to == "rag":
        return "rag_retrieve"
    # mcp / tools / direct all go through the agent node (it unions all tools)
    return "agent"


def _agent_after_call(state: State) -> str:
    """
    Path function: decide what to do after the agent node runs.

    - If the last AIMessage has tool_calls → route to the tools node.
    - Otherwise → route to memory_save to persist context and exit.
    """
    messages: list = state.get("messages", [])
    if not messages:
        return "memory_save"

    last = messages[-1]
    # Tool calls are stored as the tool_calls attribute on AIMessage
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "memory_save"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and return the compiled PersonalAssistant StateGraph.
    """
    checkpointer = InMemorySaver()
    builder = StateGraph(State)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    builder.add_node("memory_retrieve", memory_retrieve_node)
    builder.add_node("router", router)
    builder.add_node("rag_retrieve", rag_retrieve_node)
    builder.add_node("agent", agent)
    builder.add_node("tools", _build_tools_node())
    builder.add_node("memory_save", memory_save_node)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry point
    builder.set_entry_point("memory_retrieve")

    # Linear: memory_retrieve → router
    builder.add_edge("memory_retrieve", "router")

    # Conditional branching from router on route_to value
    # path_map order: ["rag_retrieve", "agent"] matches _route_by return values
    builder.add_conditional_edges(
        source="router",
        path_fn=_route_by,
        path_map=["rag_retrieve", "agent"],
    )

    # rag path: rag_retrieve → agent (injects docs into state)
    builder.add_edge("rag_retrieve", "agent")

    # Agent → tools (if tool_calls) OR memory_save (if direct answer)
    builder.add_conditional_edges(
        source="agent",
        path_fn=_agent_after_call,
        path_map={"tools": "tools", "memory_save": "memory_save"},
    )

    # Tool result bounces back to agent for response synthesis
    builder.add_edge("tools", "agent")

    # Exit: memory_save → END
    builder.add_edge("memory_save", END)

    # ── Compile ────────────────────────────────────────────────────────────────

    return builder.compile(checkpointer=checkpointer)


def _build_tools_node():
    """
    Build the ToolNode that executes tool calls.

    Combines local tools (web_search, code_executor) with warmed-up MCP tools.
    Must be called after MCP singleton has been initialized.
    Cached after first call to avoid repeated ToolNode reconstructions.
    """
    global _tools_node
    if _tools_node is not None:
        return _tools_node

    from .tools import build_local_tool_node, build_mcp_tool_node

    local_tools = build_local_tool_node()
    mcp_tools = build_mcp_tool_node()

    # Union both tool lists into one ToolNode
    all_tools = local_tools.tools + mcp_tools.tools
    from langgraph.prebuilt import ToolNode

    _tools_node = ToolNode(all_tools)
    return _tools_node


# Module-level singleton
_graph = None
_tools_node = None  # cached ToolNode to avoid repeated reconstructions


def get_graph() -> StateGraph:
    """
    Return the compiled graph singleton.

    Thread-safe: graph is immutable; state is isolated per thread_id
    via the checkpointer.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
