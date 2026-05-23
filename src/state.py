"""State definition for the personal assistant StateGraph (langgraph 1.2.1)."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages
from langgraph.graph.message import MessagesState


class State(TypedDict):
    """Main state schema for the personal assistant graph.

    All fields use LangGraph's append-only patterns where appropriate.
    """

    # Chat message history — add_messages reducer ensures append-only semantics
    messages: Annotated[list[MessagesState], add_messages]
    # User's current input string
    user_input: str
    # Routing decision written by router node, read by conditional edges
    # str covers dynamically named tool destinations from agents
    route_to: Literal["rag", "tools", "mcp", "direct"] | str | None
    # Retrieved long-term memories (from Redis vector search)
    # Each dict has: id, content, score/createdAt
    relevant_memories: list[dict]
    # Retrieved documents from Milvus RAG
    retrieved_docs: list[dict]
    # When True, triggers Human-in-the-Loop approval before tool execution
    needs_approval: bool
    # Session identifier for checkpoint + memory isolation
    session_id: str


__all__ = ["State"]
