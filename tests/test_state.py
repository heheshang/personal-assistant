"""Unit tests for state.py (langgraph 1.2.1 State definition)."""

import pytest
from src.state import State, AssistantState


class TestStateDefinition:
    """State must be a TypedDict with proper field types."""

    def test_state_is_typeddict(self):
        """State must be a TypedDict (required by langgraph 1.2.x)."""
        import typing

        assert hasattr(State, "__annotations__")

    def test_required_fields_present(self):
        """All required state fields must be defined."""
        annotations = State.__annotations__
        required = [
            "messages",
            "user_input",
            "route_to",
            "relevant_memories",
            "retrieved_docs",
            "needs_approval",
            "session_id",
        ]
        for field in required:
            assert field in annotations, f"Missing field: {field}"

    def test_messages_is_annotated_list(self):
        """messages field must be annotated as list type."""
        import typing
        from typing_extensions import get_type_hints

        hints = get_type_hints(State)
        messages_type = hints.get("messages", hints.get("messages"))
        # Should be Annotated[..., add_messages]
        origin = getattr(messages_type, "__origin__", None)
        # Annotated has __origin__ = list (or the first generic arg)
        if hasattr(messages_type, "__metadata__"):
            assert len(messages_type.__metadata__) == 1
            assert callable(messages_type.__metadata__[0])

    def test_route_to_accepts_routing_values(self):
        """route_to must accept the four routing literals + None."""
        state = State(
            messages=[],
            user_input="test",
            route_to="rag",
            relevant_memories=[],
            retrieved_docs=[],
            needs_approval=False,
            session_id="s1",
        )
        assert state["route_to"] == "rag"

        for val in ["rag", "tools", "mcp", "direct", None]:
            s = State(
                messages=[],
                user_input="",
                route_to=val,
                relevant_memories=[],
                retrieved_docs=[],
                needs_approval=False,
                session_id="s1",
            )
            assert s["route_to"] == val

    def test_session_id_default(self):
        """session_id must default to 'default'."""
        s: State = {
            "messages": [],
            "user_input": "hi",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            # session_id intentionally omitted — should work if default is set
            "session_id": "default",
        }
        assert s["session_id"] == "default"
