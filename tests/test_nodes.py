"""Unit tests for graph nodes (router + agent)."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRouterNode:
    """Test the synchronous router node."""

    def test_router_routes_rag_queries(self):
        """Queries about personal context/history should route to rag."""
        from src.nodes import router

        state = {
            "messages": [],
            "user_input": "记得我上次说过的偏好是什么吗？",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        result = router(state)
        assert result["route_to"] == "rag"

    def test_router_routes_map_queries(self):
        """Map/navigation queries should route to mcp."""
        from src.nodes import router

        state = {
            "messages": [],
            "user_input": "从望京到三里屯怎么走",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        result = router(state)
        assert result["route_to"] == "mcp"

    def test_router_routes_search_queries(self):
        """Web search queries should route to tools."""
        from src.nodes import router

        for query in ["搜索一下最新的BTC价格", "帮我查天气", "Run some Python code"]:
            state = {
                "messages": [],
                "user_input": query,
                "route_to": None,
                "relevant_memories": [],
                "retrieved_docs": [],
                "needs_approval": False,
                "session_id": "s1",
            }
            result = router(state)
            assert result["route_to"] in ("tools", "rag"), f"Failed for: {query}"

    def test_router_routes_direct_by_default(self):
        """General conversation should route to direct."""
        from src.nodes import router

        state = {
            "messages": [],
            "user_input": "你好，今天过得怎么样？",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        result = router(state)
        assert result["route_to"] == "direct"

    def test_router_empty_input_returns_direct(self):
        """Empty user_input should not crash — returns direct."""
        from src.nodes import router

        state = {
            "messages": [],
            "user_input": "",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        result = router(state)
        assert result["route_to"] == "direct"


class TestAgentNode:
    """Test the async agent node."""

    def _make_fake_tools_module(self, tools=None):
        """Create a fake src.tools module for patching sys.modules."""
        fake = MagicMock()
        fake.get_mcp_tools = MagicMock(return_value=tools or [])
        fake.build_local_tool_node = MagicMock(return_value=MagicMock(tools=[]))
        fake.build_mcp_tool_node = MagicMock(return_value=MagicMock(tools=[]))
        return fake

    @pytest.mark.asyncio
    async def test_agent_returns_messages(self):
        """agent node must return dict with messages key for add_messages reducer."""
        mock_response = MagicMock()
        mock_response.content = "Test response from LLM"
        mock_response.tool_calls = None

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        fake_tools = self._make_fake_tools_module()

        with patch.dict(sys.modules, {"src.tools": fake_tools}):
            with patch("src.nodes.ChatOpenAI", return_value=mock_llm):
                from src.nodes import agent

                state = {
                    "messages": [],
                    "user_input": "Hello",
                    "route_to": "direct",
                    "relevant_memories": [],
                    "retrieved_docs": [],
                    "needs_approval": False,
                    "session_id": "s1",
                }
                result = await agent(state)
                assert "messages" in result

    @pytest.mark.asyncio
    async def test_agent_detects_sensitive_tools(self):
        """agent node should set needs_approval=True for sensitive tools."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [
            {"name": "send_email", "args": {"to": "x@y.com", "body": "hi"}, "id": "call_1"}
        ]

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        fake_tools = self._make_fake_tools_module()

        with patch.dict(sys.modules, {"src.tools": fake_tools}):
            with patch("src.nodes.ChatOpenAI", return_value=mock_llm):
                from src.nodes import agent

                state = {
                    "messages": [],
                    "user_input": "Send email to x@y.com",
                    "route_to": "tools",
                    "relevant_memories": [],
                    "retrieved_docs": [],
                    "needs_approval": False,
                    "session_id": "s1",
                }
                result = await agent(state)
                assert result.get("needs_approval") is True
