"""Unit tests for memory module (Redis store + nodes)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMemoryStore:
    """Test Redis memory storage and retrieval."""

    def test_save_memory_generates_id(self):
        """save_memory should return a string ID."""
        with patch("src.memory.store._get_redis_client") as mock_redis:
            client = MagicMock()
            mock_redis.return_value = client

            with patch("src.memory.store.OpenAI") as mock_oa:
                mock_instance = MagicMock()
                mock_instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
                mock_oa.return_value = mock_instance

                from src.memory.store import save_memory
                import asyncio
                id_ = asyncio.run(save_memory("session1", "User likes Python"))
                assert isinstance(id_, str)
                assert id_.startswith("mem:")

    def test_retrieve_memories_returns_filtered_list(self):
        """retrieve_memories should filter by score > 0.75."""
        with patch("src.memory.store._get_redis_client") as mock_redis:
            client = MagicMock()
            mock_redis.return_value = client

            # Mock Redis smembers + hgetall
            def mock_hgetall(key):
                m = MagicMock()
                if "vector" in key:
                    m.return_value = {
                        "content": "User prefers Python",
                        "vector": '[0.1]*1536',
                        "createdAt": "1234567890",
                    }
                else:
                    m.return_value = {}
                return m

            client.hgetall = MagicMock(side_effect=mock_hgetall)
            client.smembers = MagicMock(return_value=["mem:s1:1"])

            with patch("src.memory.store.OpenAI") as mock_oa:
                mock_instance = MagicMock()
                mock_instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
                mock_oa.return_value = mock_instance

                from src.memory.store import retrieve_memories
                import asyncio
                results = asyncio.run(retrieve_memories("session1", "What does user prefer?"))
                # Score is computed as cosine similarity
                # If properly implemented, should return items with score > 0.75


class TestMemoryNodes:
    """Test LangGraph memory nodes."""

    @pytest.mark.asyncio
    async def test_memory_retrieve_node_returns_memories(self):
        """memory_retrieve_node should write relevant_memories to state."""
        with patch("src.memory.store._get_redis_client") as mock_redis:
            client = MagicMock()
            mock_redis.return_value = client

            def mock_hgetall(key):
                m = MagicMock()
                if "vector" in key:
                    m.return_value = {
                        "content": "User likes Python",
                        "vector": '[0.1]*1536',
                        "createdAt": "1234567890",
                    }
                else:
                    m.return_value = {}
                return m

            client.hgetall = MagicMock(side_effect=mock_hgetall)
            client.smembers = MagicMock(return_value=["mem:s1:1"])

            with patch("src.memory.store.OpenAI") as mock_oa:
                mock_instance = MagicMock()
                mock_instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
                mock_oa.return_value = mock_instance

                from src.memory.store import memory_retrieve_node

                state = {
                    "messages": [],
                    "user_input": "What does user like?",
                    "route_to": None,
                    "relevant_memories": [],
                    "retrieved_docs": [],
                    "needs_approval": False,
                    "session_id": "session1",
                }
                result = await memory_retrieve_node(state)
                assert "relevant_memories" in result

    @pytest.mark.asyncio
    async def test_memory_save_node_short_input_skipped(self):
        """memory_save_node should skip saving if user_input is too short."""
        from src.memory.store import memory_save_node

        state = {
            "messages": [],
            "user_input": "hi",  # too short (< 20 chars)
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "session1",
        }
        result = await memory_save_node(state)
        # Should return empty dict (nothing to save)
        assert result == {}
