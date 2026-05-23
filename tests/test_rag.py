"""Unit tests for RAG module (Milvus retriever + build_system_prompt)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRetrieveDocs:
    """Test Milvus document retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_docs_filters_by_score(self):
        """Documents with score <= 0.7 must be filtered out."""
        with patch("src.rag.retriever.Milvus") as mock_milvus_cls:
            mock_instance = MagicMock()

            # Three docs: 0.85, 0.72, 0.55 — last one should be filtered
            mock_instance.similarity_search_with_score = AsyncMock(return_value=[
                (
                    MagicMock(pageContent="High relevance doc", metadata={"source": "a.md"}),
                    0.85,
                ),
                (
                    MagicMock(pageContent="Medium relevance doc", metadata={"source": "b.md"}),
                    0.72,  # just above 0.7 threshold
                ),
                (
                    MagicMock(pageContent="Low relevance doc", metadata={"source": "c.md"}),
                    0.55,  # below 0.7 → must be filtered
                ),
            ])
            mock_milvus_cls.from_existing_collection = AsyncMock(return_value=mock_instance)

            from src.rag.retriever import retrieve_docs

            docs = await retrieve_docs("API rate limit")
            assert len(docs) == 2
            assert all(d["score"] > 0.7 for d in docs)

    @pytest.mark.asyncio
    async def test_retrieve_docs_empty_when_all_low_score(self):
        """Empty list when all docs are below threshold."""
        with patch("src.rag.retriever.Milvus") as mock_milvus_cls:
            mock_instance = MagicMock()
            mock_instance.similarity_search_with_score = AsyncMock(return_value=[
                (
                    MagicMock(pageContent="Low score doc", metadata={"source": "low.md"}),
                    0.55,
                ),
            ])
            mock_milvus_cls.from_existing_collection = AsyncMock(return_value=mock_instance)

            from src.rag.retriever import retrieve_docs

            docs = await retrieve_docs("unrelated query")
            assert len(docs) == 0


class TestBuildSystemPrompt:
    """Test system prompt construction from state."""

    def test_build_system_prompt_no_context(self):
        """Empty state → base prompt only."""
        from src.rag.retriever import build_system_prompt

        state = {
            "messages": [],
            "user_input": "Hello",
            "route_to": None,
            "relevant_memories": [],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        prompt = build_system_prompt(state)
        assert "AI 助手" in prompt or "personal assistant" in prompt.lower()

    def test_build_system_prompt_injects_memories(self):
        """relevant_memories should appear in prompt."""
        from src.rag.retriever import build_system_prompt

        state = {
            "messages": [],
            "user_input": "Continue my project",
            "route_to": "rag",
            "relevant_memories": [
                {"id": "mem:1", "content": "User prefers TypeScript", "score": 0.92, "createdAt": 1234567890}
            ],
            "retrieved_docs": [],
            "needs_approval": False,
            "session_id": "s1",
        }
        prompt = build_system_prompt(state)
        assert "TypeScript" in prompt

    def test_build_system_prompt_injects_docs_with_directive(self):
        """retrieved_docs should appear with '优先根据以下文档回答' directive."""
        from src.rag.retriever import build_system_prompt

        state = {
            "messages": [],
            "user_input": "What is the API rate limit?",
            "route_to": "rag",
            "relevant_memories": [],
            "retrieved_docs": [
                {"content": "API rate limit: 1000 req/min", "source": "api.md", "score": 0.88}
            ],
            "needs_approval": False,
            "session_id": "s1",
        }
        prompt = build_system_prompt(state)
        assert "api.md" in prompt
        assert "1000 req/min" in prompt
        # The critical directive from the article
        assert "优先根据以下文档" in prompt or "优先" in prompt


class TestRAGNode:
    """Test the rag_retrieve_node LangGraph node."""

    @pytest.mark.asyncio
    async def test_rag_retrieve_node_writes_retrieved_docs(self):
        """rag_retrieve_node should write retrieved_docs to state."""
        with patch("src.rag.retriever.Milvus") as mock_milvus_cls:
            mock_instance = MagicMock()
            mock_instance.similarity_search_with_score = AsyncMock(return_value=[
                (
                    MagicMock(pageContent="API documentation", metadata={"source": "api.md"}),
                    0.85,
                ),
            ])
            mock_milvus_cls.from_existing_collection = AsyncMock(return_value=mock_instance)

            from src.rag.retriever import rag_retrieve_node

            state = {
                "messages": [],
                "user_input": "What is the API rate limit?",
                "route_to": "rag",
                "relevant_memories": [],
                "retrieved_docs": [],
                "needs_approval": False,
                "session_id": "s1",
            }
            result = await rag_retrieve_node(state)
            assert "retrieved_docs" in result
            assert len(result["retrieved_docs"]) == 1
