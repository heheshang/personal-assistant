"""Pytest fixtures and mocks for PersonalAssistant tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Mock Redis ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """Mock Redis client for memory module tests."""
    with patch("src.memory.store.redis.Redis") as mock:
        instance = MagicMock()
        mock.return_value = instance

        # Mock pipeline
        pipeline = MagicMock()
        pipeline.execute = MagicMock(return_value=[])
        instance.pipeline = MagicMock(return_value=pipeline)

        yield instance


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for vector similarity."""
    with patch("src.memory.store.OpenAI") as mock:
        instance = MagicMock()
        instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
        instance.embed_documents = AsyncMock(return_value=[[0.1] * 1536])
        mock.return_value = instance
        yield instance


# ── Mock Milvus ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_milvus():
    """Mock Milvus vector store for RAG module tests."""
    with patch("src.rag.retriever.Milvus") as mock:
        instance = MagicMock()
        instance.similarity_search_with_score = AsyncMock(return_value=[
            (
                MagicMock(pageContent="Test document content", metadata={"source": "test.txt"}),
                0.85,
            ),
            (
                MagicMock(pageContent="Another document", metadata={"source": "doc.pdf"}),
                0.72,
            ),
            (
                MagicMock(pageContent="Low score document", metadata={"source": "low.txt"}),
                0.55,  # below 0.7 threshold
            ),
        ])
        mock.from_existing_collection = AsyncMock(return_value=instance)
        yield instance


# ── Mock LLM ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock ChatAnthropic (MiniMax) for agent node tests."""
    with patch("langchain_anthropic.ChatAnthropic") as mock:
        instance = MagicMock()
        response = MagicMock()
        response.content = "Mocked LLM response"
        response.tool_calls = None
        instance.ainvoke = AsyncMock(return_value=response)
        instance.bind_tools = MagicMock(return_value=instance)
        mock.return_value = instance
        yield instance, response


# ── Mock MCP tools ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools for tool node tests."""
    tool = MagicMock()
    tool.name = "amap_directions"
    tool.description = "Get directions from Amap"
    tool.invoke = AsyncMock(return_value="Directions: Take highway G6.")
    return [tool]


# ── State fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def basic_state():
    """Minimal valid state dict for testing."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="Hello, what is the API rate limit?")],
        "user_input": "Hello, what is the API rate limit?",
        "route_to": None,
        "relevant_memories": [],
        "retrieved_docs": [],
        "needs_approval": False,
        "session_id": "test-session-001",
    }


@pytest.fixture
def state_with_memory():
    """State with relevant memories loaded."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="Continue my TypeScript project")],
        "user_input": "Continue my TypeScript project",
        "route_to": "rag",
        "relevant_memories": [
            {"id": "mem:1", "content": "User prefers TypeScript", "score": 0.92, "createdAt": 1234567890}
        ],
        "retrieved_docs": [],
        "needs_approval": False,
        "session_id": "test-session-001",
    }


@pytest.fixture
def state_with_docs():
    """State with RAG documents retrieved."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="What is the API rate limit?")],
        "user_input": "What is the API rate limit?",
        "route_to": "rag",
        "relevant_memories": [],
        "retrieved_docs": [
            {"content": "API rate limit: 1000 requests per minute", "source": "api.md", "score": 0.88}
        ],
        "needs_approval": False,
        "session_id": "test-session-001",
    }


@pytest.fixture
def state_with_tool_call():
    """State where LLM made a tool call."""
    from langchain_core.messages import AIMessage, HumanMessage

    return {
        "messages": [
            HumanMessage(content="What is the weather in Beijing?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "Beijing weather today"},
                        "id": "call_001",
                    }
                ],
            ),
        ],
        "user_input": "What is the weather in Beijing?",
        "route_to": "tools",
        "relevant_memories": [],
        "retrieved_docs": [],
        "needs_approval": False,
        "session_id": "test-session-001",
    }
