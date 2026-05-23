"""
FastAPI server for the PersonalAssistant (langgraph 1.2.1).

Endpoints:
  POST /chat          — non-streaming, waits for full response
  POST /chat/stream   — SSE streaming via graph.astream_events
  GET  /health        — liveness probe

Auth: session_id comes from request body (authenticated upstream).
      thread_id = session_id for checkpointer isolation.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .builder import get_graph
from .state import State


# ── Request / response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat and /chat/stream request body."""

    message: str
    """The user's input string."""

    session_id: str | None = None
    """Optional session ID. A UUID is generated if omitted."""


class ChatResponse(BaseModel):
    """POST /chat response body."""

    session_id: str
    response: str


class StreamChunk(BaseModel):
    """SSE data payload for streaming chunks."""

    content: str
    node: str | None = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up MCP tools and compile the graph on startup."""
    # Pre-warm MCP tools (async singleton init)
    from .tools.mcp import init_mcp_tools

    try:
        await init_mcp_tools()
    except Exception as e:
        print(f"[startup] MCP tools warm-up skipped: {e}")

    # Pre-compile the graph (happens once, reused)
    _ = get_graph()
    print("[startup] PersonalAssistant graph compiled")
    yield
    print("[shutdown]")


app = FastAPI(title="PersonalAssistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat — invokes the graph and returns the complete response.

    thread_id = session_id for checkpointer isolation.
    """
    session_id = req.session_id or str(uuid.uuid4())

    graph = get_graph()

    # Initial state — messages must be LangChain message objects for add_messages reducer
    from langchain_core.messages import HumanMessage

    initial_state: State = {
        "messages": [HumanMessage(content=req.message)],
        "user_input": req.message,
        "route_to": None,
        "relevant_memories": [],
        "retrieved_docs": [],
        "needs_approval": False,
        "session_id": session_id,
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )

    # Extract final AIMessage content
    messages: list = result.get("messages", [])
    final_content = ""
    for msg in reversed(messages):
        # Get content from either dict (raw) or message object
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if content:
            final_content = content
            break

    return ChatResponse(session_id=session_id, response=final_content)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint.

    Uses graph.astream_events with stream_mode="messages" to stream
    token-by-token chunks as SSE events.
    """
    import asyncio

    from fastapi.responses import EventSourceResponse
    from langchain_core.messages import HumanMessage

    session_id = req.session_id or str(uuid.uuid4())

    graph = get_graph()

    initial_state: State = {
        "messages": [HumanMessage(content=req.message)],
        "user_input": req.message,
        "route_to": None,
        "relevant_memories": [],
        "retrieved_docs": [],
        "needs_approval": False,
        "session_id": session_id,
    }

    async def event_generator():
        """Stream graph events as SSE data: JSON chunks."""
        async for event in graph.astream_events(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            # event is a (substream_key, payload) tuple
            # payload structure: {"event": "messages", "data": [msg_chunks]}
            key, payload = event
            data = payload.get("data", []) if isinstance(payload, dict) else []

            for msg_chunk in data:
                # msg_chunk is a dict like {"content": "...", "type": "..."}
                if isinstance(msg_chunk, dict) and msg_chunk.get("content"):
                    yield {
                        "event": "message",
                        "data": f'data: {msg_chunk["content"]}\n\n',
                    }
                elif hasattr(msg_chunk, "content") and msg_chunk.content:
                    yield {
                        "event": "message",
                        "data": f'data: {msg_chunk.content}\n\n',
                    }
            await asyncio.sleep(0)  # yield control to event loop

        yield {"event": "message", "data": "data: [DONE]\n\n"}

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}
