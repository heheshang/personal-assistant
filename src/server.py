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


class ApprovalRequest(BaseModel):
    """POST /approvals/{session_id} request body."""

    action: str
    """'approve' or 'reject'."""


class ApprovalResponse(BaseModel):
    """POST /approvals/{session_id} response body."""

    status: str
    message: str


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up MCP tools and compile the graph on startup."""
    # Pre-warm MCP tools (async singleton init)
    from .tools.mcp import init_mcp_tools

    try:
        import asyncio
        await asyncio.wait_for(init_mcp_tools(), timeout=5.0)
    except asyncio.TimeoutError:
        print("[startup] MCP tools warm-up timed out (npx npm timeout), skipping")
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
        "pending_approval": None,
        "session_id": session_id,
    }

    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )

    # Extract final AIMessage content
    messages: list = result.get("messages", [])
    final_content = ""
    for msg in reversed(messages):
        # Get content from either dict (raw) or message object
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        # Handle content blocks (list of {type, text, ...}) or plain string
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    final_content = block.get("text", "")
                    break
        elif isinstance(content, str) and content:
            final_content = content
        if final_content:
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
        "pending_approval": None,
        "session_id": session_id,
    }

    async def event_generator():
        """Stream graph events as SSE data: SSE text strings."""
        async for event in graph.astream_events(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            if not isinstance(event, dict):
                continue
            if event.get("event") != "on_chat_model_stream":
                continue

            chunk = event.get("data", {}).get("chunk")
            if not chunk:
                continue

            content = getattr(chunk, "content", []) or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        yield f"data: {text}\n\n"
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


# ── HitL Approval Endpoints ───────────────────────────────────────────────────

@app.get("/approvals/{session_id}", response_model=dict | None)
async def get_approval(session_id: str):
    """
    Check if there is a pending approval for this session.

    Returns the pending approval record from Redis, or None if none exists.
    """
    from .hitl import get_pending

    record = get_pending(session_id)
    if record is None:
        return None
    return {
        "tool_name": record.get("tool_name"),
        "args": record.get("args", {}),
        "tool_call_id": record.get("tool_call_id"),
        "user_input": record.get("user_input"),
    }


@app.post("/approvals/{session_id}", response_model=ApprovalResponse)
async def handle_approval(session_id: str, req: ApprovalRequest):
    """
    Submit an approval or rejection decision for a pending tool call.

    On approve: sets approved=True in Redis record, then calls /chat to resume.
    On reject: sets approved=False, then calls /chat to cancel tool.
    """
    from .hitl import get_pending, store_pending

    record = get_pending(session_id)
    if record is None:
        return ApprovalResponse(
            status="error",
            message=f"No pending approval for session {session_id}",
        )

    action = req.action.lower()
    if action not in ("approve", "reject"):
        return ApprovalResponse(
            status="error",
            message="action must be 'approve' or 'reject'",
        )

    # Update the approval record in Redis
    record["approved"] = action == "approve"
    store_pending(session_id, record)

    # Now invoke /chat to continue the graph with the updated Redis state
    from .hitl import get_pending as gp  # refresh after store_pending
    from langchain_core.messages import HumanMessage

    _ = gp(session_id)  # ensure fresh read

    graph = get_graph()
    initial_state: State = {
        "messages": [HumanMessage(content="[approval_resume]")],
        "user_input": "[approval_resume]",
        "route_to": None,
        "relevant_memories": [],
        "retrieved_docs": [],
        "needs_approval": False,
        "pending_approval": None,
        "session_id": session_id,
    }

    try:
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )
        messages: list = result.get("messages", [])
        final_content = ""
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if content:
                final_content = content
                break
        return ApprovalResponse(
            status="approved" if action == "approve" else "rejected",
            message=final_content,
        )
    except Exception as e:
        return ApprovalResponse(
            status="error",
            message=f"Graph invocation failed: {e}",
        )
