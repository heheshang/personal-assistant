"""Graph nodes for the personal assistant StateGraph (langgraph 1.2.1).

- router  (sync):   reads user_input, writes route_to
- agent   (async):  reads relevant_memories+retrieved_docs via build_system_prompt,
                     binds local+MCP tools
"""

from typing import Literal

from langchain_core.messages import HumanMessage

from .config import ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
from .state import State

# Deferred imports: tools are only needed by the agent, not the router.
# Importing them lazily avoids pulling in langchain_experimental at module load.


# ── Router ─────────────────────────────────────────────────────────────────────

def router(state: State) -> dict[Literal["route_to"], str]:
    """
    Synchronous LangGraph node: classify user intent and write route_to.

    Routing rules (keyword-based, in priority order):
      - "rag"    : query about personal data/preferences/history
      - "mcp"    : map/navigation/geography queries (高德)
      - "tools"  : code execution, web search, or calculation requests
      - "direct" : general conversational question

    Writes ``route_to`` into state for use by add_conditional_edges path_map.
    """
    user_input: str = state.get("user_input", "").strip().lower()

    if not user_input:
        return {"route_to": "direct"}

    # MCP / map / geography — 高德地图
    mcp_triggers = [
        "地图", "导航", "地点", "距离", "路线", "位置", "怎么走",
        "map", "navigate", "directions", "distance", "location",
        "高德", "amap",
    ]
    if any(kw in user_input for kw in mcp_triggers):
        return {"route_to": "mcp"}

    # Code / web search / calculation / information lookup
    tool_triggers = [
        "代码", "执行", "运行", "计算", "写程序",
        "搜索", "查询", "天气", "价格", "查一下", "帮我查",
        "search", "query", "calculate", "code", "execute", "run",
        "python", "weather", "price",
    ]
    if any(kw in user_input for kw in tool_triggers):
        return {"route_to": "tools"}

    # Personal context / preferences / history — use RAG
    rag_triggers = [
        "记得", "之前", "上次", "偏好", "习惯", "爱好",
        "remember", "previous", "last time", "my", "i usually",
        "我的", "我之前", "告诉过",
    ]
    if any(kw in user_input for kw in rag_triggers):
        return {"route_to": "rag"}

    # Default: answer directly
    return {"route_to": "direct"}


# ── Agent ──────────────────────────────────────────────────────────────────────

async def agent(state: State) -> dict[str, object]:
    """
    Async LangGraph node: invoke the LLM with injected context and tools.

    - Builds the system prompt from ``relevant_memories`` + ``retrieved_docs``
      using ``build_system_prompt``.
    - Binds both local tools (web_search, code_executor) and any warmed-up
      MCP tools (高德地图).
    - Reads the latest ``messages`` and appends the LLM response.

    Returns a dict with updated ``messages`` (LangGraph will apply add_messages
    reducer automatically).
    """
    # Deferred imports — only needed by agent, not router
    from .rag.retriever import build_system_prompt
    from .tools import get_local_tools, get_mcp_tools

    # Collect all available tools (local + mcp)
    all_tools = get_local_tools() + get_mcp_tools()

    # Build system prompt with memories + RAG docs
    system_message = build_system_prompt(state)

    # LLM — MiniMax Anthropic-compatible endpoint
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        api_key=ANTHROPIC_AUTH_TOKEN,
        base_url=ANTHROPIC_BASE_URL,
        temperature=0.7,
    )
    llm_with_tools = llm.bind_tools(all_tools)

    # Existing messages + new user input
    conversation = list(state.get("messages", []))
    conversation.append(HumanMessage(content=state.get("user_input", "")))

    # Sensitive tools that require human approval before execution
    _sensitive_tools = {"send_email", "delete_data", "transfer_money", "send_message", "delete", "remove", "code_executor"}

    needs_approval = False
    pending_tool_call = None

    # Invoke
    response = await llm_with_tools.ainvoke(
        [
            ("system", system_message),
            *conversation,
        ]
    )
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name in _sensitive_tools:
                needs_approval = True
                # Extract first sensitive tool call for HitL pending record
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_call_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                pending_tool_call = {"tool_name": name, "args": args, "tool_call_id": tool_call_id}
                break

    # Store pending approval in Redis if sensitive tool detected
    if needs_approval and pending_tool_call:
        from .hitl import store_pending
        store_pending(state.get("session_id", ""), {
            **pending_tool_call,
            "user_input": state.get("user_input", ""),
            "approved": None,
        })

    return {"messages": [response], "needs_approval": needs_approval}
