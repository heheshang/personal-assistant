"""Graph nodes for the personal assistant StateGraph (langgraph 1.2.1).

- router  (sync):   reads user_input, writes route_to
- agent   (async):  reads relevant_memories+retrieved_docs via build_system_prompt,
                     binds local+MCP tools
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY
from .rag.retriever import build_system_prompt
from .state import State
from .tools import build_local_tool_node, build_mcp_tool_node, get_mcp_tools


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
        "地图", "导航", "地点", "距离", "路线", "位置",
        "map", "navigate", "directions", "distance", "location",
        "高德", "amap",
    ]
    if any(kw in user_input for kw in mcp_triggers):
        return {"route_to": "mcp"}

    # Code / web search / calculation
    tool_triggers = [
        "代码", "执行", "运行", "计算", "写程序",
        "search", "查询", "搜索",
        "code", "execute", "run", "calculate", "python",
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
    # Ensure MCP tools are warmed up (sync accessor — returns cached list)
    get_mcp_tools()

    # Collect all available tools (local + mcp)
    local_node = build_local_tool_node()
    mcp_node = build_mcp_tool_node()

    # Merge tool lists — each ToolNode exposes its tools; we union them
    all_tools = local_node.tools + mcp_node.tools

    # Build system prompt with memories + RAG docs
    system_message = build_system_prompt(state)

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )
    llm_with_tools = llm.bind_tools(all_tools)

    # Existing messages + new user input
    conversation = list(state.get("messages", []))
    conversation.append(HumanMessage(content=state.get("user_input", "")))

    # Invoke
    response = await llm_with_tools.ainvoke(
        [
            ("system", system_message),
            *conversation,
        ]
    )

    return {"messages": [response]}
