# 踩坑记录

> 项目中遇到的所有坑及解决方案。持续更新。

---

## 1. Checkpoint 混串（生产事故高发区）

**现象：** A 用户看到 B 用户的对话历史。

**根因：** `thread_id` 来自请求 body，任何用户可以传别人的 sessionId。

**解法：**
```python
# session_id 必须来自鉴权层，不信任请求 body
session_id = f"{req.user.id}:{req.body.conversation_id}"
```

---

## 2. `add_edge("agent", "tools")` 无条件跳转 → 死循环

**现象：** 无 tool_calls 时 Graph 在 agent ↔ tools 之间无限循环。

**根因：** 用了 `add_edge` 而非 `add_conditional_edges` 判断是否真的有工具调用。

**解法：** 第二个 conditional edge 判断 `tool_calls` 是否存在：
```python
builder.add_conditional_edges(
    source="agent",
    path_fn=lambda s: "tools" if s.get("messages", [])[-1].tool_calls else "memory_save",
    path_map={"tools": "tools", "memory_save": "memory_save"},
)
```

---

## 3. `messages` 用 dict 而非 LangChain Message 对象

**现象：** `add_messages` reducer 报类型错误，或消息被覆盖。

**根因：** 初始化时直接传 `{"role": "user", "content": "..."}` 而非 `HumanMessage(...)`。

**解法：**
```python
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage(content=req.message)],  # ✓
    # "messages": [{"role": "user", "content": req.message}],  # ✗
    ...
}
```

---

## 4. `MemorySaver` 已重命名为 `InMemorySaver`

**现象：** `ImportError: cannot import name 'MemorySaver' from 'langgraph'`

**根因：** langgraph 1.2.x 将 `MemorySaver` 重命名为 `InMemorySaver`。

**解法：**
```python
from langgraph.checkpoint.memory import InMemorySaver  # langgraph 1.2.x
```

---

## 5. MCP Client 每次请求重新握手超时

**现象：** 第一次请求高德地图工具时超时。

**根因：** `MultiServerMCPClient` 每次 `__init__` 都要与外部进程握手（1-3秒）。

**解法：** 单例模式 + 服务启动时预热：
```python
# mcp.py
_mcp_tools: list | None = None

async def init_mcp_tools() -> list:
    global _mcp_tools
    if _mcp_tools is None:
        client = MultiServerMCPClient(connections={...})
        _mcp_tools = await client.get_tools()
    return _mcp_tools

def get_mcp_tools() -> list:
    if _mcp_tools is None:
        raise RuntimeError("Call init_mcp_tools() first")
    return _mcp_tools

# server.py lifespan
async def lifespan(app):
    await init_mcp_tools()
    yield
```

---

## 6. `build_system_prompt` 缺少明确指令

**现象：** Milvus 检索到了文档，但 LLM 完全没用上。

**根因：** 检索结果直接拼入 system message，但没有告诉 LLM「优先参考这些文档」。

**解法：** 在文档前加明确指令：
```python
if state.retrieved_docs:
    system += "\n\n## 参考知识库（请优先根据以下内容回答）\n"
    for i, doc in enumerate(state.retrieved_docs, 1):
        system += f"[文档{i}] 来源: {doc['source']}\n{doc['content']}\n\n"
```

---

## 7. RAG 检索无 score 导致无法过滤

**现象：** 低分文档（0.3）被灌入 system prompt，LLM 被误导。

**根因：** 用 `vectorstore.invoke(query)` 而非 `similarity_search_with_score(query, k=4)`。

**解法：**
```python
results = await vectorstore.similarity_search_with_score(query, k=top_k)
docs = [doc for doc, score in results if score > 0.7]
```

---

## 8. `route_to` 类型不支持动态工具名

**现象：** 路由到特定工具时类型报错。

**根因：** `routeTo` 类型是 `Literal["rag", "tools", "mcp", "direct", None]`，不包含动态工具名。

**解法：** 扩展类型并在使用处加类型断言：
```python
route_to: Annotated[
    Literal["rag", "tools", "mcp", "direct", str, None],
    lambda _, next: next,
] = None
```

---

## 9. langgraph 1.2.1 `add_conditional_edges` 的 path_map 限制

**现象：** `path_map` 传入字符串列表时行为不符预期。

**根因：** `path_map` 必须是 `dict[Hashable, str]` 或 `list[str]`（对应 path_fn 返回值的枚举），与 path_fn 返回值顺序无关。

**解法：** path_map 的 key 必须与 path_fn 返回值完全匹配：
```python
# path_fn 返回 "rag_retrieve" 或 "agent"
# path_map 必须显式映射这两个值
path_map={"rag_retrieve": "rag_retrieve", "agent": "agent"}
# 或者 list（按 path_fn 返回值枚举）
path_map=["rag_retrieve", "agent"]
```

---

## 10. `MultiServerMCPClient.get_tools()` 是 async

**现象：** `TypeError: object list can't be used in 'await' expression`

**根因：** `get_tools()` 在 0.2.x 版本是 async 方法，但代码里当同步调用。

**解法：**
```python
# 预热时用 await
_mcp_tools = await client.get_tools()

# 访问时用同步单例
def get_mcp_tools() -> list:
    if _mcp_tools is None:
        raise RuntimeError("Call init_mcp_tools() first")
    return _mcp_tools
```
