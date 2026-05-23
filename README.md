# PersonalAssistant

> LangGraph 1.2.1 RAG + Memory + MCP 个人助理，支持非流 / SSE 流式对话

[![pytest](https://img.shields.io/badge/tests-33%2F33%20%E9%80%9A%E8%BF%87-44cc44)]()
[![Python](https://img.shields.io/badge/python-3.10+-4b89d7)]()

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| LLM | `langchain-openai` (GPT-4o-mini) | 统一语言模型 |
| 编排 | `langgraph 1.2.1` StateGraph | 节点调度、Conditional Edge、Checkpoint |
| RAG | `langchain-milvus` + Milvus | 知识库检索，score > 0.7 过滤 |
| Memory | Redis + OpenAI Embeddings | 长期记忆，向量相似度 0.75 |
| Tools | `@tool` DynamicStructuredTool | 本地工具（搜索、代码执行） |
| MCP | `langchain-mcp-adapters` | 外部服务（高德地图） |
| 服务 | FastAPI + SSE | HTTP API、流式输出 |

## 架构

```
                    ┌──────────────────┐
                    │   START          │
                    └────────┬─────────┘
                             ▼
              ┌──────────────────────────┐
              │  memory_retrieve          │  ← Redis 向量检索
              │  (相关记忆注入上下文)       │
              └────────────┬─────────────┘
                            ▼
              ┌──────────────────────────┐
              │  router                  │  ← LLM 意图分类
              │  route_to ∈ {agent,      │
              │    rag_retrieve, end}    │
              └────────────┬─────────────┘
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌──────────┐
│rag_retrieve  │  │    agent       │  │   END   │
│(Milvus RAG)  │  │(tools/MCP/direct)│ └──────────┘
└──────┬───────┘  └───────┬────────┘
       │                   ▼
       │        ┌──────────────────────┐
       │        │  conditional_edge    │
       │        │  (tool_calls?)       │
       │        └──┬───────────┬──────┘
       │           ▼           ▼
       │    ┌──────────┐  ┌───────────┐
       │    │  tools   │  │memory_save│
       │    └────┬─────┘  └─────┬─────┘
       │         ▼              ▼
       └──────► agent ◄────────┘
                           (循环)
```

## 目录结构

```
personal-assistant/
├── src/
│   ├── config.py          # 环境变量（OPENAI_API_KEY、MILVUS_*、REDIS_*）
│   ├── state.py           # TypedDict State — langgraph 1.2.1 格式
│   ├── nodes.py           # router（sync）+ agent（async）
│   ├── builder.py         # StateGraph 组装 + 2 个 conditional edge
│   ├── server.py          # FastAPI：/chat（非流）+ /chat/stream（SSE）
│   ├── memory/
│   │   └── store.py       # Redis 向量记忆 + SKIP_EMBEDDING=true 可跳过
│   ├── rag/
│   │   └── retriever.py   # Milvus 检索 + build_system_prompt
│   └── tools/
│       ├── local.py       # @tool web_search + code_executor
│       └── mcp.py         # langchain-mcp-adapters 单例（MCP 超时 5 秒）
├── tests/                 # 单元测试（Mock Redis/Milvus/LLM）
├── requirements.txt
├── .env.example
└── pyproject.toml
```

## 快速开始

### 1. 安装

```bash
cd personal-assistant
pip install -r requirements.txt
cp .env.example .env
```

### 2. 配置 `.env`

```env
# LLM（必需）
OPENAI_API_KEY=sk-...

# 长期记忆（可选，跳过 embedding：SKIP_EMBEDDING=true）
REDIS_HOST=localhost
REDIS_PORT=6379

# RAG 知识库（可选）
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=knowledge_base

# Web 搜索（可选）
TAVILY_API_KEY=...

# MCP 高德地图（可选，npm npx 超时 5 秒）
AMAP_KEY=...

# 跳过 embedding（无有效 OPENAI_API_KEY 时设为 true）
SKIP_EMBEDDING=true
```

### 3. 启动

```bash
# 开发模式
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
pm2 start -n personal-assistant -- uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### 4. 调用

**非流式：**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "1+1等于几", "session_id": "user-001"}'

# {"chat_id":"...","session_id":"user-001","response":"1+1 = 2","requires_approval":false}
```

**SSE 流式：**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "1+1等于几", "session_id": "user-001"}'

# data: **1 + 1 = 2**
#
# data: [DONE]
```

**健康检查：**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## API

### `POST /chat`

非流式对话，返回完整响应。

**请求：**

```json
{
  "message": "用户消息",
  "session_id": "user-001"
}
```

**响应：**

```json
{
  "chat_id": "uuid",
  "session_id": "user-001",
  "response": "LLM 回复文本",
  "requires_approval": false
}
```

### `POST /chat/stream`

SSE 流式对话，逐块推送 `data:` 事件。

**请求：** 同 `/chat`

**响应（SSE）：**

```
data: 第一段文本
data: 第二段文本
data: [DONE]
```

### `GET /health`

返回 `{"status":"ok"}`。

## API 测试

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok"}
```

### `POST /chat` 非流式对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "1+1等于几", "session_id": "user-001"}'
```

```json
{
  "chat_id": "uuid",
  "session_id": "user-001",
  "response": "1+1 = **2**",
  "requires_approval": false
}
```

### `POST /chat/stream` SSE 流式对话

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "2+2等于几", "session_id": "user-002"}'
```

```
data:
2+2 = 4

data: [DONE]
```

### `GET /approvals/{session_id}` 查询待审批

```bash
curl http://localhost:8000/approvals/user-001
```

无待审批时返回 `null`：

```json
null
```

有待审批时返回审批详情：

```json
{
  "tool_name": "代码执行工具",
  "args": {"code": "print('hello')"},
  "tool_call_id": "call_abc123",
  "user_input": "执行这段代码"
}
```

### `POST /approvals/{session_id}` 提交审批决策

```bash
curl -X POST http://localhost:8000/approvals/user-001 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

```json
{
  "status": "approved",
  "message": "代码执行结果: hello"
}
```

reject 时：

```json
{
  "status": "rejected",
  "message": "工具调用已被拒绝"
}
```

无待审批时：

```json
{
  "status": "error",
  "message": "No pending approval for session user-001"
}
```

### `GET /docs` OpenAPI 文档

```bash
curl http://localhost:8000/docs
```

返回 Swagger UI HTML 页面。

### `GET /openapi.json` OpenAPI 规范

```bash
curl http://localhost:8000/openapi.json | jq .paths
```

```json
["/chat", "/chat/stream", "/health", "/approvals/{session_id}"]
```

## Human-in-the-Loop 审批流

### 敏感工具

`code_executor`（代码执行）等工具为敏感操作，触发前需要人工确认：

```
send_email, delete_data, transfer_money, send_message,
delete, remove, code_executor
```

### 完整链路

```
┌──────────────────────────────────────────────────────────────────┐
│ 第一次 /chat                                                   │
│                                                                  │
│  user_input: "帮我执行 print('hello')"                            │
│       │                                                        │
│       ▼                                                        │
│  agent (检测到敏感工具 code_executor)                             │
│       │                                                        │
│       ▼                                                        │
│  needs_approval=True → hitl_node                                │
│       │                                                        │
│       ▼                                                        │
│  store_pending(approved=None) → Redis                            │
│       │                                                        │
│       ▼                                                        │
│  暂停，返回给用户  ───────────────────────────────────────────────┘
│
│  用户调用 GET /approvals/{session_id}  查询待审批
│  用户调用 POST /approvals/{session_id} action=approve/reject
│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第二次 /chat（approval_resume）                                   │
│                                                                  │
│  Redis approved=True → hitl_node 读取记录                        │
│       │                                                        │
│       ├── approved=True → _execute_tool() 执行工具              │
│       │    → ToolMessage 注入 messages → agent → memory_save   │
│       │                                                        │
│       └── approved=False → 跳过工具 → memory_save → END          │
└──────────────────────────────────────────────────────────────────┘
```

### 端到端测试（模拟）

```bash
# Session A：触发审批 → 查询 → 拒绝
SESSION="hitl-001"

# 1. /chat 触发审批（敏感工具）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "执行代码 print('hello')", "session_id": "hitl-001"}'
# → needs_approval=True（图中暂停）

# 2. 查询待审批
curl http://localhost:8000/approvals/hitl-001
# → {"tool_name":"code_executor","args":{"code":"print('hello')"},"tool_call_id":"...","user_input":"执行代码"}

# 3. 拒绝
curl -X POST http://localhost:8000/approvals/hitl-001 \
  -H "Content-Type: application/json" \
  -d '{"action": "reject"}'
# → {"status":"rejected","message":"工具调用已被拒绝"}

# 4. 第二次 /chat（hitl_node 处理拒绝）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "[approval_resume]", "session_id": "hitl-001"}'
# → {"status":"rejected","message":"..."}
```

### approve 流程

```bash
SESSION="hitl-002"

# 1. 触发审批
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "用code_executor执行 print('hello world')", "session_id": "hitl-002"}'

# 2. 查询
curl http://localhost:8000/approvals/hitl-002
# → {"tool_name":"code_executor","args":{"code":"print('hello world')"},...}

# 3. 批准
curl -X POST http://localhost:8000/approvals/hitl-002 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
# → {"status":"approved","message":"hello world"}

# 4. 第二次 /chat（hitl_node 执行工具 → agent 合成 → memory_save）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "[approval_resume]", "session_id": "hitl-002"}'
# → {"status":"approved","message":"代码执行结果: hello world"}
```

### 内部实现

| 组件 | 文件 | 说明 |
|------|------|------|
| `store_pending` / `get_pending` | `src/hitl.py` | Redis 读写 pending 记录 |
| `hitl_node` | `src/hitl.py` | 审批决策处理器（approve/reject/超时） |
| `_execute_tool` | `src/hitl.py` | 工具执行（从工具列表匹配并调用） |
| `_check_pending_start` | `src/hitl.py` | START 条件边，判断是否进入 hitl_node |
| `needs_approval` | `src/nodes.py` | agent 检测到敏感工具时写入 Redis |
| `POST /approvals/{id}` | `src/server.py` | 更新 Redis approved 字段，触发第二次 /chat |

## 测试

```bash
pytest -v
pytest --cov=src --cov-report=term-missing
```

## 核心设计

### State 是共享数据槽

State 只存跨节点共享数据。用 `Annotated[type, add_messages]` 标注 `messages`，保证 reducer 追加而非覆盖。

### Memory 分层

| 类型 | 存储 | 触发条件 |
|------|------|----------|
| 会话历史 | `messages` (State) | Checkpointer 自动管理 |
| 长期记忆 | Redis + 向量 | 相似度 > 0.75 |

### RAG 必须注入 system prompt

检索到的文档不会自动被 LLM 使用。`build_system_prompt` 必须在 system message 中显式插入：

```
## 参考知识库
[文档内容...]
```

### Conditional Edge 防死循环

```
agent ── conditional_edge(tool_calls)
  ├── 有 tool_calls → tools → agent（循环）
  └── 无 tool_calls → memory_save → END
```

禁止 `add_edge("agent", "tools")` 无条件跳转。

## 环境要求

- Python 3.10+
- Redis（长期记忆）
- Milvus（RAG 知识库，可选）
- OpenAI API Key（或 `SKIP_EMBEDDING=true`）
