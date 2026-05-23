# PersonalAssistant

> LangGraph 1.2.1 RAG + Memory + MCP 个人助理，支持非流 / SSE 流式对话、敏感工具 Human-in-the-Loop 审批

[![Python](https://img.shields.io/badge/python-3.10+-4b89d7)]()

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| LLM | `langchain-anthropic` (MiniMax-M2.7) | 统一语言模型 |
| 编排 | `langgraph 1.2.1` StateGraph | 节点调度、Conditional Edge、Checkpoint |
| RAG | `langchain-qdrant` + Qdrant | 知识库检索，score > 0.7 过滤 |
| Memory | Redis + OpenAI Embeddings | 长期记忆，向量相似度 0.75 |
| Tools | `@tool` DynamicStructuredTool | 本地工具（搜索、代码执行） |
| MCP | `langchain-mcp-adapters` | 外部服务（高德地图） |
| 服务 | FastAPI + SSE | HTTP API、流式输出 |
| 协调 | Redis | HitL 审批状态存储 |

## 架构

```
                              ┌─────────────────────────────────────┐
                              │  FastAPI /chat                      │
                              │  POST /chat/stream                  │
                              │  POST /approvals/{session_id}       │
                              └──────────────┬──────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │       START                   │
                              └──────────────┬───────────────┘
                                             │
                              ┌──────────────▼───────────────┐
                              │  start_check                 │
                              │  (_check_pending_start)       │
                              │  读 Redis hitl:pending:{sid}   │
                              └──────┬────────────┬──────────┘
                                     │            │
                              ┌──────▼──────┐    │ (Redis 无待审批)
                              │  hitl       │    │
                              │  审批节点    │    │
                              └──────┬──────┘    │
                                     │            │
                              ┌──────▼────────────▼───────────┐
                              │       memory_retrieve         │
                              │  Redis 向量检索相似记忆        │
                              └──────────────┬───────────────┘
                                             │
                              ┌──────────────▼───────────────┐
                              │         router               │
                              │  LLM 意图分类                 │
                              │  route_to ∈ {agent,         │
                              │    rag_retrieve}             │
                              └──────┬────────────┬──────────┘
                                     │            │
                    ┌───────────────▼─┐    ┌──────▼──────────┐
                    │  rag_retrieve    │    │     agent       │
                    │  Qdrant RAG     │    │  LLM + Tools    │
                    └────────┬────────┘    └───────┬──────────┘
                             │                     │
                             └─────────┬───────────┘
                                       │
                              ┌────────▼─────────────┐
                              │  _agent_after_call    │
                              │  needs_approval=True  │
                              │  → hitl               │
                              │  has tool_calls       │
                              │  → tools              │
                              │  → memory_save        │
                              └────────┬─────────────┘
                                       │
                              ┌────────▼─────────────┐
                              │      tools            │
                              │  ToolNode 执行工具    │
                              └────────┬─────────────┘
                                       │ 循环回 agent
                                       ▼
                              ┌──────────────────────┐
                              │    memory_save        │
                              │  保存记忆到 Redis     │
                              └──────────┬───────────┘
                                         │
                                         ▼
                                      [END]
```

### 节点说明

| 节点 | 文件 | 职责 |
|------|------|------|
| `start_check` | `builder.py` | 入口，调用 `_check_pending_start` 读 Redis 判断是否进入 HitL |
| `hitl` | `hitl.py` | 审批处理器：pending 等待 / 执行工具 / 拒绝 |
| `memory_retrieve` | `memory/store.py` | Redis 向量检索相关记忆 |
| `router` | `nodes.py` | LLM 意图分类（rag / tools / direct） |
| `rag_retrieve` | `rag/retriever.py` | Qdrant 知识库检索 |
| `agent` | `nodes.py` | LLM 对话 + 工具调用（敏感工具写入 pending_approval） |
| `tools` | `builder.py` ToolNode | 执行 web_search / code_executor |
| `memory_save` | `memory/store.py` | 存入 Redis 长期记忆 |

## 目录结构

```
personal-assistant/
├── src/
│   ├── config.py          # 环境变量（MINIMAX_*, REDIS_*, QDRANT_*, ANTHROPIC_*）
│   ├── state.py           # TypedDict State — langgraph 1.2.1 格式
│   ├── nodes.py           # router（sync）+ agent（async）
│   ├── builder.py         # StateGraph 组装 + conditional edges
│   ├── hitl.py            # HitL 审批逻辑：_check_pending_start / hitl_node / _execute_tool
│   ├── server.py          # FastAPI：/chat（非流）+ /chat/stream（SSE）+ /approvals
│   ├── memory/
│   │   ├── store.py       # Redis 向量记忆 + SKIP_EMBEDDING=true 可跳过
│   │   └── __init__.py
│   ├── rag/
│   │   ├── retriever.py   # Qdrant 检索 + build_system_prompt
│   │   └── __init__.py
│   └── tools/
│       ├── local.py       # @tool web_search + code_executor
│       ├── mcp.py         # langchain-mcp-adapters 单例（MCP 超时 5 秒）
│       └── __init__.py
├── tests/                 # 单元测试（Mock Redis/Qdrant/LLM）
├── api.http               # HTTP 测试用例（VS Code REST Client）
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
# LLM（必需）- MiniMax
ANTHROPIC_AUTH_TOKEN=sk-ant-...
ANTHROPIC_BASE_URL=https://api.minimaxi.chat/v1
ANTHROPIC_MODEL=MiniMax-M2.7

# 长期记忆（可选，跳过 embedding：SKIP_EMBEDDING=true）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# RAG 知识库（可选）
QDRANT_URI=http://localhost:6333
QDRANT_COLLECTION=knowledge_base

# Web 搜索（可选）
TAVILY_API_KEY=...

# MCP 高德地图（可选，npm npx 超时 5 秒）
AMAP_KEY=...
AMAP_SECRET=...

# 跳过 embedding（无有效 API key 时设为 true）
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

# {"session_id":"user-001","response":"1 + 1 = **2**"}
```

**SSE 流式：**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "1+1等于几", "session_id": "user-001"}'

# data: 1 + 1 =
# data: **2**
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
  "session_id": "user-001",
  "response": "LLM 回复文本"
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

### `GET /approvals/{session_id}`

查询待审批的敏感工具调用。

**无待审批：**

```json
null
```

**有待审批：**

```json
{
  "tool_name": "code_executor",
  "args": {"code": "print('hello')"},
  "tool_call_id": "call_abc123",
  "user_input": "执行这段代码"
}
```

### `POST /approvals/{session_id}`

提交审批决策。

```bash
curl -X POST http://localhost:8000/approvals/user-001 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

**approve 响应：**

```json
{
  "status": "approved",
  "message": "hello"
}
```

**reject 响应：**

```json
{
  "status": "rejected",
  "message": "Tool execution cancelled"
}
```

**无待审批时：**

```json
{
  "status": "error",
  "message": "No pending approval for session user-001"
}
```

## Human-in-the-Loop 审批流

### 敏感工具

以下工具为敏感操作，触发前需要人工确认：

```
code_executor, send_email, delete_data, transfer_money, send_message, delete, remove
```

### 完整链路

```
┌─────────────────────────────────────────────────────────────────┐
│ 第一次 POST /chat                                               │
│                                                                 │
│  message: "帮我执行 print('hello')"                             │
│       │                                                         │
│       ▼                                                         │
│  agent → 检测到敏感工具 code_executor                            │
│       │                                                         │
│       ▼                                                         │
│  store_pending(sid, {tool_name, args, approved: None})          │
│       │  → Redis hitl:pending:{sid}                             │
│       ▼                                                         │
│  hitl_node(approved=None)                                       │
│       │                                                         │
│       ▼                                                         │
│  返回 "[等待审批] 敏感操作需您确认，请调用 POST /approvals/{sid}"  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GET /approvals/{sid}      ← 可选：查询待审批详情                │
│  POST /approvals/{sid}      ← 提交审批决策                       │
│                                                                 │
│  action=approve: 直接执行工具 → 返回工具结果                     │
│  action=reject:  清除 pending → 返回 cancelled                   │
└─────────────────────────────────────────────────────────────────┘
```

### 端到端测试

```bash
BASE="http://localhost:8000"
SID="hitl-$(date +%s)"

# Step 1: 触发敏感工具
curl -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"execute python code: print(1+1)\", \"session_id\": \"$SID\"}"
# → {"session_id":"...","response":"[等待审批] 敏感操作需您确认..."}

# Step 2: 查询待审批（可选）
curl $BASE/approvals/$SID
# → {"tool_name":"code_executor","args":{"code":"print(1+1)"},"tool_call_id":"...","user_input":"..."}

# Step 3: 审批
curl -X POST $BASE/approvals/$SID \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
# → {"status":"approved","message":"2\n"}

# reject 示例
curl -X POST $BASE/approvals/$SID \
  -H "Content-Type: application/json" \
  -d '{"action": "reject"}'
# → {"status":"rejected","message":"Tool execution cancelled"}
```

### 内部实现

| 组件 | 文件 | 说明 |
|------|------|------|
| `store_pending` / `get_pending` / `clear_pending` | `src/hitl.py` | Redis 读写 pending 记录 |
| `hitl_node` | `src/hitl.py` | 审批处理器：approved=None→等待, True→执行工具, False→拒绝 |
| `_execute_tool` | `src/hitl.py` | 根据 tool_name 查找并调用工具 |
| `_check_pending_start` | `src/hitl.py` | START 条件边，判断是否进入 hitl_node |
| `needs_approval` | `src/nodes.py` | agent 检测到敏感工具时写入 Redis |
| `POST /approvals/{id}` | `src/server.py` | approve 直接执行工具；reject 清除 pending |

## 测试

```bash
# 全部测试
pytest -v

# 覆盖率
pytest --cov=src --cov-report=term-missing

# HTTP 接口测试（VS Code REST Client）
# 打开 api.http 点击 "Send Request"
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

### HitL 通过 Redis 解耦两次调用

LangGraph 1.2.1 没有 `Command(suspend=True)`，所以用 Redis 作为两次 HTTP 调用之间的协调器：

1. 第一次 `/chat` 检测到敏感工具 → 写入 `approved=None` 到 Redis → 返回等待提示
2. 用户调用 `/approvals/{sid}` → 直接执行工具（绕过完整 graph）
3. 工具结果通过 `/approvals` 响应直接返回给用户

## 环境要求

- Python 3.10+
- Redis（长期记忆 + HitL 协调）
- Qdrant（RAG 知识库，可选）
- MiniMax API Key（`ANTHROPIC_AUTH_TOKEN`）
