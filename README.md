# PersonalAssistant

> 用 Python LangGraph 1.2.1 RAG + Memory + MCP 架构

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| LLM | `langchain-openai` (GPT-4o-mini) | 统一语言模型 |
| 编排 | `langgraph 1.2.1` StateGraph | 节点调度、Conditional Edge、Checkpoint |
| RAG | `langchain-milvus` + Milvus 向量库 | 知识库检索，score > 0.7 过滤 |
| Memory | Redis + OpenAI Embeddings | 长期记忆，向量相似度 0.75 |
| Tools | `@tool` DynamicStructuredTool | 本地工具（搜索、代码执行） |
| MCP | `langchain-mcp-adapters` MultiServerMCPClient | 外部服务（高德地图） |
| 服务 | FastAPI + SSE | HTTP API、流式输出 |

## 架构拓扑

```
START
  │
  ▼
memory_retrieve ──────────────────────────────────────────── (?) 长期记忆
  │                                                           /
  ▼                                                          /
router ── conditional_edge(route_to) ──┬─ rag_retrieve ──► agent
  │                                   │    (Milvus)           │
  │                                   └─ agent (mcp/tools/direct)
  │                                                      │
  ▼                                                      ▼
agent ── conditional_edge(tool_calls) ──┬─ tools ──► agent (循环)
  │                                    └─ memory_save
  ▼                                       │
memory_save ─────────────────────────────────────────► END
```

## 目录结构

```
personal-assistant/
├── src/
│   ├── config.py          # 环境变量（OPENAI_API_KEY、MILVUS_*、REDIS_*、MCP_*）
│   ├── state.py           # TypedDict State — langgraph 1.2.1 格式
│   ├── nodes.py           # router（sync）+ agent（async）
│   ├── builder.py         # StateGraph 组装 + 2 个 conditional edge
│   ├── server.py          # FastAPI：/chat（非流）+ /chat/stream（SSE）
│   ├── memory/
│   │   └── store.py       # Redis 向量记忆 + memory_retrieve_node / memory_save_node
│   ├── rag/
│   │   └── retriever.py   # Milvus 检索 + rag_retrieve_node + build_system_prompt
│   └── tools/
│       ├── local.py       # @tool web_search + code_executor
│       └── mcp.py         # langchain-mcp-adapters MultiServerMCPClient 单例
├── tests/                 # 单元测试（Mock Redis/Milvus/LLM）
├── requirements.txt       # 版本锁定
├── .env.example           # 环境变量模板
└── pyproject.toml         # pytest 配置
```

## 快速开始

### 1. 安装依赖

```bash
cd personal-assistant
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入真实密钥
```

**必须配置：**

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `MILVUS_URI` | Milvus 服务器地址（如 `http://localhost:19530`） |
| `MILVUS_COLLECTION` | 向量集合名 |
| `REDIS_HOST` / `REDIS_PORT` | Redis 连接信息 |
| `TAVILY_API_KEY` | Tavily 搜索 API（如使用 web_search 工具） |

**MCP 可选配置（高德地图）：**

```env
AMAP_KEY=你的高德地图Key
```

### 3. 启动服务

```bash
# 开发模式（热重载）
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
pm2 start -n personal-assistant -- uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### 4. 调用接口

**非流式对话：**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我的API限流策略是什么？", "session_id": "user-001"}'
```

**流式输出（SSE）：**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "从望京到三里屯怎么走", "session_id": "user-001"}'
```

**健康检查：**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## 核心设计原则

### 1. State 是地基

State 只存「跨节点需要共享的数据」，不存临时计算结果。用 `Annotated[type, add_messages]` 标注 `messages` 字段，保证 reducer 正确追加而非覆盖。

### 2. Memory 要分层

| 类型 | 存储 | 用途 |
|------|------|------|
| 会话历史 | `messages` (State) | 本轮上下文，Checkpointer 自动管理 |
| 长期记忆 | Redis + 向量 | 跨会话偏好，相似度 > 0.75 触发 |

### 3. RAG 要告知 LLM

检索到文档不等于 LLM 会用。`build_system_prompt` 必须在 system message 里加明确指令：

```
## 参考知识库（请优先根据以下内容回答）
[文档内容...]
```

### 4. 工具调用要单例

MCP Client 每次初始化都要与外部进程握手，耗时 1-3 秒。`langchain-mcp-adapters` 单例预热避免每次请求超时。

### 5. Conditional Edge 防止死循环

```
agent → conditional_edge(tool_calls)
  ├── 有 tool_calls → tools → agent（循环）
  └── 无 tool_calls → memory_save → END
```

**禁止**用 `add_edge("agent", "tools")` 无条件跳转，否则会死循环。

## 运行测试

```bash
cd personal-assistant
pytest -v

# 带覆盖率
pytest --cov=src --cov-report=term-missing
```

## 环境要求

- Python 3.10+
- Redis（长期记忆）
- Milvus（知识库向量库）
- OpenAI API Key
