"""RAG retriever module using Milvus vector database (langchain-milvus 0.3.x)."""

import os
from typing import Annotated, Any, Literal

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

from ..state import State

# ── Config (reads from env directly to avoid circular imports) ────────────────
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "personal_assistant_kb")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))


def _get_embedding_model() -> OpenAIEmbeddings:
    """Create OpenAI embedding model (text-embedding-3-small)."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)


# Module-level singleton
_vector_store: Milvus | None = None


def _get_vector_store() -> Milvus:
    """Get or create Milvus vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Milvus.from_existing_collection(
            embedding=_get_embedding_model(),
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI},
        )
    return _vector_store


def retrieve_docs(query: str, topK: int = 4) -> list[dict[str, Any]]:
    """
    Retrieve documents from Milvus with similarity score > 0.7.

    Args:
        query: Search query string.
        topK: Number of top results (default 4).

    Returns:
        List of dicts with keys: content, source, score.
    """
    try:
        store = _get_vector_store()
        results = store.similarity_search_with_score(query, k=topK)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
            }
            for doc, score in results
            if score > 0.7
        ]
    except Exception as e:
        print(f"[RAG] retrieval error: {e}")
        return []


def build_system_prompt(state: State) -> str:
    """
    Build system prompt injecting memories + RAG docs.

    Explicitly instructs the LLM to answer based on the retrieved docs first.
    """
    parts = ["你是一个专业的 AI 助手，擅长回答各类问题。"]

    # User preferences from long-term memory
    memories = state.get("relevant_memories", [])
    if memories:
        parts.append("\n\n## 关于用户的偏好")
        for mem in memories:
            content = mem if isinstance(mem, str) else mem.get("content", str(mem))
            parts.append(f"- {content}")

    # RAG docs with explicit directive
    docs = state.get("retrieved_docs", [])
    if docs:
        parts.append("\n\n## 参考知识库（请优先根据以下内容回答）")
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            parts.append(f"[文档{i}](来源:{source})\n{content}")

    return "".join(parts)


def rag_retrieve_node(state: State) -> dict:
    """
    LangGraph node: retrieve RAG docs and write to state.

    Reads user_input from state, writes retrieved_docs (score > 0.7 only).
    No async operations — wrapper is sync even though this may be called
    inside an async context by LangGraph.
    """
    user_input = state.get("user_input", "")

    if not user_input:
        return {"retrieved_docs": []}

    docs = retrieve_docs(user_input, topK=4)
    return {"retrieved_docs": docs}
