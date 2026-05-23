"""
Memory module using Redis for storage with OpenAI embeddings for vector search.
"""

import json
import os
import uuid
from typing import Annotated, Any, Optional

import numpy as np
import redis
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from ..config import REDIS_HOST, REDIS_PORT, REDIS_DB, OPENAI_API_KEY
from ..state import State

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Cosine similarity threshold for retrieval
SIMILARITY_THRESHOLD = 0.75

# Redis key prefixes
MEMORY_HASH_PREFIX = "memory:session:"
VECTOR_KEY_PREFIX = "memory:vector:"

# Module-level singletons
_redis_client: redis.Redis | None = None
_openai_client: OpenAI | None = None


def _get_redis_client() -> redis.Redis:
    """Get Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
    return _redis_client


def _get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using OpenAI embeddings."""
    if os.getenv("SKIP_EMBEDDING", "").lower() in ("true", "1", "yes"):
        # Return dummy zero vector when embedding is disabled
        return [0.0] * EMBEDDING_DIMENSIONS
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def _store_vector(session_id: str, memory_id: str, vector: list[float], content: str) -> None:
    """Store embedding vector in Redis."""
    client = _get_redis_client()
    vector_key = f"{VECTOR_KEY_PREFIX}{session_id}"
    
    # Store vector with metadata
    vector_data = {
        "id": memory_id,
        "vector": json.dumps(vector),
        "content": content
    }
    client.hset(vector_key, memory_id, json.dumps(vector_data))


def save_memory(session_id: str, content: str, ai_response: str | None = None) -> str:
    """
    Save a memory for a session using Redis hash + OpenAI embeddings.

    Stores both user input and optional AI response to form a complete dialogue
    memory (问过+答过). createdAt is stored as Unix timestamp (int) for accurate
    temporal filtering.
    """
    import time

    client = _get_redis_client()
    memory_id = str(uuid.uuid4())
    created_at = int(time.time())

    # Get embedding for the content
    vector = _get_embedding(content)

    # Store in Redis hash (session metadata)
    hash_key = f"{MEMORY_HASH_PREFIX}{session_id}"
    memory_data = {
        "id": memory_id,
        "content": content,
        "createdAt": created_at,
    }
    if ai_response:
        memory_data["ai_response"] = ai_response
    client.hset(hash_key, memory_id, json.dumps(memory_data))

    # Store vector separately for similarity search
    _store_vector(session_id, memory_id, vector, content)

    return memory_id


def retrieve_memories(session_id: str, query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve memories for a session using cosine similarity.
    
    Args:
        session_id: Unique identifier for the session
        query: The query to search for similar memories
        top_k: Number of top results to return (default: 3)
        
    Returns:
        List of memory dicts with similarity score above threshold
    """
    client = _get_redis_client()
    vector_key = f"{VECTOR_KEY_PREFIX}{session_id}"
    
    # Get query embedding
    query_vector = _get_embedding(query)
    query_vector = np.array(query_vector).reshape(1, -1)
    
    # Get all stored vectors for this session
    all_vectors = client.hgetall(vector_key)
    
    if not all_vectors:
        return []
    
    memories_with_scores = []
    
    for memory_id, vector_data_str in all_vectors.items():
        vector_data = json.loads(vector_data_str)
        stored_vector = np.array(json.loads(vector_data["vector"])).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector, stored_vector)[0][0]
        
        if similarity >= SIMILARITY_THRESHOLD:
            memories_with_scores.append({
                "id": memory_id,
                "content": vector_data["content"],
                "similarity": float(similarity)
            })
    
    # Sort by similarity and return top_k
    memories_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    return memories_with_scores[:top_k]


def memory_retrieve_node(state: State) -> dict:
    """
    LangGraph node: retrieve relevant memories at conversation start.

    Reads session_id + user_input from state, writes relevant_memories.
    """
    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")

    if not session_id or not user_input:
        return {"relevant_memories": []}

    memories = retrieve_memories(session_id, user_input, top_k=3)
    return {"relevant_memories": memories}


def memory_save_node(state: State) -> dict:
    """
    LangGraph node: save session context to long-term memory after conversation.

    Saves both the user question and the last AI response to form a complete
    dialogue memory (问过+答过). Input must be non-trivial (>20 chars).
    """
    session_id = state.get("session_id", "")
    user_input = state.get("user_input", "")

    if not session_id or len(user_input) < 20:
        return {}

    # Extract the last AI response from messages
    ai_response: str | None = None
    messages: list = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            ai_response = getattr(msg, "content", None) or ""
            if isinstance(ai_response, list):
                # Handle content blocks (e.g., text blocks)
                ai_response = " ".join(
                    b.text if hasattr(b, "text") else str(b)
                    for b in ai_response
                )
            break

    memory_id = save_memory(
        session_id,
        f"用户问过：{user_input[:200]}",
        ai_response=ai_response,
    )
    return {}  # memory_id is stored in Redis; no state field needed
