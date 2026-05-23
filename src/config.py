"""Application configuration module."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# ── LLM (Anthropic / MiniMax) ───────────────────────────────────────────────
ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
ANTHROPIC_AUTH_TOKEN: str = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.7")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Vector stores ────────────────────────────────────────────────────────────
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "personal_assistant_kb")

# ── Memory ───────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

# ── Tools ─────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
AMAP_KEY: str = os.getenv("AMAP_KEY", "")

# ── Checkpoint storage (production) ───────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# ── App settings ─────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
