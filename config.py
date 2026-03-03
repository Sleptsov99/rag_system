import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Paths ---
    DATA_DIR: Path = field(default_factory=lambda: Path("data/documents"))
    CHROMA_DIR: Path = field(default_factory=lambda: Path("data/chroma_db"))

    # --- Embedding model ---
    # Multilingual model: handles Russian + English out of the box
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- Text chunking ---
    CHUNK_SIZE: int = 1000      # characters per chunk
    CHUNK_OVERLAP: int = 150    # overlap between adjacent chunks

    # --- Retrieval ---
    TOP_K: int = 3              # how many chunks to retrieve per query

    # --- LLM backend ---
    # Options: "ollama" (local, default) | "openai" (needs OPENAI_API_KEY)
    LLM_PROVIDER: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "ollama"))

    # Ollama settings — override via env variables when running in Docker
    OLLAMA_MODEL: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2:1b"))
    OLLAMA_URL: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))

    # OpenAI settings (set OPENAI_API_KEY env variable)
    OPENAI_MODEL: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    # Groq settings (set GROQ_API_KEY env variable)
    GROQ_MODEL: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))

    # --- ChromaDB ---
    COLLECTION_NAME: str = "rag_documents"

    # --- Telegram bot ---
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    TELEGRAM_ADMIN_IDS: list = field(
        default_factory=lambda: [
            int(x) for x in os.environ.get("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()
        ]
    )


# Singleton used across the project
config = Config()
