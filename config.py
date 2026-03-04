import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    DATA_DIR: Path = field(default_factory=lambda: Path("data/documents"))
    CHROMA_DIR: Path = field(default_factory=lambda: Path("data/chroma_db"))
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- Text chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # --- Retrieval ---
    TOP_K: int = 3

    # --- LLM backend ---
    LLM_PROVIDER: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "ollama"))
    OLLAMA_MODEL: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2:1b"))
    OLLAMA_URL: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    OPENAI_MODEL: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
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
