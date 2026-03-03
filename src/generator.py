"""
Generator (LLM backend)
=======================
Sends the retrieved context + user question to an LLM and returns its answer.

Supported providers (set LLM_PROVIDER in config.py):

  • "ollama"  — local LLM via Ollama (https://ollama.com).
                Install: brew install ollama && ollama pull llama3.2
                Free, runs offline, recommended for development.

  • "openai"  — OpenAI Chat Completions API.
                Needs OPENAI_API_KEY environment variable.

  • "groq"    — Groq cloud API (free tier, very fast).
                Needs GROQ_API_KEY environment variable.

  • "dummy"   — returns the raw context without calling any LLM.
                Useful for debugging the retrieval step in isolation.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import requests

from config import config

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — точный аналитический ассистент. Твоя задача — давать конкретные, \
подробные ответы на основе предоставленного контекста из документов.

Правила:
- Отвечай ТОЛЬКО на основе контекста. Не придумывай факты.
- Если в контексте есть конкретные данные (числа, термины, определения, задачи) — обязательно их приведи.
- Структурируй ответ: используй списки и абзацы для удобства чтения.
- Если информации недостаточно — прямо скажи, какого именно контекста не хватает.
- Отвечай на том языке, на котором задан вопрос."""

USER_PROMPT_TEMPLATE = """\
### Контекст из документов:
{context}

### Вопрос:
{question}

### Подробный ответ на основе контекста:"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Return the LLM's answer given question and retrieved context."""

    def _build_prompt(self, question: str, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaGenerator(BaseGenerator):
    """
    Calls the local Ollama server (default: http://localhost:11434).

    Start Ollama:
        ollama serve
    Pull a model:
        ollama pull llama3.2
    """

    def __init__(self):
        self._url = f"{config.OLLAMA_URL}/api/chat"
        self._model = config.OLLAMA_MODEL

    def generate(self, question: str, context: str) -> str:
        payload = {
            "model": self._model,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "num_ctx": 4096,       # explicit context window (default 2048 is too small)
                "num_predict": 512,
                "temperature": 0.1,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": self._build_prompt(question, context)},
            ],
        }
        try:
            resp = requests.post(
                self._url, json=payload, timeout=(10, 600), stream=True
            )
            if not resp.ok:
                body = resp.text[:500]
                return f"[Ошибка Ollama {resp.status_code}] {body}"
            parts: list[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                if "message" in chunk:
                    parts.append(chunk["message"]["content"])
                if chunk.get("done"):
                    break
            return "".join(parts).strip()
        except requests.ConnectionError:
            return (
                "[Ошибка] Ollama не запущена. "
                "Выполните: ollama serve  &&  ollama pull llama3.2"
            )
        except Exception as exc:
            return f"[Ошибка Ollama] {exc}"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIGenerator(BaseGenerator):
    """
    Uses the OpenAI Chat Completions API.
    Requires: pip install openai  +  export OPENAI_API_KEY=sk-…
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        self._client = OpenAI(api_key=api_key)
        self._model = config.OPENAI_MODEL

    def generate(self, question: str, context: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_prompt(question, context)},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Ошибка OpenAI] {exc}"


# ---------------------------------------------------------------------------
# Groq (free cloud, OpenAI-compatible)
# ---------------------------------------------------------------------------

class GroqGenerator(BaseGenerator):
    """
    Uses the Groq cloud API (free tier, very fast LPU inference).
    Needs GROQ_API_KEY environment variable.
    Sign up at https://console.groq.com
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = config.GROQ_MODEL

    def generate(self, question: str, context: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_prompt(question, context)},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Ошибка Groq] {exc}"


# ---------------------------------------------------------------------------
# Dummy (no LLM — useful for testing retrieval)
# ---------------------------------------------------------------------------

class DummyGenerator(BaseGenerator):
    """Returns the context as-is. Lets you validate retrieval without an LLM."""

    def generate(self, question: str, context: str) -> str:
        return (
            f"[DummyGenerator — LLM не используется]\n\n"
            f"Вопрос: {question}\n\n"
            f"Найденный контекст:\n{context}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_generator(provider: str | None = None) -> BaseGenerator:
    """Return the appropriate generator based on config or explicit provider."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "ollama":
        return OllamaGenerator()
    elif provider == "openai":
        return OpenAIGenerator()
    elif provider == "groq":
        return GroqGenerator()
    elif provider == "dummy":
        return DummyGenerator()
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. "
                         "Choose 'ollama', 'openai', or 'dummy'.")
