# RAG System — Telegram Bot

Telegram-бот для анализа документов на основе технологии **RAG (Retrieval-Augmented Generation)**.

Пользователь загружает файлы (PDF, DOCX, TXT), бот их индексирует и отвечает на вопросы на основе содержимого этих файлов.

## Возможности

- Загрузка и индексация документов (PDF, DOCX, TXT)
- Ответы на вопросы по содержимому загруженных файлов
- Изолированная база для каждого пользователя
- Система регистрации с одобрением администратором
- Поддержка нескольких LLM-провайдеров: Groq, OpenAI, Ollama

## Структура проекта

```
rag_system/
├── bot.py                  # Telegram-бот
├── main.py                 # CLI-интерфейс
├── config.py               # Конфигурация
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example            # Шаблон переменных окружения
├── src/
│   ├── pipeline.py         # RAG-пайплайн
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── generator.py        # LLM-генераторы (Groq / OpenAI / Ollama)
│   ├── retriever.py
│   └── vector_store.py     # ChromaDB
└── data/
    ├── documents/          # Директория для документов (CLI)
    ├── chroma_db/          # Векторная база (создаётся автоматически)
    ├── allowed_users.json  # Вайтлист пользователей
    └── registrations.json  # Заявки на доступ
```

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/<your-username>/rag_system.git
cd rag_system
```

### 2. Создать `.env`

```bash
cp .env.example .env
```

Заполнить `.env`:

```env
TELEGRAM_BOT_TOKEN=токен_от_botfather
TELEGRAM_ADMIN_IDS=ваш_telegram_id

LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant
```

### 3. Запустить

**Локально:**
```bash
pip install -r requirements.txt
python3 bot.py
```

**Через Docker:**
```bash
docker compose up -d
```

## Переменные окружения

| Переменная | Описание | Пример |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather | `123456:ABC...` |
| `TELEGRAM_ADMIN_IDS` | ID администраторов (через запятую) | `452095992` |
| `LLM_PROVIDER` | Провайдер LLM | `groq` / `openai` / `ollama` |
| `GROQ_API_KEY` | Ключ Groq API (бесплатно) | `gsk_...` |
| `GROQ_MODEL` | Модель Groq | `llama-3.1-8b-instant` |
| `OPENAI_API_KEY` | Ключ OpenAI (опционально) | `sk-...` |
| `OLLAMA_URL` | URL Ollama (только локально) | `http://localhost:11434` |

## LLM-провайдеры

| Провайдер | Стоимость | Скорость | Как получить |
|---|---|---|---|
| **Groq** | Бесплатно | Очень быстро | [console.groq.com](https://console.groq.com) |
| **OpenAI** | Платно | Быстро | [platform.openai.com](https://platform.openai.com) |
| **Ollama** | Бесплатно | Медленно (CPU) | [ollama.com](https://ollama.com) |

## Команды бота

| Команда | Описание |
|---|---|
| `/start` | Регистрация / приветствие |
| `/help` | Список команд |
| `/about` | О боте и создателе |
| `/clear` | Очистить свои документы |
| `/requests` | Заявки на доступ *(только админ)* |
| `/users` | Список пользователей *(только админ)* |
| `/adduser <id>` | Добавить пользователя *(только админ)* |
| `/removeuser <id>` | Удалить пользователя *(только админ)* |

## Деплой на сервер

Рекомендуется **Oracle Cloud Always Free** (бесплатно навсегда, 1GB RAM):

```bash
# На сервере
curl -fsSL https://get.docker.com | sh
git clone https://github.com/<your-username>/rag_system.git
cd rag_system
nano .env          # вставить ключи
docker compose up -d

# Логи
docker compose logs -f

# Перезапуск после изменений
git pull
docker compose up -d --build
```

## Автор

**Кирилл Слепцов**
Telegram: [@kira2299](https://t.me/kira2299)
