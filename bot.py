"""
Telegram bot for the RAG system.

Registration flow:
  1. User sends /start → asked to describe their purpose
  2. Admin receives notification with Approve / Reject buttons
  3. On approval the user is added to the whitelist and notified

Commands (all users):
  /start  — welcome / registration
  /help   — show available commands
  /about  — info about the bot and its creator
  /clear  — wipe your own document index

Commands (admin only):
  /adduser <id>     — manually add user to whitelist
  /removeuser <id>  — remove user from whitelist
  /users            — list allowed users
  /requests         — list pending registration requests
"""

import asyncio
import copy
import json
import logging
import tempfile
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

load_dotenv()

from config import config as default_config
from src.pipeline import RAGPipeline

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ASK_PURPOSE = 0  # ConversationHandler state

BOT_DESCRIPTION = (
    "Этот бот позволяет загружать документы (PDF, DOCX, TXT) "
    "и задавать вопросы по их содержимому. "
    "Внутри используется технология RAG (Retrieval-Augmented Generation): "
    "бот находит нужные фрагменты из ваших файлов и формирует точный ответ."
)

CREATOR_NAME = "Кирилл Слепцов"
CREATOR_TG = "@kira2299"

_WHITELIST_PATH = Path("data/allowed_users.json")


def _load_whitelist() -> set[int]:
    if not _WHITELIST_PATH.exists():
        return set()
    try:
        return set(json.loads(_WHITELIST_PATH.read_text()))
    except Exception:
        return set()


def _save_whitelist(users: set[int]) -> None:
    _WHITELIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WHITELIST_PATH.write_text(json.dumps(sorted(users)))


def _is_allowed(user_id: int) -> bool:
    return user_id in _load_whitelist() or user_id in default_config.TELEGRAM_ADMIN_IDS


def _is_admin(user_id: int) -> bool:
    return user_id in default_config.TELEGRAM_ADMIN_IDS


_REGISTRATIONS_PATH = Path("data/registrations.json")


def _load_registrations() -> dict:
    if not _REGISTRATIONS_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRATIONS_PATH.read_text())
    except Exception:
        return {}


def _save_registrations(registrations: dict) -> None:
    _REGISTRATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRATIONS_PATH.write_text(json.dumps(registrations, ensure_ascii=False, indent=2))


def _get_registration(user_id: int) -> dict | None:
    return _load_registrations().get(str(user_id))


def _upsert_registration(user_id: int, username: str, first_name: str,
                          purpose: str, status: str = "pending") -> None:
    regs = _load_registrations()
    regs[str(user_id)] = {
        "user_id": user_id,
        "username": username or "",
        "first_name": first_name or "",
        "purpose": purpose,
        "status": status,
    }
    _save_registrations(regs)


def _update_registration_status(user_id: int, status: str) -> None:
    regs = _load_registrations()
    key = str(user_id)
    if key in regs:
        regs[key]["status"] = status
        _save_registrations(regs)


_pipelines: dict[int, RAGPipeline] = {}


def _get_pipeline(user_id: int) -> RAGPipeline:
    if user_id not in _pipelines:
        cfg = copy.copy(default_config)
        cfg.COLLECTION_NAME = f"rag_user_{user_id}"
        _pipelines[user_id] = RAGPipeline(cfg=cfg)
    return _pipelines[user_id]


def require_access(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if _is_allowed(user_id):
            return await func(update, context)

        reg = _get_registration(user_id)
        if reg and reg["status"] == "pending":
            await update.message.reply_text(
                "Ваша заявка на рассмотрении. Ожидайте одобрения администратора."
            )
        elif reg and reg["status"] == "rejected":
            await update.message.reply_text(
                "Ваша заявка была отклонена. Обратитесь к администратору."
            )
        else:
            await update.message.reply_text(
                "Нет доступа. Отправьте /start для регистрации."
            )
    return wrapper


def require_admin(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not _is_admin(update.effective_user.id):
            await update.message.reply_text("Эта команда доступна только администратору.")
            return
        return await func(update, context)
    return wrapper


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_id = user.id

    if _is_admin(user_id):
        await update.message.reply_text(
            f"Добро пожаловать, администратор!\n\n"
            f"{BOT_DESCRIPTION}\n\n"
            f"Команды:\n"
            f"/help — справка\n"
            f"/about — о боте\n"
            f"/requests — заявки на доступ\n"
            f"/users — список пользователей",
        )
        return ConversationHandler.END

    if _is_allowed(user_id):
        await update.message.reply_text(
            "Вы уже зарегистрированы и можете пользоваться ботом.\n\n"
            "Отправьте файл (.pdf / .docx / .txt) и задавайте вопросы!\n"
            "/help — список команд"
        )
        return ConversationHandler.END

    reg = _get_registration(user_id)
    if reg:
        if reg["status"] == "pending":
            await update.message.reply_text(
                "⏳ Ваша заявка уже отправлена и ожидает одобрения администратора.\n"
                "Мы уведомим вас о решении."
            )
            return ConversationHandler.END
        elif reg["status"] == "rejected":
            await update.message.reply_text(
                "❌ Ваша предыдущая заявка была отклонена.\n"
                "Если считаете, что это ошибка — обратитесь к администратору: "
                f"{CREATOR_TG}"
            )
            return ConversationHandler.END

    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n\n"
        f"{BOT_DESCRIPTION}\n\n"
        f"Для получения доступа нужно пройти регистрацию.\n\n"
        f"<b>Опишите, для чего вы планируете использовать бот:</b>",
        parse_mode="HTML",
    )
    return ASK_PURPOSE


async def receive_purpose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    purpose = update.message.text.strip()

    if len(purpose) < 5:
        await update.message.reply_text(
            "Пожалуйста, опишите цель использования подробнее (минимум 5 символов)."
        )
        return ASK_PURPOSE

    _upsert_registration(
        user_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        purpose=purpose,
        status="pending",
    )

    username_str = f"@{user.username}" if user.username else "без username"
    text = (
        f"📬 <b>Новая заявка на доступ</b>\n\n"
        f"👤 {user.first_name} ({username_str})\n"
        f"🆔 <code>{user.id}</code>\n\n"
        f"📝 <b>Цель использования:</b>\n{purpose}"
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user.id}"),
            InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{user.id}"),
        ]
    ])

    for admin_id in default_config.TELEGRAM_ADMIN_IDS:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.warning("Could not notify admin %s: %s", admin_id, e)

    await update.message.reply_text(
        "✅ Заявка отправлена! Ожидайте решения администратора.\n"
        "Как только заявка будет рассмотрена — вы получите уведомление."
    )
    return ConversationHandler.END


async def handle_approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if not _is_admin(query.from_user.id):
        return

    action, user_id_str = query.data.split("_", 1)
    user_id = int(user_id_str)
    reg = _get_registration(user_id)
    name = reg["first_name"] if reg else str(user_id)

    if action == "approve":
        users = _load_whitelist()
        users.add(user_id)
        _save_whitelist(users)
        _update_registration_status(user_id, "approved")
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "✅ Ваша заявка одобрена! Добро пожаловать!\n\n"
                    "Отправьте файл (.pdf / .docx / .txt) — я его проиндексирую, "
                    "после чего вы сможете задавать вопросы по документу.\n\n"
                    "/help — список команд"
                ),
            )
        except Exception as e:
            logger.warning("Could not notify user %s: %s", user_id, e)
        await query.edit_message_text(f"✅ Пользователь {name} ({user_id}) одобрен.")

    elif action == "reject":
        _update_registration_status(user_id, "rejected")
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "❌ Ваша заявка на доступ отклонена.\n"
                    f"По вопросам обращайтесь к администратору: {CREATOR_TG}"
                ),
            )
        except Exception as e:
            logger.warning("Could not notify user %s: %s", user_id, e)
        await query.edit_message_text(f"❌ Пользователь {name} ({user_id}) отклонён.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lines = [
        "<b>Команды:</b>",
        "/start — регистрация / приветствие",
        "/help — эта справка",
        "/about — о боте и авторе",
        "/clear — очистить свои документы",
    ]
    if _is_admin(user_id):
        lines += [
            "",
            "<b>Администратор:</b>",
            "/adduser &lt;id&gt; — добавить пользователя вручную",
            "/removeuser &lt;id&gt; — удалить пользователя",
            "/users — список разрешённых пользователей",
            "/requests — заявки на доступ",
        ]
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"<b>О боте</b>\n\n"
        f"{BOT_DESCRIPTION}\n\n"
        f"<b>Поддерживаемые форматы:</b> PDF, DOCX, TXT\n\n"
        f"<b>Создатель:</b>\n"
        f"{CREATOR_NAME}\n"
        f"{CREATOR_TG}",
        parse_mode="HTML",
    )


@require_access
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pipeline = _get_pipeline(update.effective_user.id)
    pipeline.clear_index()
    await update.message.reply_text("Все твои документы удалены из индекса.")


@require_admin
async def cmd_adduser(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /adduser <telegram_id>")
        return
    try:
        new_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("ID должен быть числом.")
        return

    users = _load_whitelist()
    if new_id in users:
        await update.message.reply_text(f"Пользователь {new_id} уже в списке.")
        return
    users.add(new_id)
    _save_whitelist(users)
    _update_registration_status(new_id, "approved")
    await update.message.reply_text(f"Пользователь {new_id} добавлен.")


@require_admin
async def cmd_removeuser(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /removeuser <telegram_id>")
        return
    try:
        target_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("ID должен быть числом.")
        return

    users = _load_whitelist()
    if target_id not in users:
        await update.message.reply_text(f"Пользователь {target_id} не найден.")
        return
    users.discard(target_id)
    _save_whitelist(users)
    _pipelines.pop(target_id, None)
    await update.message.reply_text(f"Пользователь {target_id} удалён.")


@require_admin
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    users = _load_whitelist()
    if not users:
        await update.message.reply_text("Список разрешённых пользователей пуст.")
        return
    lines = ["<b>Разрешённые пользователи:</b>"]
    regs = _load_registrations()
    for uid in sorted(users):
        reg = regs.get(str(uid))
        name = reg["first_name"] if reg else "—"
        uname = f"@{reg['username']}" if reg and reg.get("username") else ""
        lines.append(f"• <code>{uid}</code> {name} {uname}".strip())
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


@require_admin
async def cmd_requests(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    regs = _load_registrations()
    pending = [r for r in regs.values() if r["status"] == "pending"]
    if not pending:
        await update.message.reply_text("Нет заявок, ожидающих рассмотрения.")
        return

    lines = [f"<b>Заявки на рассмотрении ({len(pending)}):</b>"]
    for r in pending:
        uname = f"@{r['username']}" if r.get("username") else "без username"
        lines.append(
            f"\n👤 {r['first_name']} ({uname})\n"
            f"🆔 <code>{r['user_id']}</code>\n"
            f"📝 {r['purpose']}"
        )
    keyboard = None
    if len(pending) == 1:
        uid = pending[0]["user_id"]
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{uid}"),
            InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{uid}"),
        ]])
    await update.message.reply_text(
        "\n".join(lines), parse_mode="HTML", reply_markup=keyboard
    )


@require_access
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    file_name = doc.file_name or "document"
    ext = Path(file_name).suffix.lower()

    if ext not in {".txt", ".pdf", ".docx"}:
        await update.message.reply_text(
            "Поддерживаются только файлы: .txt, .pdf, .docx"
        )
        return

    await update.message.reply_text("Загружаю и индексирую файл...")
    await update.message.chat.send_action("typing")

    tg_file = await context.bot.get_file(doc.file_id)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await tg_file.download_to_drive(tmp_path)
        pipeline = _get_pipeline(update.effective_user.id)
        n = await asyncio.to_thread(pipeline.ingest_file, tmp_path)
        total = pipeline.document_count
        await update.message.reply_text(
            f"Готово! Проиндексировано <b>{n}</b> чанков из <i>{file_name}</i>.\n"
            f"Всего в индексе: {total} чанков.\n\n"
            f"Задавай вопросы!",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"Ошибка: {e}")
    except Exception as e:
        logger.exception("Error ingesting file %s", file_name)
        await update.message.reply_text(f"Не удалось обработать файл: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


_CASUAL_PHRASES = {
    "спасибо", "спс", "благодарю", "ок", "окей", "хорошо", "понял",
    "понятно", "да", "нет", "ладно", "отлично", "супер", "класс", "норм",
    "thanks", "thank you", "ok", "okay", "cool", "nice", "great",
    "привет", "хай", "hi", "hello", "пока", "bye", "👍", "👌",
}


@require_access
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text.strip()

    if question.lower().rstrip("!?. ") in _CASUAL_PHRASES or len(question) <= 3:
        await update.message.reply_text(
            "Задайте вопрос по вашим документам — я постараюсь найти ответ!"
        )
        return

    pipeline = _get_pipeline(update.effective_user.id)

    if pipeline.document_count == 0:
        await update.message.reply_text(
            "Индекс пуст. Сначала отправь файл (.pdf / .docx / .txt)."
        )
        return

    await update.message.chat.send_action("typing")
    question = update.message.text
    try:
        answer = await asyncio.to_thread(pipeline.query, question)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.exception("Error answering question")
        await update.message.reply_text(f"Ошибка при генерации ответа: {e}")


def _warmup_ollama() -> None:
    if default_config.LLM_PROVIDER != "ollama":
        return
    try:
        import requests as _req
        _req.post(
            f"{default_config.OLLAMA_URL}/api/chat",
            json={
                "model": default_config.OLLAMA_MODEL,
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 1},
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=(10, 120),
        )
        logger.info("Ollama model warmed up: %s", default_config.OLLAMA_MODEL)
    except Exception as e:
        logger.warning("Ollama warmup failed (non-fatal): %s", e)


def main() -> None:
    token = default_config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не задан. "
            "Добавьте его в .env или переменные окружения."
        )

    _warmup_ollama()

    app = Application.builder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            ASK_PURPOSE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_purpose)
            ],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
    )

    app.add_handler(conv_handler)
    app.add_handler(CallbackQueryHandler(handle_approval_callback, pattern=r"^(approve|reject)_\d+$"))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("adduser", cmd_adduser))
    app.add_handler(CommandHandler("removeuser", cmd_removeuser))
    app.add_handler(CommandHandler("users", cmd_users))
    app.add_handler(CommandHandler("requests", cmd_requests))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
