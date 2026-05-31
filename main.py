import io
import os
import re
import json
import logging
import requests
from collections import deque
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, CouldNotRetrieveTranscript
from telegram import Update, BotCommand
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes 
from telegram import InputFile
from io import BytesIO
from gtts import gTTS
from groq import Groq

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

URL_PATTERN = re.compile(r"https?://[^\s]+")
YOUTUBE_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)
MAX_CONTENT_CHARS = 12000
MAX_TRANSCRIPT_CHARS = 14000
PREFS_FILE = "user_prefs.json"
FACTS_FILE = "facts.json"
MAX_MEMORY_MESSAGES = 60  # 30 exchanges
MAX_HISTORY_ITEMS = 5

VALID_MODES = {
    "assistant",
    "mentor",
    "programmer",
    "psychologist",
    "critic",
    "brainstorm"
}

MODE_SYSTEM_PROMPTS = {
    "mentor": (
    "Ты AI-наставник. "
    "Объясняй просто, по шагам, с примерами."
),

"critic": (
    "Ты жёсткий критик. "
    "Честно указывай на ошибки и слабые места."
),

"psychologist": (
    "Ты спокойный психолог. "
    "Поддерживай пользователя и помогай разобраться в эмоциях."
),

"programmer": (
    "Ты senior Python разработчик. "
    "Помогай писать чистый код и объясняй ошибки."
),
    "assistant": (
    "Ты ThinkMate AI — Telegram AI-ассистент, созданный Андреем. "
    "Твой создатель — Андрей, независимый AI-разработчик. "
    "Никогда не говори, что тебя создали Meta, OpenAI, Google или другие компании. "
    "Если пользователь спрашивает 'кто тебя создал?', отвечай только: "
    "'Меня создал Андрей.' "
    "Не придумывай других разработчиков, компаний или историй. "
    "Отвечай кратко, уверенно и дружелюбно."
    ),
    "tutor": (
        "You are a patient, friendly tutor. Explain concepts simply and clearly as if teaching a student. "
        "Use examples and analogies to make things easy to understand. "
        "Break down complex ideas into digestible steps. "
        "If the user says 'explain simpler' or 'as a beginner', simplify your previous explanation further."
    ),
    "coder": (
        "You are an expert programming assistant. "
        "Help with code, debugging, algorithms, architecture, and technical questions. "
        "Provide clean, well-commented code examples when relevant. Be precise and technical. "
        "Use conversation context to understand follow-ups like 'fix this', 'optimize', 'add tests'."
    ),
    "summarizer": (
        "You are a concise summarizer. Provide short, clear summaries. "
        "Be brief and to the point. Use 2-5 sentences maximum unless more detail is explicitly requested."
    ),
    "analyst": (
        "You are a sharp analyst. When given documents, articles, or data, extract key findings, "
        "identify patterns, compare ideas, and surface what matters most. "
        "Be structured and evidence-based. Use bullet points or sections when helpful."
    ),
    "brainstorm": (
        "You are a creative brainstorming partner. Generate diverse ideas, product concepts, "
        "business strategies, and creative directions. Think laterally and expansively. "
        "Offer multiple angles, challenge assumptions, and spark new thinking. "
        "Be energetic and generative."
    ),
}

# In-memory stores — reset on bot restart
user_memory: dict[str, deque] = {}
user_history: dict[str, list] = {}   # list of {"user": str, "bot": str}
user_facts = {}

# ─── Prefs ────────────────────────────────────────────────────────────────────

def load_prefs() -> dict:
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE) as f:
                raw = json.load(f)
            migrated = {}
            for uid, val in raw.items():
                if isinstance(val, str):
                    migrated[uid] = {"language": val, "mode": "assistant"}
                else:
                    migrated[uid] = val
            return migrated
        except Exception:
            pass
    return {}


def save_prefs(prefs: dict) -> None:
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
    except Exception as e:
        logger.error("Failed to save prefs: %s", e)


user_prefs: dict = load_prefs()


def ensure_user_prefs(user_id: str) -> None:
    if user_id not in user_prefs or not isinstance(user_prefs[user_id], dict):
        user_prefs[user_id] = {"language": "auto", "mode": "assistant"}


def get_language_instruction(user_id: int) -> str:
    ensure_user_prefs(str(user_id))
    lang = user_prefs[str(user_id)].get("language", "auto")
    if lang == "auto":
        return "Respond in the same language the user is writing in."
    return f"Always respond in {lang}, regardless of the language of the input."


def get_mode(user_id: int) -> str:
    ensure_user_prefs(str(user_id))
    return user_prefs[str(user_id)].get("mode", "assistant")

FACTS_FILE = "user_facts.json"


def load_facts():
    if os.path.exists(FACTS_FILE):
        try:
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_facts():
    with open(FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(user_facts, f, ensure_ascii=False, indent=2)
# ─── Memory ───────────────────────────────────────────────────────────────────

def get_memory(user_id: int) -> deque:
    uid = str(user_id)
    if uid not in user_memory:
        user_memory[uid] = deque(maxlen=MAX_MEMORY_MESSAGES)
    return user_memory[uid]


def add_to_memory(user_id: int, role: str, content: str) -> None:
    get_memory(user_id).append({"role": role, "content": content})


def add_to_history(user_id: int, user_msg: str, bot_reply: str) -> None:
    uid = str(user_id)
    if uid not in user_history:
        user_history[uid] = []
    user_display = (user_msg[:80] + "...") if len(user_msg) > 80 else user_msg
    bot_display = (bot_reply[:120] + "...") if len(bot_reply) > 120 else bot_reply
    user_history[uid].append({"user": user_display, "bot": bot_display})
    user_history[uid] = user_history[uid][-MAX_HISTORY_ITEMS:]


# ─── Content helpers ──────────────────────────────────────────────────────────

def extract_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_youtube_id(url: str) -> str | None:
    match = YOUTUBE_PATTERN.search(url)
    return match.group(1) if match else None


def fetch_webpage_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AssistantBot/3.5)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CONTENT_CHARS]


def fetch_youtube_transcript(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)
    # Prefer English; fall back to the first available transcript
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        transcript = next(iter(transcript_list))
    fetched = transcript.fetch()
    text = " ".join(snippet.text for snippet in fetched)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TRANSCRIPT_CHARS]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CONTENT_CHARS]


# ─── Groq ─────────────────────────────────────────────────────────────────────

def call_groq(messages: list) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


def build_messages(user_id: int, system_prompt: str, user_content: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(list(get_memory(user_id)))
    messages.append({"role": "user", "content": user_content})
    return messages


# ─── Shared processing logic ──────────────────────────────────────────────────

async def process_url_content(
    update: Update,
    context,
    url: str,
    user_text: str,
    user_id: int,
    override_mode: str | None = None,
) -> None:
    youtube_id = extract_youtube_id(url)
    lang_instruction = get_language_instruction(user_id)
    mode = override_mode or get_mode(user_id)
    mode_prompt = MODE_SYSTEM_PROMPTS[mode]

    if youtube_id:
        await update.message.reply_text("Fetching YouTube transcript...")
        try:
            transcript = fetch_youtube_transcript(youtube_id)
        except (NoTranscriptFound, TranscriptsDisabled):
            await update.message.reply_text(
                "No transcript or subtitles available for this video."
            )
            return
        except Exception as e:
            logger.error("YouTube transcript error for %s: %s", youtube_id, e)
            await update.message.reply_text(
                "Couldn't fetch the transcript. The video may be unavailable or restricted."
            )
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        await update.message.reply_text("Analyzing transcript...")
        system_prompt = (
            f"{mode_prompt} "
            "The user has shared a YouTube video. You have been given its transcript. "
            "Summarize the key points or answer the user's specific question about it. "
            f"{lang_instruction}"
        )
        user_content = (
            f"YouTube URL: {url}\n"
            f"User message: {user_text}\n\n"
            f"Transcript:\n{transcript}"
        )
    else:
        await update.message.reply_text("Fetching the page...")
        try:
            content = fetch_webpage_text(url)
        except requests.exceptions.Timeout:
            await update.message.reply_text("The page took too long to load. Please try a different URL.")
            return
        except requests.exceptions.RequestException as e:
            logger.error("Fetch error for %s: %s", url, e)
            await update.message.reply_text("Couldn't fetch that URL. It may be unavailable or block bots.")
            return

        if not content.strip():
            await update.message.reply_text("The page didn't contain readable text.")
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        await update.message.reply_text("Analyzing the page...")
        system_prompt = (
            f"{mode_prompt} "
            "The user has shared a URL. Analyze the webpage content and respond helpfully. "
            "If the user asked a specific question about it, prioritize answering that question. "
            f"{lang_instruction}"
        )
        user_content = (
            f"URL: {url}\n"
            f"User message: {user_text}\n\n"
            f"Page content:\n{content}"
        )

    messages = build_messages(user_id, system_prompt, user_content)
    try:
        reply = call_groq(messages)
        if override_mode is None:
            add_to_memory(user_id, "user", user_content)
            add_to_memory(user_id, "assistant", reply)
            add_to_history(user_id, user_text, reply)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error("Groq error: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


# ─── Commands ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! I'm your AI assistant.\n\n"
        "I can help with:\n"
        "- General chat: questions, coding, writing, brainstorming\n"
        "- URL analysis: send any link and I'll read and analyze it\n"
        "- YouTube: send a YouTube link and I'll summarize the video\n"
        "- PDF analysis: send a document and I'll analyze its content\n"
        "- Multiple languages: use /language to set your preference\n"
        "- AI modes: assistant, tutor, coder, summarizer, analyst, brainstorm\n"
        "- Conversation memory: I remember up to 30 exchanges\n\n"
        "Commands:\n"
        "/help — show all commands\n"
        "/about — информация о боте\n"
        "/mode — switch AI mode\n"
        "/language — set response language\n"
        "/summarize — one-shot summarize (text, URL, or YouTube)\n"
        "/history — view recent conversations\n"
        "/reset — clear conversation memory"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Commands:\n\n"
        "/start — launch the bot\n"
        "/help — show this list\n"
        "/reset — clear your conversation memory\n"
        "/history — show recent conversation\n"
        "/mode — сменить режим ИИ\n"
"assistant — универсальный помощник\n"
"mentor — AI наставник\n"
"programmer — senior разработчик\n"
"psychologist — психолог\n"
"critic — критик идей\n"
"brainstorm — генератор идей\n\n"
        "/language — set response language\n"
        "  e.g. /language russian | /language english | /language auto\n"
        "/summarize — one-shot summarize without changing your mode\n"
        "  /summarize <text>\n"
        "  /summarize <url>\n"
        "  /summarize <youtube url>"
    )
async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 ThinkMate AI\n\n"
        "Создан Андреем.\n\n"
        "Режимы:\n"
        "🎓 Наставник\n"
        "💻 Программист\n"
        "🧘 Психолог\n"
        "🔥 Критик\n"
        "💡 Генератор идей\n\n"
        "Версия: 1.0"
    )

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    try:
        if uid in user_memory:
            user_memory[uid].clear()
        await update.message.reply_text("Conversation memory cleared.")
    except Exception as e:
        logger.error("Reset failed for user %s: %s", uid, e)
        await update.message.reply_text("Something went wrong while clearing memory. Please try again.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    history = user_history.get(uid, [])

    if not history:
        await update.message.reply_text("No history yet.")
        return

    lines = ["History:\n"]
    for i, item in enumerate(history, 1):
        lines.append(f"{i}.\nYou: {item['user']}\nBot: {item['bot']}\n")
    await update.message.reply_text("\n".join(lines))


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    ensure_user_prefs(user_id)
    args = context.args

    if not args:
        current = user_prefs[user_id].get("language", "russian")
        await update.message.reply_text(
            f"Your current language setting is: {current}\n\n"
            "To change it: /language <language>\n"
            "Examples: /language russian | /language english | /language auto"
        )
        return

    chosen = " ".join(args).strip().lower()
    user_prefs[user_id]["language"] = chosen
    save_prefs(user_prefs)

    if chosen == "auto":
        await update.message.reply_text("Language set to auto — I'll match the language you're writing in.")
    else:
        await update.message.reply_text(f"Language set to {chosen}. I'll respond in {chosen} from now on.")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    ensure_user_prefs(user_id)
    args = context.args

    if not args:
        current = user_prefs[user_id].get("mode", "assistant")
        await update.message.reply_text(
            f"Current mode: {current}\n\n"
            "Available modes:\n"
            "    assistant — универсальный AI помощник\n"
        "    mentor — AI наставник, объясняет по шагам\n"
"    programmer — senior разработчик\n"
"    psychologist — психолог и поддержка\n"
"    critic — жёсткий критик идей\n"
"    brainstorm — генератор идей\n"
"Example: /mode mentor"
        )
        return

    chosen = args[0].strip().lower()
    if chosen not in VALID_MODES:
        await update.message.reply_text(
            f"Unknown mode: '{chosen}'\n\n"
            "Valid modes: assistant, mentor, programmer, psychologist, critic, brainstorm\n"
"Example: /mode mentor"
        )
        return

    user_prefs[user_id]["mode"] = chosen
    save_prefs(user_prefs)
    await update.message.reply_text(f"Mode changed to {chosen}.")


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    args = context.args

    if not args:
        await update.message.reply_text(
            "Usage: /summarize <text or URL>\n\n"
            "Examples:\n"
            "/summarize Artificial intelligence is...\n"
            "/summarize https://example.com\n"
            "/summarize https://youtube.com/watch?v=..."
        )
        return

    text = " ".join(args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    url = extract_url(text)
    if url:
        await process_url_content(update, context, url, text, user_id, override_mode="summarizer")
        return

    lang_instruction = get_language_instruction(user_id)
    system_prompt = (
        f"{MODE_SYSTEM_PROMPTS['summarizer']} "
        f"{lang_instruction}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    try:
        reply = call_groq(messages)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error("Groq error in /summarize: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


# ─── Message handlers ─────────────────────────────────────────────────────────

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    document = update.message.document

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    await update.message.reply_text("Reading the PDF...")

    try:
        file = await context.bot.get_file(document.file_id)
        pdf_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.error("Failed to download PDF: %s", e)
        await update.message.reply_text("Couldn't download the PDF. Please try again.")
        return

    try:
        content = extract_pdf_text(bytes(pdf_bytes))
    except Exception as e:
        logger.error("Failed to parse PDF: %s", e)
        await update.message.reply_text("Couldn't read the PDF. The file may be corrupted.")
        return

    if not content.strip():
        await update.message.reply_text(
            "This PDF doesn't contain any readable text. "
            "It may be a scanned document or image-only PDF."
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    await update.message.reply_text("Analyzing the PDF...")

    lang_instruction = get_language_instruction(user_id)
    mode = get_mode(user_id)
    system_prompt = (
        f"{MODE_SYSTEM_PROMPTS[mode]} "
        "The user has shared a PDF document. "
        "Analyze the content and provide a clear, helpful response based on your current mode. "
        f"{lang_instruction}"
    )
    filename = document.file_name or "document.pdf"
    user_content = f"PDF filename: {filename}\n\nContent:\n{content}"
    messages = build_messages(user_id, system_prompt, user_content)

    try:
        reply = call_groq(messages)
        add_to_memory(user_id, "user", user_content)
        add_to_memory(user_id, "assistant", reply)
        add_to_history(user_id, f"[PDF] {filename}", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error("Groq error: %s", e)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)

    if not prompt:
        await update.message.reply_text(
            "Напиши описание после команды.\n\nПример:\n/imagine futuristic cyberpunk city"
        )
        return

    await update.message.reply_text("🎨 Генерирую изображение...")

    try:
        style = "realistic detailed cinematic"

        if "anime" in prompt.lower():
            style = "anime style, vibrant, studio ghibli"

        elif "cyberpunk" in prompt.lower():
            style = "cyberpunk neon futuristic"

        elif "fantasy" in prompt.lower():
            style = "epic fantasy art"

        elif "realistic" in prompt.lower():
            style = "ultra realistic photo"

            full_prompt = f"{style}, {prompt}"

            image_url = f"https://image.pollinations.ai/prompt/{full_prompt}"

        response = requests.get(image_url)

        if response.status_code != 200:
            await update.message.reply_text("Ошибка генерации изображения.")
            return

        image_bytes = BytesIO(response.content)

        await update.message.reply_photo(
            photo=InputFile(image_bytes, filename="image.jpg"),
            caption=f"🖼 Запрос: {prompt}"
        )

    except Exception as e:
        logger.error("Image generation error: %s", e)

    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    user_id = str(update.effective_user.id)
    if "меня зовут" in user_text.lower():
        name = user_text.lower().replace("меня зовут", "").strip()

        if user_id not in user_facts:
            user_facts[user_id] = {}

        user_facts[user_id]["name"] = name
        save_facts()
        if user_text.lower() == "как меня зовут?":
    if user_id in user_facts and "name" in user_facts[user_id]:
        await update.message.reply_text(
            f'Тебя зовут {user_facts[user_id]["name"].title()}!'
        )
    else:
        await update.message.reply_text(
            "Я пока не знаю твоего имени."
        )
    return
    if not user_text or not user_text.strip():
        await update.message.reply_text("I'm here to help — send me a message, a URL, or a PDF.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    url = extract_url(user_text)

    if url:
        await process_url_content(update, context, url, user_text, user_id)
        return

    lang_instruction = get_language_instruction(user_id)
    mode = get_mode(user_id)
    system_prompt = f"{MODE_SYSTEM_PROMPTS[mode]} {lang_instruction}"
    messages = build_messages(user_id, system_prompt, user_text)
    try:
        reply = call_groq(messages)

        add_to_memory(user_id, "user", user_text)
        add_to_memory(user_id, "assistant", reply)
        add_to_history(user_id, user_text, reply)

        await update.message.reply_text(reply)

    except Exception as e:
            logger.error("Groq error: %s", e)
            await update.message.reply_text("Ошибка.")


# ─── Entry point ──────────────────────────────────────────────────────────────

async def post_init(application) -> None:
    await application.bot.set_my_commands([
    BotCommand("start", "Запустить бота"),
BotCommand("help", "Список команд"),
BotCommand("about", "О боте"),
BotCommand("reset", "Очистить память"),
BotCommand("history", "История диалога"),
BotCommand("mode", "Сменить режим"),
BotCommand("language", "Сменить язык"),
BotCommand("summarize", "Краткое резюме"),
BotCommand("imagine", "Создать изображение"),
    ])


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"].replace(" ", "")
    app = ApplicationBuilder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("imagine", imagine_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()