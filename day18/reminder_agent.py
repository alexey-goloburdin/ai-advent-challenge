"""
Reminder Agent — 24/7 агент с естественным языком.

Коннектится к уже запущенному reminder_server.py по SSE.

Два параллельных потока:
  1. Чат с пользователем — естественный язык, LLM вызывает MCP tools.
  2. Фоновый polling — каждые POLL_INTERVAL секунд проверяет сработавшие напоминания.


Переменные окружения:
  OPENAI_BASE_URL  — базовый URL API (например https://api.openai.com/v1)
  OPENAI_API_KEY   — ключ доступа к API

Запуск:
  pip install "mcp[cli]"
  export OPENAI_BASE_URL=https://api.openai.com/v1
  export OPENAI_API_KEY=sk-...
  python reminder_agent.py
  python reminder_agent.py --server http://localhost:9000/sse --model gpt-4o
"""


import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,

    format="[agent]  %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Глушим HTTP-логи от httpx/httpcore (используются внутри sse_client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# ---------- константы ----------
POLL_INTERVAL = 10

LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_reminder",
            "description": (
                "Добавить напоминание. Вызывай когда пользователь хочет "
                "поставить напоминание на конкретное время или через промежуток времени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст напоминания — что именно нужно сделать.",
                    },
                    "delay_seconds": {
                        "type": "number",
                        "description": (
                            "Через сколько секунд должно сработать напоминание. "
                            "Переводи: 1 мин = 60, 1 час = 3600, 1 день = 86400."
                        ),
                    },
                },
                "required": ["text", "delay_seconds"],
            },
        },

    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "Показать все напоминания (активные и выполненные).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """Ты умный помощник-планировщик. Помогаешь пользователю управлять напоминаниями.

Текущее время: {now}

Когда пользователь говорит что-то вроде:
- «напомни мне через 10 минут позвонить маме» → вызови add_reminder
- «поставь напоминание на завтра в 9 утра» → вычисли delay_seconds до этого момента и вызови add_reminder
- «какие у меня напоминания» → вызови list_reminders
- любой другой вопрос → отвечай текстом, без вызова инструментов

Отвечай кратко и по-дружески."""


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="24/7 Reminder Agent — естественный язык + MCP SSE + LLM",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000/sse",
        help="URL SSE-эндпоинта MCP-сервера (default: http://127.0.0.1:8000/sse)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Название LLM-модели (default: gpt-4o-mini)",
    )
    return parser.parse_args()


# ---------- конфигурация из env ----------
def load_config() -> tuple[str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        log.error("Переменная окружения OPENAI_BASE_URL не задана.")
        sys.exit(1)
    if not api_key:
        log.error("Переменная окружения OPENAI_API_KEY не задана.")
        sys.exit(1)
    return base_url, api_key


# ---------- urllib helper ----------
def llm_request(payload: dict, base_url: str, api_key: str) -> dict:
    url = f"{base_url}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error("HTTP %d от LLM API: %s", e.code, body)
        raise
    except urllib.error.URLError as e:
        log.error("Ошибка соединения с LLM API: %s", e.reason)
        raise



# ---------- MCP helpers ----------
async def mcp_call(session: ClientSession, tool: str, arguments: dict) -> str:
    result = await session.call_tool(tool, arguments=arguments)
    return result.content[0].text


async def get_due(session: ClientSession) -> list[dict]:
    raw = await mcp_call(session, "get_due_reminders", {})
    return json.loads(raw)


# ---------- агентный цикл чата ----------
async def chat_turn(
    user_input: str,
    history: list[dict],
    session: ClientSession,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    history.append({"role": "user", "content": user_input})

    while True:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                },
                *history,
            ],
            "tools": LLM_TOOLS,
            "tool_choice": "auto",

            "max_tokens": 512,
        }

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, llm_request, payload, base_url, api_key
            )
        except Exception:
            return "⚠️ Не удалось связаться с LLM."

        message = response["choices"][0]["message"]

        history.append(message)

        if not message.get("tool_calls"):
            return message.get("content", "").strip()

        for tc in message["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            log.info("LLM вызывает: %s(%s)", fn_name, fn_args)
            try:
                tool_result = await mcp_call(session, fn_name, fn_args)
            except Exception as e:
                tool_result = f"Ошибка: {e}"

            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })


# ---------- фоновый polling ----------
async def polling_loop(
    session: ClientSession,
    base_url: str,
    api_key: str,
    model: str,
) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            due = await get_due(session)
        except Exception as exc:
            log.error("Ошибка при опросе MCP-сервера: %s", exc)
            continue


        for reminder in due:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты помощник-напоминалка. Пользователь получает уведомление. "
                            "Напомни дружелюбно и кратко (1-2 предложения)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Сработало напоминание: «{reminder['text']}»",
                    },
                ],
                "max_tokens": 150,
            }
            loop = asyncio.get_running_loop()
            try:
                response = await loop.run_in_executor(
                    None, llm_request, payload, base_url, api_key
                )
                comment = response["choices"][0]["message"]["content"].strip()
            except Exception:
                comment = reminder["text"]


            print(f"\n{'=' * 54}")
            print(f"🔔  НАПОМИНАНИЕ #{reminder['id']}: {reminder['text']}")

            print(f"    ⏰  {reminder['fire_at_human']}")
            print(f"    🤖  {comment}")
            print(f"{'=' * 54}\n", flush=True)


# ---------- чат-интерфейс ----------
async def chat_loop(
    session: ClientSession,
    base_url: str,
    api_key: str,
    model: str,
) -> None:
    history: list[dict] = []
    print("💬  Напишите напоминание на естественном языке. Для выхода — Ctrl+C.\n")

    loop = asyncio.get_running_loop()
    while True:

        try:

            user_input = await loop.run_in_executor(None, lambda: input("Вы: ").strip())
        except EOFError:

            break
        if not user_input:
            continue
        answer = await chat_turn(user_input, history, session, base_url, api_key, model)
        print(f"🤖  {answer}\n")


# ---------- точка входа ----------
async def main() -> None:
    args = parse_args()

    base_url, api_key = load_config()

    log.info("Подключаемся к MCP-серверу: %s", args.server)

    async with sse_client(args.server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            log.info("MCP tools: %s", [t.name for t in tools.tools])

            async with asyncio.TaskGroup() as tg:
                tg.create_task(chat_loop(session, base_url, api_key, args.model))
                tg.create_task(polling_loop(session, base_url, api_key, args.model))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:

        print("\n[agent] Остановлен пользователем.")
