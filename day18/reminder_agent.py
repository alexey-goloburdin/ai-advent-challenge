"""
Reminder Agent — 24/7 агент, который:
  1. Подключается к reminder_server.py через stdio
  2. Добавляет тестовые напоминания при старте
  3. Каждые POLL_INTERVAL секунд опрашивает get_due_reminders
  4. При срабатывании — отправляет напоминание в LLM и печатает ответ


Переменные окружения:
  OPENAI_BASE_URL  — базовый URL API (например https://api.openai.com/v1)
  OPENAI_API_KEY   — ключ доступа к API

Запуск:
  pip install mcp
  export OPENAI_BASE_URL=https://api.openai.com/v1
  export OPENAI_API_KEY=sk-...
  python reminder_agent.py --model gpt-4o-mini
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------- logging ----------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[agent]  %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------- константы ----------
SERVER_SCRIPT = Path(__file__).parent / "reminder_server.py"
POLL_INTERVAL = 10   # секунд между проверками
PYTHON = sys.executable


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="24/7 Reminder Agent с MCP-сервером и LLM-уведомлениями",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Название LLM-модели (default: gpt-4o-mini)",
    )
    return parser.parse_args()


# ---------- конфигурация из env ----------
def load_config() -> tuple[str, str]:
    """Читает OPENAI_BASE_URL и OPENAI_API_KEY из переменных окружения."""
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not base_url:
        log.error("Переменная окружения OPENAI_BASE_URL не задана.")
        sys.exit(1)
    if not api_key:
        log.error("Переменная окружения OPENAI_API_KEY не задана.")
        sys.exit(1)


    return base_url, api_key


# ---------- LLM через urllib ----------
def llm_notify(reminder_text: str, base_url: str, api_key: str, model: str) -> str:
    """
    Отправляет текст напоминания в LLM и возвращает ответ.
    Использует только стандартную библиотеку urllib — без сторонних HTTP-клиентов.
    """
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник-напоминалка. "
                    "Когда пользователь получает напоминание, "
                    "кратко и дружелюбно напомни ему об этом (1-2 предложения)."
                ),
            },
            {
                "role": "user",
                "content": f"Сработало напоминание: «{reminder_text}»",
            },
        ],
        "max_tokens": 150,
    }

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
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        log.error("HTTP %d от LLM API: %s", e.code, error_body)
        return f"[LLM недоступен: HTTP {e.code}]"
    except urllib.error.URLError as e:
        log.error("Ошибка соединения с LLM API: %s", e.reason)
        return f"[LLM недоступен: {e.reason}]"


# ---------- MCP helpers ----------
async def add_reminder(session: ClientSession, text: str, delay_seconds: float) -> str:
    result = await session.call_tool(
        "add_reminder",
        arguments={"text": text, "delay_seconds": delay_seconds},
    )
    return result.content[0].text



async def get_due(session: ClientSession) -> list[dict]:
    result = await session.call_tool("get_due_reminders", arguments={})
    return json.loads(result.content[0].text)


# ---------- вывод в консоль ----------
def print_notification(reminder: dict, llm_response: str) -> None:
    print()
    print("=" * 54)
    print(f"🔔  НАПОМИНАНИЕ #{reminder['id']}: {reminder['text']}")
    print(f"    ⏰  {reminder['fire_at_human']}")
    print(f"    🤖  {llm_response}")
    print("=" * 54)
    print()


# ---------- основной цикл ----------
async def agent_loop(
    session: ClientSession,
    base_url: str,
    api_key: str,
    model: str,
) -> None:
    log.info("Агент запущен. Модель: %s. Интервал проверки: %d сек.", model, POLL_INTERVAL)

    # --- демо: добавляем тестовые напоминания при старте ---
    for text, delay in [
        ("Выпить стакан воды 💧", 15),
        ("Сделать разминку 🏃",   30),
        ("Проверить почту 📧",    60),
    ]:
        msg = await add_reminder(session, text, delay_seconds=delay)
        log.info(msg)
    # --------------------------------------------------------

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        try:
            due = await get_due(session)
        except Exception as exc:
            log.error("Ошибка при опросе MCP-сервера: %s", exc)
            continue

        if not due:
            log.info("Нет новых напоминаний.")
            continue

        for reminder in due:

            log.info("Сработало напоминание #%d, отправляю в LLM...", reminder["id"])
            # LLM-вызов блокирующий — запускаем в executor, чтобы не блокировать event loop
            loop = asyncio.get_running_loop()
            llm_response = await loop.run_in_executor(
                None,
                llm_notify,
                reminder["text"],
                base_url,
                api_key,
                model,
            )
            print_notification(reminder, llm_response)


# ---------- точка входа ----------
async def main() -> None:
    args = parse_args()
    base_url, api_key = load_config()


    log.info("Подключаемся к MCP-серверу: %s", SERVER_SCRIPT.name)

    server_params = StdioServerParameters(

        command=PYTHON,
        args=[str(SERVER_SCRIPT)],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            log.info("Доступные tools: %s", [t.name for t in tools.tools])

            await agent_loop(session, base_url, api_key, args.model)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Агент остановлен пользователем.")
