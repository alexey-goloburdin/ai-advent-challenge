"""
MCP Reminder Server (stdio transport)
Хранит напоминания в reminders.json рядом с этим файлом.

Tools:
  add_reminder(text, delay_seconds) -> str
  get_due_reminders()               -> str   (JSON-список сработавших)
  list_reminders()                  -> str   (все напоминания)
"""

import asyncio
import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions

# ---------- logging (только stderr, иначе сломаем stdio-протокол) ----------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[server] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------- хранилище ----------
DB_PATH = Path(__file__).parent / "reminders.json"


def load_db() -> list[dict]:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return []


def save_db(reminders: list[dict]) -> None:
    DB_PATH.write_text(json.dumps(reminders, ensure_ascii=False, indent=2), encoding="utf-8")


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


# ---------- MCP-сервер ----------
server = Server("reminder-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add_reminder",
            description="Добавить напоминание, которое сработает через delay_seconds секунд.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст напоминания",
                    },
                    "delay_seconds": {
                        "type": "number",
                        "description": "Через сколько секунд сработать (например 3600 = 1 час)",
                    },
                },
                "required": ["text", "delay_seconds"],
            },
        ),
        types.Tool(
            name="get_due_reminders",
            description=(
                "Вернуть список напоминаний, у которых наступило время. "
                "Отработавшие напоминания помечаются как done."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_reminders",
            description="Вернуть все напоминания (включая выполненные).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "add_reminder":
        text = arguments["text"]
        delay = float(arguments["delay_seconds"])
        fire_at = now_ts() + delay

        reminders = load_db()
        reminder = {
            "id": len(reminders) + 1,
            "text": text,
            "fire_at": fire_at,
            "fire_at_human": datetime.fromtimestamp(fire_at).strftime("%Y-%m-%d %H:%M:%S"),
            "done": False,
            "created_at": now_ts(),
        }
        reminders.append(reminder)
        save_db(reminders)

        log.info("Добавлено напоминание #%d: «%s» (через %.0f сек)", reminder["id"], text, delay)
        result = f"✅ Напоминание #{reminder['id']} добавлено. Сработает в {reminder['fire_at_human']}."

    elif name == "get_due_reminders":
        reminders = load_db()
        ts = now_ts()
        due = [r for r in reminders if not r["done"] and r["fire_at"] <= ts]

        for r in due:
            r["done"] = True

        if due:
            save_db(reminders)

        result = json.dumps(due, ensure_ascii=False)

    elif name == "list_reminders":
        reminders = load_db()
        result = json.dumps(reminders, ensure_ascii=False, indent=2)

    else:
        result = f"Неизвестный инструмент: {name}"

    return [types.TextContent(type="text", text=result)]


# ---------- запуск ----------
async def main() -> None:
    log.info("Reminder MCP-сервер запущен (stdio)")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="reminder-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
