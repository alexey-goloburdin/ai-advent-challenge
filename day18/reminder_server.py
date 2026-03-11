"""
MCP Reminder Server — SSE транспорт.

Запускается один раз отдельно, агент коннектится по URL.

Установка:
  pip install "mcp[cli]" uvicorn

Запуск:
  python reminder_server.py
  python reminder_server.py --host 0.0.0.0 --port 9000


После запуска сервер доступен по адресу:
  http://localhost:8000/sse
"""


import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

# ---------- logging ----------

logging.basicConfig(
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
    DB_PATH.write_text(
        json.dumps(reminders, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


# ---------- FastMCP ----------
mcp = FastMCP("reminder-server")


@mcp.tool()
def add_reminder(text: str, delay_seconds: float) -> str:
    """

    Добавить напоминание.

    Args:
        text: Текст напоминания.
        delay_seconds: Через сколько секунд сработать (60 = 1 мин, 3600 = 1 час).
    """
    fire_at = now_ts() + delay_seconds
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

    log.info("Добавлено #%d: «%s» (через %.0f сек)", reminder["id"], text, delay_seconds)
    return f"✅ Напоминание #{reminder['id']} добавлено. Сработает в {reminder['fire_at_human']}."


@mcp.tool()
def get_due_reminders() -> str:
    """Вернуть список сработавших напоминаний и пометить их как выполненные."""

    reminders = load_db()
    ts = now_ts()
    due = [r for r in reminders if not r["done"] and r["fire_at"] <= ts]
    for r in due:
        r["done"] = True
    if due:
        save_db(reminders)
    return json.dumps(due, ensure_ascii=False)


@mcp.tool()
def list_reminders() -> str:
    """Вернуть все напоминания (активные и выполненные)."""
    return json.dumps(load_db(), ensure_ascii=False, indent=2)


# ---------- запуск ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP Reminder Server (SSE)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = Starlette(routes=[Mount("/", app=mcp.sse_app())])

    log.info("Сервер запущен на http://%s:%d/sse", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
