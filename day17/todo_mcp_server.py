#!/usr/bin/env python3
"""
MCP Todo List Server (stdio transport)
Хранит задачи в JSON-файле, общается через stdin/stdout по протоколу JSON-RPC 2.0
"""

import json
import sys
import os

import uuid
from datetime import datetime
from pathlib import Path

# ─── Хранилище ────────────────────────────────────────────────────────────────

TODOS_FILE = Path("todos.json")



def load_todos() -> list[dict]:
    if TODOS_FILE.exists():
        return json.loads(TODOS_FILE.read_text(encoding="utf-8"))
    return []



def save_todos(todos: list[dict]) -> None:
    TODOS_FILE.write_text(
        json.dumps(todos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── Инструменты (tools) ──────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "todo_add",

        "description": "Добавить новую задачу в список",
        "inputSchema": {

            "type": "object",
            "properties": {
                "title": {

                    "type": "string",
                    "description": "Название задачи",
                },

                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Приоритет задачи (по умолчанию medium)",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "todo_list",
        "description": "Получить список задач (все, только активные или только выполненные)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["all", "active", "done"],

                    "description": "Фильтр: all — все, active — незавершённые, done — выполненные",
                },
            },
        },
    },
    {

        "name": "todo_done",
        "description": "Отметить задачу как выполненную по её ID",
        "inputSchema": {
            "type": "object",
            "properties": {

                "id": {
                    "type": "string",
                    "description": "ID задачи",
                },
            },
            "required": ["id"],
        },
    },
    {
        "name": "todo_delete",
        "description": "Удалить задачу из списка по её ID",

        "inputSchema": {

            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "ID задачи",
                },
            },
            "required": ["id"],
        },

    },
    {
        "name": "todo_clear_done",
        "description": "Удалить все выполненные задачи",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]



def tool_todo_add(args: dict) -> str:
    title = args.get("title", "").strip()
    if not title:
        return "Ошибка: название задачи не может быть пустым."

    priority = args.get("priority", "medium")

    todos = load_todos()
    task = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "priority": priority,
        "done": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    todos.append(task)
    save_todos(todos)
    return f"✅ Задача добавлена: [{task['id']}] {title} (приоритет: {priority})"



def tool_todo_list(args: dict) -> str:
    filter_mode = args.get("filter", "all")
    todos = load_todos()

    if filter_mode == "active":
        todos = [t for t in todos if not t["done"]]
    elif filter_mode == "done":

        todos = [t for t in todos if t["done"]]

    if not todos:
        return "Список задач пуст."

    priority_order = {"high": 0, "medium": 1, "low": 2}
    todos = sorted(todos, key=lambda t: (t["done"], priority_order.get(t["priority"], 1)))

    priority_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = []

    for t in todos:
        status = "☑" if t["done"] else "☐"
        icon = priority_icons.get(t["priority"], "⚪")

        lines.append(f"{status} {icon} [{t['id']}] {t['title']}")

    header = {"all": "Все задачи", "active": "Активные задачи", "done": "Выполненные задачи"}[filter_mode]
    return f"{header} ({len(todos)}):\n" + "\n".join(lines)


def tool_todo_done(args: dict) -> str:
    task_id = args.get("id", "").strip()
    todos = load_todos()


    for task in todos:
        if task["id"] == task_id:
            if task["done"]:

                return f"Задача [{task_id}] уже отмечена как выполненная."
            task["done"] = True
            task["done_at"] = datetime.now().isoformat(timespec="seconds")
            save_todos(todos)
            return f"✅ Задача [{task_id}] «{task['title']}» отмечена как выполненная."

    return f"Ошибка: задача с ID «{task_id}» не найдена."



def tool_todo_delete(args: dict) -> str:
    task_id = args.get("id", "").strip()
    todos = load_todos()


    for i, task in enumerate(todos):
        if task["id"] == task_id:
            removed = todos.pop(i)

            save_todos(todos)
            return f"🗑 Задача [{task_id}] «{removed['title']}» удалена."


    return f"Ошибка: задача с ID «{task_id}» не найдена."



def tool_todo_clear_done(args: dict) -> str:
    todos = load_todos()
    active = [t for t in todos if not t["done"]]
    removed = len(todos) - len(active)

    if removed == 0:
        return "Нет выполненных задач для удаления."

    save_todos(active)
    return f"🗑 Удалено выполненных задач: {removed}."



TOOL_HANDLERS = {

    "todo_add": tool_todo_add,
    "todo_list": tool_todo_list,
    "todo_done": tool_todo_done,

    "todo_delete": tool_todo_delete,
    "todo_clear_done": tool_todo_clear_done,
}



# ─── JSON-RPC / MCP протокол ──────────────────────────────────────────────────


def send(obj: dict) -> None:
    """Отправить JSON-объект в stdout и сразу сбросить буфер."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()



def handle_request(req: dict) -> dict | None:

    method = req.get("method", "")
    req_id = req.get("id")  # None для notifications


    # ── initialize ──────────────────────────────────────────────────────────
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {

                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "todo-mcp-server",
                    "version": "1.0.0",

                },
            },

        }


    # ── initialized (notification, без ответа) ───────────────────────────────
    if method == "notifications/initialized":

        return None


    # ── tools/list ───────────────────────────────────────────────────────────
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    # ── tools/call ───────────────────────────────────────────────────────────
    if method == "tools/call":

        params = req.get("params", {})

        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {

                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Инструмент «{tool_name}» не найден",
                },
            }


        try:
            result_text = handler(tool_args)
            return {
                "jsonrpc": "2.0",

                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False,
                },

            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",

                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Ошибка: {exc}"}],
                    "isError": True,
                },

            }

    # ── неизвестный метод ────────────────────────────────────────────────────

    if req_id is not None:

        return {
            "jsonrpc": "2.0",

            "id": req_id,

            "error": {
                "code": -32601,
                "message": f"Метод «{method}» не поддерживается",

            },
        }

    return None  # notification — молчим



# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:

    # Переключаем stderr в режим без буферизации для отладочных логов
    sys.stderr.reconfigure(line_buffering=True)

    print(f"[todo-mcp] сервер запущен, PID={os.getpid()}", file=sys.stderr)


    for raw_line in sys.stdin:
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send({
                "jsonrpc": "2.0",
                "id": None,

                "error": {"code": -32700, "message": f"Parse error: {e}"},
            })
            continue


        response = handle_request(req)

        if response is not None:
            send(response)


if __name__ == "__main__":
    main()
