#!/usr/bin/env python3
"""
MCP stdio-сервер для CRM.
Инструменты: get_user, get_ticket, list_user_tickets, search_user_by_email
"""

import json
import sys
import os
from pathlib import Path


def read_crm() -> dict:
    crm_path = Path(__file__).parent / "crm.json"
    with open(crm_path, encoding="utf-8") as f:
        return json.load(f)


# ── инструменты ────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> dict:
    crm = read_crm()
    for user in crm["users"]:
        if user["id"] == user_id:
            return user
    return {"error": f"Пользователь {user_id!r} не найден"}


def search_user_by_email(email: str) -> dict:
    crm = read_crm()
    email = email.strip().lower()
    for user in crm["users"]:
        if user["email"].lower() == email:
            return user
    return {"error": f"Пользователь с email {email!r} не найден"}


def get_ticket(ticket_id: str) -> dict:
    crm = read_crm()
    for ticket in crm["tickets"]:
        if ticket["id"] == ticket_id:
            return ticket
    return {"error": f"Тикет {ticket_id!r} не найден"}


def list_user_tickets(user_id: str) -> list:
    crm = read_crm()
    tickets = [t for t in crm["tickets"] if t["user_id"] == user_id]
    return tickets if tickets else []


# ── MCP protocol ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_user",
        "description": "Получить данные пользователя по его ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "ID пользователя (например u001)"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "search_user_by_email",
        "description": "Найти пользователя по email-адресу",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Email пользователя"}
            },
            "required": ["email"]
        }
    },
    {
        "name": "get_ticket",
        "description": "Получить тикет поддержки по его ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "ID тикета (например t001)"}
            },
            "required": ["ticket_id"]
        }
    },
    {
        "name": "list_user_tickets",
        "description": "Получить список всех тикетов пользователя",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "ID пользователя"}
            },
            "required": ["user_id"]
        }
    }
]

TOOL_MAP = {
    "get_user": get_user,
    "search_user_by_email": search_user_by_email,
    "get_ticket": get_ticket,
    "list_user_tickets": list_user_tickets,
}


def send(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def handle(req: dict) -> None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "crm-server", "version": "1.0.0"}
            }
        })

    elif method == "notifications/initialized":
        pass  # без ответа

    elif method == "tools/list":
        send({
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": TOOLS}
        })

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOL_MAP:
            send({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Инструмент {tool_name!r} не найден"}
            })
            return

        try:
            result = TOOL_MAP[tool_name](**arguments)
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                    "isError": False
                }
            })
        except Exception as e:
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Ошибка: {e}"}],
                    "isError": True
                }
            })

    else:
        if req_id is not None:
            send({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Метод {method!r} не поддерживается"}
            })


def main() -> None:
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}})
            continue
        handle(req)


if __name__ == "__main__":
    main()
