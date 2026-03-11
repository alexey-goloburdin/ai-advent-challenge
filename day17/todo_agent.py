#!/usr/bin/env python3
"""
Todo Agent — общается с OpenAI API через urllib,
управляет MCP-сервером todo_mcp_server.py через stdio.

Запуск:
    python todo_agent.py
    python todo_agent.py --model gpt-4o-mini
    python todo_agent.py --server ./todo_mcp_server.py
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path



# ─── MCP Client ───────────────────────────────────────────────────────────────

class MCPClient:
    """Общается с MCP-сервером через subprocess stdin/stdout (stdio transport)."""

    def __init__(self, server_path: str):
        self._proc = subprocess.Popen(
            [sys.executable, server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,          # логи сервера видны в терминале
            text=True,
            bufsize=1,
        )
        self._next_id = 1

        self._initialize()

    # ── низкоуровневый транспорт ──────────────────────────────────────────────

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)

        self._proc.stdin.flush()

    def _recv(self) -> dict:
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError("MCP server closed stdout unexpectedly")
        return json.loads(raw)

    def _request(self, method: str, params: dict | None = None) -> dict:
        req_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        return self._recv()

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ── MCP handshake ─────────────────────────────────────────────────────────


    def _initialize(self) -> None:
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "todo-agent", "version": "1.0.0"},
        })
        if "error" in resp:
            raise RuntimeError(f"MCP init failed: {resp['error']}")
        self._notify("notifications/initialized")

    # ── публичный API ─────────────────────────────────────────────────────────


    def list_tools(self) -> list[dict]:
        resp = self._request("tools/list")
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return f"MCP error: {resp['error']['message']}"
        contents = resp["result"].get("content", [])
        return "\n".join(c["text"] for c in contents if c.get("type") == "text")

    def close(self) -> None:
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ─── OpenAI Client (urllib only) ──────────────────────────────────────────────

class OpenAIClient:
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key

        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:

        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"


        data = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"OpenAI HTTP {e.code}: {body}") from e


# ─── Конвертация MCP tools → OpenAI function tools ────────────────────────────


def mcp_tools_to_openai(mcp_tools: list[dict]) -> list[dict]:
    return [
        {

            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in mcp_tools
    ]



# ─── Agent loop ───────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """Ты — персональный ассистент для управления задачами.
У тебя есть инструменты для работы со списком задач (todo list).

Отвечай кратко и по делу. Всегда подтверждай, что именно сделал."""


def run_agent(mcp: MCPClient, openai: OpenAIClient) -> None:
    mcp_tools = mcp.list_tools()
    openai_tools = mcp_tools_to_openai(mcp_tools)


    print(f"\n🤖 Todo Agent (модель: {openai.model})")
    print(f"   Доступно инструментов: {len(mcp_tools)}")
    print("   Введите задачу или вопрос. Для выхода: exit / quit / Ctrl+C\n")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        # ── ввод пользователя ─────────────────────────────────────────────────
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):

            print("\n👋 Пока!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "выход"}:
            print("👋 Пока!")
            break

        messages.append({"role": "user", "content": user_input})

        # ── agentic loop: LLM → tool calls → LLM → … ─────────────────────────
        while True:
            response = openai.chat(messages, tools=openai_tools)
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice["finish_reason"]

            messages.append(message)  # сохраняем ответ ассистента

            # ── нет вызовов инструментов → финальный ответ ───────────────────
            if finish_reason != "tool_calls" or not message.get("tool_calls"):
                print(f"\n🤖 Агент: {message.get('content', '')}\n")
                break


            # ── выполняем все tool_calls ──────────────────────────────────────
            for tc in message["tool_calls"]:

                fn = tc["function"]

                tool_name = fn["name"]
                tool_args = json.loads(fn["arguments"])

                print(f"   🔧 {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                result = mcp.call_tool(tool_name, tool_args)
                print(f"   📋 {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Todo Agent с MCP сервером")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI модель (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--server", default="todo_mcp_server.py",
        help="Путь к MCP серверу (default: todo_mcp_server.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Переменная OPENAI_API_KEY не задана.", file=sys.stderr)
        sys.exit(1)

    server_path = Path(args.server)

    if not server_path.exists():
        print(f"❌ MCP сервер не найден: {server_path}", file=sys.stderr)
        sys.exit(1)

    mcp = MCPClient(str(server_path))
    openai = OpenAIClient(api_key, args.model)
    print(f"   Base URL: {openai.base_url}")

    try:
        run_agent(mcp, openai)
    finally:
        mcp.close()


if __name__ == "__main__":
    main()
