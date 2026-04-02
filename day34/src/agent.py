"""
Agent that talks to the MCP server and uses tools to work with project files.
Uses OpenAI-compatible API (proxyapi.ru).
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path


# ------------------------------------------------------------------ #
#  MCP client (talks to mcp_server.py via stdio)                      #
# ------------------------------------------------------------------ #

class McpClient:

    def __init__(self, server_script: str):
        self._proc = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._req_id = 0
        self._pending: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._initialize()

    def _send(self, msg: dict):
        body = json.dumps(msg).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        self._proc.stdin.write(header + body)

        self._proc.stdin.flush()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        req_id = self._next_id()
        event = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": event, "result": None, "error": None}
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        if not event.wait(timeout):
            raise TimeoutError(f"MCP request '{method}' timed out")
        with self._lock:
            entry = self._pending.pop(req_id)
        if entry["error"]:
            raise RuntimeError(f"MCP error: {entry['error']}")
        return entry["result"]

    def _read_loop(self):
        while self._proc.stdout:
            try:
                header = b""
                while b"\r\n\r\n" not in header:
                    ch = self._proc.stdout.read(1)
                    if not ch:
                        return
                    header += ch
                length = int(
                    next(
                        p for p in header.decode().split("\r\n") if p.startswith("Content-Length:")
                    ).split(":")[1].strip()
                )
                body = self._proc.stdout.read(length)
                msg = json.loads(body)
                if "id" in msg:
                    with self._lock:
                        entry = self._pending.get(msg["id"])
                    if entry:
                        entry["result"] = msg.get("result")
                        entry["error"] = msg.get("error")
                        entry["event"].set()
            except Exception:
                return

    def _initialize(self):
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "file-agent", "version": "1.0"},
        })

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list", {})
        return result.get("tools", [])


    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments}, timeout=120.0)
        content = result.get("content", [])
        return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    def stop(self):
        self._proc.terminate()


# ------------------------------------------------------------------ #
#  OpenAI-compatible HTTP client (urllib only)                        #
# ------------------------------------------------------------------ #

def _llm_request(messages: list[dict], tools: list[dict], model: str) -> dict:
    import urllib.request

    url = os.environ["OPENAI_API_URL"].rstrip("/") + "/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 16000,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ------------------------------------------------------------------ #
#  Tool output printer                                                 #
# ------------------------------------------------------------------ #

def _print_tool_result(tool_name: str, result: str):
    """Print tool results to stdout. For find_references print full structured output."""
    if tool_name == "find_references":
        try:
            data = json.loads(result)
            print(f"\n[результат] Символ: {data['symbol']}")
            defs = data.get("definitions", [])
            if defs:
                print("Определения:")
                for d in defs:
                    print(f"  {d['file']}:{d['line']}  {d['context']}")
            refs = data.get("references", [])
            print(f"\nВсе места использования ({data['total_references']} шт.):")
            for r in refs:
                print(f"  {r['file']}:{r['line']}:{r['column']}  {r['context']}")
            print()
            return
        except Exception:
            pass
    display = result if len(result) < 500 else result[:500] + "...[truncated]"
    print(f"[result] {display}\n")


# ------------------------------------------------------------------ #
#  Single agentic turn                                                 #
# ------------------------------------------------------------------ #

def _run_turn(messages: list[dict], openai_tools: list[dict], mcp: McpClient, model: str):
    """
    Run one user turn: call LLM in a loop until it stops using tools.
    Mutates messages in place (appends assistant + tool results).
    """
    max_iterations = 20
    for i in range(max_iterations):
        print(f"[agent] thinking... (iteration {i + 1})")

        response = _llm_request(messages, openai_tools, model)
        msg = response["choices"][0]["message"]
        finish_reason = response["choices"][0]["finish_reason"]

        messages.append(msg)


        if msg.get("content"):
            print(f"\n{msg['content']}\n")

        if finish_reason == "stop" or not msg.get("tool_calls"):
            break

        for tc in msg["tool_calls"]:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])

            print(f"[tool] {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
            result = mcp.call_tool(tool_name, tool_args)
            _print_tool_result(tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],

                "content": result,
            })


# ------------------------------------------------------------------ #
#  Chat loop                                                           #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an AI assistant that works with Python project files.
You have access to tools for reading, searching, and modifying files, as well as
LSP-powered symbol analysis via pyright.

When the user asks to find usages of a symbol — use find_references with the project root.
When the user asks to check that code follows a rule — use check_rule to load files,
then carefully analyze each file's content against the rule and produce a detailed report
with specific violations (file, line, what exactly is wrong).

Always be proactive: don't ask the user to specify files — explore the project yourself.
Output results directly in your response — do NOT write to a file unless the user asks.
Respond in the same language the user uses."""


def run_chat(model: str, project_root: str):
    server_script = str(Path(__file__).parent / "mcp_server.py")
    mcp = McpClient(server_script)

    try:
        mcp_tools = mcp.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in mcp_tools
        ]

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nProject root: {project_root}",
            }
        ]

        print(f"\n{'='*60}")
        print(f"File Assistant | project: {project_root}")
        print(f"Model: {model}")
        print(f"Type 'exit' or Ctrl+C to quit.")
        print(f"{'='*60}\n")

        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Bye.")
                break

            messages.append({"role": "user", "content": user_input})
            _run_turn(messages, openai_tools, mcp, model)

    finally:
        mcp.stop()


# backward-compat: single-task mode
def run_agent(task: str, model: str, project_root: str):
    server_script = str(Path(__file__).parent / "mcp_server.py")
    mcp = McpClient(server_script)
    try:
        mcp_tools = mcp.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in mcp_tools
        ]
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nProject root: {project_root}",
            },
            {"role": "user", "content": task},
        ]

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"Project: {project_root}")
        print(f"{'='*60}\n")
        _run_turn(messages, openai_tools, mcp, model)
    finally:
        mcp.stop()
