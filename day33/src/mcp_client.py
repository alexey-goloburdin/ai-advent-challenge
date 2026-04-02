"""
MCP stdio-клиент. Запускает crm_server.py как subprocess,
общается через stdin/stdout по JSON-RPC 2.0.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path


class MCPClient:
    def __init__(self, server_script: Path):
        self._proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._req_id = 0
        self._lock = threading.Lock()
        self._initialize()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP-сервер закрыл соединение")
        return json.loads(line)

    def _request(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            req_id = self._next_id()
            msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params:
                msg["params"] = params
            self._send(msg)
            resp = self._recv()
            if "error" in resp:
                raise RuntimeError(f"MCP error: {resp['error']}")
            return resp.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        self._send(msg)

    def _initialize(self) -> None:
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "support-agent", "version": "1.0.0"}
        })
        self._notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        parts = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(parts)

    def close(self) -> None:
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
