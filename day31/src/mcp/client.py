import json
import subprocess
import sys


class GitMCPClient:
    """Запускает git_server.py как subprocess и общается через stdio."""

    def __init__(self, server_script: str):
        self._proc = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._msg_id = 0
        self._initialize()

    def _send(self, method: str, params: dict = None) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "id": self._msg_id, "method": method, "params": params or {}}
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        return json.loads(response_line)

    def _initialize(self):
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "dev-assistant", "version": "1.0"},
        })
        self._send("notifications/initialized")

    def call_tool(self, name: str, arguments: dict = None) -> str:
        resp = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        content = resp.get("result", {}).get("content", [])
        return content[0].get("text", "") if content else ""

    def git_branch(self) -> str:
        return self.call_tool("git_branch")

    def close(self):
        self._proc.terminate()
