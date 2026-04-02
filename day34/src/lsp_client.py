"""
LSP client for pyright.
Communicates with pyright --lsp via JSON-RPC 2.0 over stdin/stdout.
"""

import json
import subprocess
import threading
import time
from pathlib import Path


class LspClient:
    def __init__(self, project_root: str):
        self.project_root = str(Path(project_root).resolve())
        self._proc: subprocess.Popen | None = None
        self._req_id = 0
        self._pending: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  transport                                                           #
    # ------------------------------------------------------------------ #

    def start(self):
        self._proc = subprocess.Popen(
            ["pyright", "--lsp"],
            stdin=subprocess.PIPE,

            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._handshake()

    def stop(self):
        if self._proc:
            try:
                self._send_notification("exit", {})
            except Exception:
                pass
            self._proc.terminate()
            self._proc = None
        self._initialized = False

    def _send(self, message: dict):
        body = json.dumps(message)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        data = (header + body).encode()
        self._proc.stdin.write(data)
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
            raise TimeoutError(f"LSP request '{method}' timed out after {timeout}s")

        with self._lock:
            entry = self._pending.pop(req_id)
        if entry["error"]:
            raise RuntimeError(f"LSP error: {entry['error']}")
        return entry["result"]

    def _send_notification(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _reader(self):
        while self._proc and self._proc.stdout:
            try:
                header = b""
                while b"\r\n\r\n" not in header:
                    chunk = self._proc.stdout.read(1)
                    if not chunk:
                        return

                    header += chunk
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


    # ------------------------------------------------------------------ #
    #  LSP lifecycle                                                       #
    # ------------------------------------------------------------------ #


    def _handshake(self):

        root_uri = Path(self.project_root).as_uri()
        self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didOpen": True},
                        "references": {"dynamicRegistration": False},
                        "definition": {"dynamicRegistration": False},
                    },
                    "workspace": {"workspaceFolders": True},
                },
                "workspaceFolders": [{"uri": root_uri, "name": "project"}],
                "initializationOptions": {
                    "python": {
                        "analysis": {
                            "diagnosticMode": "workspace",
                        }
                    }
                },
            },
        )
        self._send_notification("initialized", {})
        self._initialized = True
        # open all .py files so pyright indexes the full workspace
        py_files = list(Path(self.project_root).rglob("*.py"))
        n = len([f for f in py_files if not any(p in f.parts for p in (".venv", "venv", ".git", "__pycache__", "node_modules"))])
        import sys as _sys
        print(f"[LSP] indexing {n} files...", file=_sys.stderr, flush=True)
        self._open_all_python_files()
        # give pyright time to index — larger projects need more time
        wait = min(2.0 + n * 0.02, 10.0)
        time.sleep(wait)
        print(f"[LSP] ready", file=_sys.stderr, flush=True)

    def _open_all_python_files(self):
        root = Path(self.project_root)
        for py_file in root.rglob("*.py"):
            # skip venv / .git / __pycache__
            parts = py_file.parts
            if any(p in parts for p in (".venv", "venv", ".git", "__pycache__", "node_modules")):
                continue
            self._did_open(py_file)

    def _did_open(self, path: Path):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return
        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": "python",
                    "version": 1,
                    "text": text,
                }
            },
        )


    # ------------------------------------------------------------------ #
    #  public API                                                          #
    # ------------------------------------------------------------------ #

    def find_references(self, symbol: str) -> list[dict]:
        """

        Find all usages of a symbol across the project.
        Returns list of {file, line, column, context} dicts.
        """
        root = Path(self.project_root)
        results = []

        for py_file in root.rglob("*.py"):
            parts = py_file.parts
            if any(p in parts for p in (".venv", "venv", ".git", "__pycache__", "node_modules")):
                continue

            try:

                lines = py_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue


            for line_no, line_text in enumerate(lines):
                col = line_text.find(symbol)
                if col == -1:
                    continue

                try:
                    refs = self._request(
                        "textDocument/references",
                        {
                            "textDocument": {"uri": py_file.as_uri()},
                            "position": {"line": line_no, "character": col},
                            "context": {"includeDeclaration": True},
                        },
                        timeout=10.0,
                    )
                except Exception:
                    refs = None

                if not refs:
                    continue

                for ref in refs:
                    ref_path = ref["uri"].replace("file://", "")
                    ref_line = ref["range"]["start"]["line"]
                    try:
                        ref_lines = Path(ref_path).read_text(encoding="utf-8").splitlines()
                        context = ref_lines[ref_line].strip()
                    except Exception:
                        context = ""


                    entry = {
                        "file": ref_path,
                        "line": ref_line + 1,
                        "column": ref["range"]["start"]["character"] + 1,
                        "context": context,
                    }
                    if entry not in results:
                        results.append(entry)

                # found the symbol in this file, no need to check other occurrences
                # for the same file — pyright returns all refs project-wide
                break

        return results

    def get_definitions(self, symbol: str) -> list[dict]:
        """Find where symbol is defined."""
        root = Path(self.project_root)
        results = []

        for py_file in root.rglob("*.py"):
            parts = py_file.parts

            if any(p in parts for p in (".venv", "venv", ".git", "__pycache__", "node_modules")):

                continue

            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for line_no, line_text in enumerate(lines):
                col = line_text.find(symbol)
                if col == -1:
                    continue

                try:
                    defs = self._request(
                        "textDocument/definition",
                        {
                            "textDocument": {"uri": py_file.as_uri()},
                            "position": {"line": line_no, "character": col},
                        },
                        timeout=10.0,
                    )
                except Exception:
                    defs = None

                if not defs:

                    continue

                for d in (defs if isinstance(defs, list) else [defs]):

                    def_path = d["uri"].replace("file://", "")
                    def_line = d["range"]["start"]["line"]
                    try:
                        def_lines = Path(def_path).read_text(encoding="utf-8").splitlines()
                        context = def_lines[def_line].strip()
                    except Exception:
                        context = ""
                    entry = {
                        "file": def_path,
                        "line": def_line + 1,
                        "context": context,
                    }
                    if entry not in results:
                        results.append(entry)
                break

        return results
