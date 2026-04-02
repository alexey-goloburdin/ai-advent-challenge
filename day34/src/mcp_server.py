"""
MCP server (stdio transport) exposing file-level and LSP tools.

Tools:
  - list_files(directory, extensions)
  - read_file(path)
  - write_file(path, content)
  - search_in_files(query, directory, extensions)
  - find_references(symbol, project_root)
  - check_rule(rule, project_root)   ← plain-language rule checking via LLM
"""


import json
import os
import sys
from pathlib import Path

# ensure project root (parent of src/) is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------ #

#  JSON-RPC helpers                                                    #
# ------------------------------------------------------------------ #

def _read_message() -> dict | None:
    header = b""
    while b"\r\n\r\n" not in header:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            return None
        header += ch
    length = int(
        next(
            p for p in header.decode().split("\r\n") if p.startswith("Content-Length:")
        ).split(":")[1].strip()
    )
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def _write_message(msg: dict):
    body = json.dumps(msg).encode()
    header = f"Content-Length: {len(body)}\r\n\r\n".encode()
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def _ok(req_id, result):
    _write_message({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id, code: int, message: str):
    _write_message({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


# ------------------------------------------------------------------ #
#  Tool implementations                                                #
# ------------------------------------------------------------------ #

SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".mypy_cache"}


def _iter_files(directory: str, extensions: list[str]) -> list[Path]:
    root = Path(directory).resolve()
    result = []
    for f in root.rglob("*"):
        if f.is_file() and not any(p in SKIP_DIRS for p in f.parts):
            if not extensions or f.suffix.lstrip(".") in extensions:

                result.append(f)
    return sorted(result)


def tool_list_files(args: dict) -> dict:
    directory = args.get("directory", ".")
    extensions = args.get("extensions", ["py"])
    files = _iter_files(directory, extensions)
    return {"files": [str(f) for f in files], "count": len(files)}


def tool_read_file(args: dict) -> dict:
    path = Path(args["path"])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return {"path": str(path), "content": text, "lines": len(lines)}


def tool_write_file(args: dict) -> dict:
    path = Path(args["path"])
    content = args["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "written_bytes": len(content.encode())}


def tool_search_in_files(args: dict) -> dict:

    query = args["query"].lower()
    directory = args.get("directory", ".")
    extensions = args.get("extensions", ["py"])
    case_sensitive = args.get("case_sensitive", False)

    matches = []
    for f in _iter_files(directory, extensions):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            haystack = line if case_sensitive else line.lower()
            if query in haystack:
                matches.append({"file": str(f), "line": i, "text": line.strip()})

    return {"query": args["query"], "matches": matches, "total": len(matches)}


# LSP client is created lazily per project_root
_lsp_clients: dict[str, object] = {}


def _get_lsp(project_root: str):
    if project_root not in _lsp_clients:
        from src.lsp_client import LspClient
        client = LspClient(project_root)
        client.start()
        _lsp_clients[project_root] = client
    return _lsp_clients[project_root]



def tool_find_references(args: dict) -> dict:
    symbol = args["symbol"]
    project_root = args.get("project_root", ".")
    lsp = _get_lsp(project_root)
    refs = lsp.find_references(symbol)
    defs = lsp.get_definitions(symbol)
    return {
        "symbol": symbol,
        "definitions": defs,
        "references": refs,
        "total_references": len(refs),
    }


def tool_check_rule(args: dict) -> dict:
    """
    Read files and return their contents for the agent to evaluate against the rule.
    The agent (LLM) performs the actual semantic check.
    """
    rule = args["rule"]
    project_root = args.get("project_root", ".")
    extensions = args.get("extensions", ["py"])
    max_files = args.get("max_files", 20)

    files = _iter_files(project_root, extensions)[:max_files]
    file_contents = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            file_contents.append({"file": str(f), "content": content})
        except Exception:
            pass

    return {
        "rule": rule,
        "files_to_check": file_contents,
        "file_count": len(file_contents),
    }


# ------------------------------------------------------------------ #
#  Tool registry                                                       #
# ------------------------------------------------------------------ #

TOOLS = {
    "list_files": {
        "fn": tool_list_files,
        "description": "List files in a directory filtered by extension.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path to list"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to include, e.g. ['py']",
                },
            },
            "required": [],
        },
    },
    "read_file": {
        "fn": tool_read_file,
        "description": "Read the full content of a file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "fn": tool_write_file,
        "description": "Write content to a file, creating parent directories as needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    "search_in_files": {
        "fn": tool_search_in_files,
        "description": "Text search across files in a directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for"},
                "directory": {"type": "string", "description": "Directory to search in"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to search",
                },
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search"},
            },
            "required": ["query"],
        },
    },
    "find_references": {
        "fn": tool_find_references,
        "description": (
            "Use pyright LSP to find all usages of a symbol (class, function, variable) "
            "across the project. Returns definitions and all reference locations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name to find references for"},
                "project_root": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["symbol", "project_root"],
        },
    },

    "check_rule": {
        "fn": tool_check_rule,
        "description": (

            "Load project files for semantic rule checking. "
            "Use this when the user asks to verify that code follows a plain-language rule. "
            "Returns file contents so the agent can evaluate compliance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string", "description": "Plain-language rule to check"},
                "project_root": {"type": "string", "description": "Root directory of the project"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to check",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to load (default 20)",
                },
            },
            "required": ["rule", "project_root"],
        },
    },
}


# ------------------------------------------------------------------ #
#  MCP request handlers                                                #
# ------------------------------------------------------------------ #


def handle_initialize(req_id, _params):
    _ok(req_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "file-assistant", "version": "1.0.0"},
    })



def handle_tools_list(req_id, _params):
    tools = [
        {
            "name": name,
            "description": meta["description"],
            "inputSchema": meta["inputSchema"],
        }
        for name, meta in TOOLS.items()
    ]
    _ok(req_id, {"tools": tools})



def handle_tools_call(req_id, params):
    name = params.get("name")
    args = params.get("arguments", {})


    if name not in TOOLS:
        _err(req_id, -32601, f"Unknown tool: {name}")
        return


    try:
        result = TOOLS[name]["fn"](args)
        _ok(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        })
    except Exception as e:
        _ok(req_id, {

            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        })


HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}



# ------------------------------------------------------------------ #
#  Main loop                                                           #
# ------------------------------------------------------------------ #

def run():
    while True:
        try:
            msg = _read_message()
        except Exception:
            break
        if msg is None:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")

        # notifications have no id
        if req_id is None:
            continue

        handler = HANDLERS.get(method)
        if handler:

            handler(req_id, msg.get("params", {}))
        else:
            _err(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    run()
