"""
День 19. MCP-сервер с тремя инструментами пайплайна.

Инструменты:
  search      — поиск строки по файлам, возвращает список файлов с совпадениями
  readFile    — читает файл целиком (или его часть)
  summarize   — суммаризация текста через OpenAI API (urllib)
  saveToFile  — сохранение результата в Markdown-файл

Транспорт: SSE (HTTP)
"""

import fnmatch
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import TransportSecuritySettings

# ---------------------------------------------------------------------------
# Настройки сервера
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="pipeline-server",
    host="127.0.0.1",
    port=8000,
    log_level="WARNING",

    # Отключаем DNS rebinding protection — она блокирует подключение
    # клиента, когда заголовок Host не совпадает с ожидаемым паттерном
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)

# ---------------------------------------------------------------------------
# Вспомогательная функция: вызов OpenAI API через urllib
# ---------------------------------------------------------------------------
def _openai_chat(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан в переменных окружения")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")

    # Если OPENAI_BASE_URL уже содержит /v1 — не дублируем его

    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"

    else:
        url = f"{base_url}/v1/chat/completions"

    # Новые модели OpenAI (o-серия, gpt-5+) используют max_completion_tokens
    _new_models = ("o1", "o2", "o3", "o4", "gpt-5")
    _tokens_key = "max_completion_tokens" if any(model.startswith(p) for p in _new_models) else "max_tokens"
    payload = json.dumps(
        {"model": model, "messages": messages, _tokens_key: max_tokens}
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",

            "Authorization": f"Bearer {api_key}",
        },
        method="POST",

    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI API error {e.code}: {body}") from e

    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Инструмент 1: search
# ---------------------------------------------------------------------------
# Директории, которые search всегда пропускает
_SKIP_DIRS = {
    ".venv", "venv", ".env", "env",
    ".git", "__pycache__", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs",
}

def _is_binary(path: Path, sample: int = 8192) -> bool:
    """Эвристика: файл считается бинарным если в первых sample байтах есть нулевой байт."""
    try:

        chunk = path.read_bytes()[:sample]
        return b"\x00" in chunk
    except OSError:
        return True


@mcp.tool()
def search(
    query: str,
    directory: str = ".",
    pattern: str = "*",
    case_sensitive: bool = False,
    max_matches: int = 200,
) -> str:
    """
    Ищет строку `query` в текстовых файлах директории `directory`.
    Возвращает список файлов с совпадениями (без содержимого).
    Для чтения конкретных файлов используй инструмент readFile.

    Автоматически пропускает бинарные файлы и служебные директории
    (.venv, .git, __pycache__, node_modules и др.).


    Args:
        query:          Строка для поиска.
        directory:      Путь к директории (по умолчанию текущая).
        pattern:        Glob-маска файлов (по умолчанию все файлы).
        case_sensitive: Учитывать ли регистр (по умолчанию нет).
        max_matches:    Максимальное число файлов в результате (по умолчанию 200).

    Returns:
        JSON: {"files": [{"file": ..., "match_count": N, "match_lines": [...]}], "total_files": N, "truncated": bool}
    """
    root = Path(directory).resolve()
    if not root.exists():
        return json.dumps({"error": f"Директория не найдена: {directory}"})

    needle = query if case_sensitive else query.lower()
    # file -> список номеров строк с совпадением
    file_hits: dict[str, list[int]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Пропускаем служебные директории
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        # Пропускаем по glob-маске
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        # Пропускаем бинарные файлы
        if _is_binary(path):
            continue
        try:

            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        hit_lines = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                hit_lines.append(lineno)

        if hit_lines:

            rel = str(path.relative_to(root))
            file_hits[rel] = hit_lines

        if len(file_hits) >= max_matches:

            break

    files = [

        {"file": f, "match_count": len(lines), "match_lines": lines}
        for f, lines in file_hits.items()
    ]
    return json.dumps(
        {

            "query": query,
            "directory": str(root),
            "files": files,
            "total_files": len(files),
            "truncated": len(file_hits) >= max_matches,
        },
        ensure_ascii=False,
    )



# ---------------------------------------------------------------------------
# Инструмент 2: readFile
# ---------------------------------------------------------------------------

@mcp.tool()
def readFile(
    file: str,
    directory: str = ".",
    max_chars: int = 50_000,
) -> str:
    """
    Читает текстовый файл целиком и возвращает его содержимое.


    Args:
        file:       Путь к файлу относительно directory.
        directory:  Базовая директория (по умолчанию текущая).
        max_chars:  Максимальное число символов в ответе (по умолчанию 50000).
                    Если файл длиннее — возвращается начало с пометкой truncated.

    Returns:
        JSON: {"file": ..., "content": "...", "chars": N, "truncated": bool}
    """
    root = Path(directory).resolve()
    path = (root / file).resolve()

    # Защита от path traversal
    if not str(path).startswith(str(root)):
        return json.dumps({"error": "Доступ за пределы директории запрещён"})
    if not path.exists():
        return json.dumps({"error": f"Файл не найден: {file}"})
    if not path.is_file():
        return json.dumps({"error": f"Не является файлом: {file}"})
    if _is_binary(path):
        return json.dumps({"error": f"Файл бинарный, чтение невозможно: {file}"})


    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return json.dumps({"error": str(e)})

    truncated = len(content) > max_chars
    return json.dumps(
        {
            "file": file,
            "content": content[:max_chars],
            "chars": len(content),
            "truncated": truncated,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Инструмент 3: summarize
# ---------------------------------------------------------------------------

@mcp.tool()
def summarize(
    text: str,
    model: str = "gpt-4o-mini",
    language: str = "Russian",
) -> str:
    """
    Суммаризирует переданный текст с помощью OpenAI API.

    Args:
        text:     Текст для суммаризации.
        model:    Модель OpenAI (по умолчанию gpt-4o-mini).
        language: Язык итогового резюме (по умолчанию Russian).

    Returns:
        JSON-строка: {"summary": "...", "model": "..."}
    """
    if not text.strip():
        return json.dumps({"error": "Текст для суммаризации пуст"})

    messages = [
        {
            "role": "system",
            "content": (
                f"Ты помощник-аналитик. Напиши краткое резюме предоставленного текста на языке: {language}. "
                "Выдели ключевые темы и факты. Ответ оформи как связный текст без списков."
            ),
        },
        {"role": "user", "content": text},
    ]


    summary = _openai_chat(messages, model=model)
    return json.dumps({"summary": summary, "model": model}, ensure_ascii=False)



# ---------------------------------------------------------------------------
# Инструмент 4: saveToFile
# ---------------------------------------------------------------------------
@mcp.tool()
def saveToFile(

    content: str,
    filename: str,
    output_dir: str = ".",
    title: str = "",
) -> str:
    """
    Сохраняет текст в Markdown-файл.

    Args:
        content:    Текст для сохранения.
        filename:   Имя файла (без расширения или с .md).
        output_dir: Директория для сохранения (по умолчанию текущая).
        title:      Заголовок H1 в начале файла (если не пустой).

    Returns:
        JSON-строка: {"saved": true, "path": "...", "bytes": N}
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    out_path = out_dir / f"{stem}.md"


    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    lines.append(content)

    full_text = "\n".join(lines)
    out_path.write_text(full_text, encoding="utf-8")

    return json.dumps(
        {"saved": True, "path": str(out_path), "bytes": len(full_text.encode())},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Запуск MCP-сервера (SSE) на http://127.0.0.1:8000/sse ...")
    mcp.run(transport="sse")
