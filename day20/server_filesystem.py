"""
MCP Server: Filesystem
Инструменты для работы с локальными файлами.

Транспорт: stdio
"""

import os
from pathlib import Path


from mcp.server.fastmcp import FastMCP


mcp = FastMCP("filesystem-server")


# Корневая директория, в которой разрешены операции.
# По умолчанию — текущая рабочая директория.
WORKSPACE = Path(os.environ.get("FS_WORKSPACE", ".")).resolve()


def _safe_path(filename: str) -> Path:
    """Проверяет, что итоговый путь находится внутри WORKSPACE."""
    target = (WORKSPACE / filename).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise ValueError(f"Выход за пределы рабочей директории: {filename}")
    return target


@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """

    Записать текст в файл (создать или перезаписать).

    Args:
        filename: имя файла или относительный путь внутри workspace
        content: содержимое файла

    Returns:
        Сообщение об успехе с абсолютным путём
    """
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Файл сохранён: {path}"



@mcp.tool()
def read_file(filename: str) -> str:
    """
    Прочитать содержимое файла.


    Args:
        filename: имя файла или относительный путь внутри workspace


    Returns:
        Содержимое файла в виде строки
    """
    path = _safe_path(filename)
    if not path.exists():
        return f"Файл не найден: {path}"
    return path.read_text(encoding="utf-8")


@mcp.tool()
def list_files(subdir: str = ".") -> str:
    """
    Показать список файлов в рабочей директории (или поддиректории).

    Args:
        subdir: поддиректория для просмотра (по умолчанию корень workspace)

    Returns:
        Список файлов с размерами
    """
    target = _safe_path(subdir)
    if not target.is_dir():
        return f"Директория не найдена: {target}"


    lines = []
    for p in sorted(target.iterdir()):
        if p.is_file():

            size = p.stat().st_size
            lines.append(f"  {p.name}  ({size} байт)")
        elif p.is_dir():
            lines.append(f"  {p.name}/")

    return "\n".join(lines) if lines else "Директория пуста."



@mcp.tool()
def append_file(filename: str, content: str) -> str:
    """
    Дописать текст в конец существующего файла (или создать новый).

    Args:

        filename: имя файла
        content: текст для добавления


    Returns:
        Сообщение об успехе
    """
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:

        f.write(content)
    return f"Текст добавлен в файл: {path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
