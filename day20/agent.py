"""
Day 20 — Orchestration MCP
CLI-агент, который оркестрирует два MCP-сервера:
  - server_search.py   (search_web, search_news)
  - server_filesystem.py (write_file, read_file, list_files, append_file)

Агент получает тему от пользователя, проводит исследование
и сохраняет итоговый отчёт в файл.

Использование:
    python agent.py --model gpt-4o
    python agent.py --model gpt-4o-mini --topic "Квантовые компьютеры 2024"
"""

import argparse
import asyncio

import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import AsyncExitStack

from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ── Настройки OpenAI-совместимого API ─────────────────────────────────────────

API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """\
Ты — агент-исследователь. Твоя задача: провести подробное исследование \

заданной темы и сохранить структурированный отчёт в файл.

Процесс работы:
1. Сделай несколько поисковых запросов через search_web и search_news, \
   чтобы охватить тему с разных сторон.
2. Проанализируй найденное.
3. Сохрани итоговый отчёт в файл через write_file. \
   Имя файла: report_<slug_темы>.md. \
   Формат отчёта: Markdown с разделами: \
   ## Обзор, ## Ключевые факты, ## Последние новости, ## Источники.
4. Сообщи пользователю, что отчёт готов и где он сохранён.

Используй инструменты последовательно — сначала собери достаточно данных, \
потом пиши отчёт. Не пиши отчёт после первого же поиска.
"""



# ── HTTP-запрос к OpenAI API ───────────────────────────────────────────────────

def call_llm(model: str, messages: list[dict], tools: list[dict]) -> dict:
    """Отправить запрос к OpenAI-совместимому API через urllib."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(

        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[HTTP Error {e.code}] {error_body}", file=sys.stderr)
        raise


# ── Сбор инструментов со всех MCP-сессий ──────────────────────────────────────

async def collect_tools(sessions: dict[str, ClientSession]) -> list[dict]:
    """
    Запросить список инструментов у каждого MCP-сервера
    и привести к формату OpenAI function-calling.
    """
    tools = []
    for server_name, session in sessions.items():

        result = await session.list_tools()
        for tool in result.tools:
            tools.append({
                "type": "function",

                "function": {
                    "name": tool.name,

                    "description": f"[{server_name}] {tool.description or ''}",
                    "parameters": tool.inputSchema,
                },
            })
    return tools



# ── Маршрутизация вызова инструмента ──────────────────────────────────────────

async def dispatch_tool(
    tool_name: str,
    tool_args: dict,
    sessions: dict[str, ClientSession],
    tool_registry: dict[str, str],
) -> str:
    """
    Найти нужный MCP-сервер по имени инструмента и вызвать его.
    tool_registry: {tool_name -> server_name}
    """
    server_name = tool_registry.get(tool_name)
    if not server_name:

        return f"Инструмент '{tool_name}' не найден ни на одном сервере."

    session = sessions[server_name]
    result = await session.call_tool(tool_name, tool_args)


    # Собираем текстовый результат из всех content-блоков
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else "(пустой результат)"


# ── Агентный цикл ─────────────────────────────────────────────────────────────


async def agent_loop(
    topic: str,

    model: str,
    sessions: dict[str, ClientSession],

) -> None:
    """Основной agentic loop: LLM → tool calls → LLM → ... → финальный ответ."""


    tools = await collect_tools(sessions)

    # Строим реестр: имя инструмента → имя сервера
    tool_registry: dict[str, str] = {}
    for server_name, session in sessions.items():
        result = await session.list_tools()

        for tool in result.tools:
            tool_registry[tool.name] = server_name

    print(f"\n📚 Доступные инструменты: {', '.join(tool_registry.keys())}")
    print(f"🔍 Тема исследования: {topic}\n")

    print("─" * 60)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Проведи исследование на тему: {topic}"},

    ]


    step = 0
    max_steps = 20  # защита от бесконечного цикла

    while step < max_steps:
        step += 1
        print(f"\n[Шаг {step}] Запрос к модели...")

        response = call_llm(model, messages, tools)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        # Добавляем ответ модели в историю
        messages.append(message)

        # Если модель закончила работу — выводим финальный текст
        if finish_reason == "stop":
            final_text = message.get("content", "")
            print(f"\n✅ Агент завершил работу:\n\n{final_text}")
            break


        # Обрабатываем tool calls
        if finish_reason == "tool_calls":
            tool_calls = message.get("tool_calls", [])

            print(f"  → Вызовов инструментов: {len(tool_calls)}")

            tool_results = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                server = tool_registry.get(fn_name, "?")

                print(f"  🔧 [{server}] {fn_name}({_format_args(fn_args)})")

                result_text = await dispatch_tool(fn_name, fn_args, sessions, tool_registry)

                # Краткий превью результата в консоли

                preview = result_text[:120].replace("\n", " ")

                print(f"     ↳ {preview}{'...' if len(result_text) > 120 else ''}")

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })

            messages.extend(tool_results)
        else:
            # Неожиданный finish_reason
            print(f"  ⚠️  Неожиданный finish_reason: {finish_reason}")
            break

    else:
        print(f"\n⚠️  Достигнут лимит шагов ({max_steps}).")



def _format_args(args: dict) -> str:
    """Краткое отображение аргументов для лога."""
    parts = []
    for k, v in args.items():

        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        parts.append(f"{k}={repr(v_str)}")
    return ", ".join(parts)


# ── Запуск MCP-серверов и старт агента ────────────────────────────────────────

async def main(topic: str, model: str) -> None:
    if not API_KEY:
        print("❌ Не задана переменная окружения OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Параметры запуска серверов через stdio
    search_params = StdioServerParameters(
        command=sys.executable,
        args=["server_search.py"],
    )
    filesystem_params = StdioServerParameters(
        command=sys.executable,
        args=["server_filesystem.py"],
    )

    async with AsyncExitStack() as stack:
        # Поднимаем оба сервера
        print("🚀 Запуск MCP-серверов...")

        search_transport = await stack.enter_async_context(
            stdio_client(search_params)
        )
        search_session = await stack.enter_async_context(
            ClientSession(*search_transport)
        )

        await search_session.initialize()

        print("  ✓ search-server готов")


        fs_transport = await stack.enter_async_context(

            stdio_client(filesystem_params)
        )

        fs_session = await stack.enter_async_context(
            ClientSession(*fs_transport)
        )
        await fs_session.initialize()
        print("  ✓ filesystem-server готов")


        sessions = {
            "search": search_session,
            "filesystem": fs_session,
        }

        await agent_loop(topic, model, sessions)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 20 — Research Agent с оркестрацией MCP-серверов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python agent.py --model gpt-4o-mini
  python agent.py --model gpt-4o --topic "Искусственный интеллект в медицине"

Переменные окружения:
  OPENAI_API_KEY   — токен доступа к API (обязательно)
  OPENAI_API_URL   — базовый URL API (по умолчанию: https://api.openai.com/v1/chat/completions)
  FS_WORKSPACE     — рабочая директория для сохранения файлов (по умолчанию: .)
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Название модели (например: gpt-4o, gpt-4o-mini)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Тема исследования. Если не задана — спросит интерактивно.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    topic = args.topic
    if not topic:
        topic = input("Введите тему для исследования: ").strip()
        if not topic:
            print("Тема не задана.", file=sys.stderr)
            sys.exit(1)

    asyncio.run(main(topic, args.model))
