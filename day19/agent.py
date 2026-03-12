"""
День 19. Агент с автоматическим пайплайном MCP-инструментов.

Пайплайн:
  1. search     — находит совпадения в локальных файлах
  2. summarize  — суммаризирует найденный контекст через LLM
  3. saveToFile — сохраняет итоговое резюме в Markdown-файл

Агент сам решает, в каком порядке вызывать инструменты (tool-calling loop).
Транспорт к MCP-серверу: SSE (HTTP).
"""

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------------------------------------------------------------------------
# OpenAI API через urllib
# ---------------------------------------------------------------------------

def openai_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """Выполняет запрос к OpenAI Chat Completions API. Возвращает объект choice."""
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
    body: dict[str, Any] = {

        "model": model,
        "messages": messages,
        _tokens_key: 16000,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    payload = json.dumps(body).encode()
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
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_err = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI API error {e.code}: {body_err}") from e


    return data["choices"][0]


# ---------------------------------------------------------------------------
# Преобразование MCP-инструментов в формат OpenAI tools
# ---------------------------------------------------------------------------


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    result = []
    for tool in mcp_tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return result



# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

async def run_agent(
    task: str,
    model: str,
    sse_url: str,
    search_dir: str,
    output_dir: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"Задача: {task}")
    print(f"Модель: {model}")
    print(f"MCP-сервер: {sse_url}")
    print(f"{'='*60}\n")

    async with sse_client(url=sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Инициализация MCP-сессии
            await session.initialize()
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools

            print(f"Доступные инструменты: {[t.name for t in mcp_tools]}\n")


            openai_tools = mcp_tools_to_openai(mcp_tools)

            # Начальные сообщения
            messages: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "Ты — агент-исследователь. У тебя есть четыре инструмента:\n"
                        "  search(query, directory, ...) — находит файлы содержащие строку, возвращает СПИСОК ФАЙЛОВ\n"
                        "  readFile(file, directory) — читает файл ЦЕЛИКОМ\n"
                        "  summarize(text, language) — суммаризирует текст через LLM\n"
                        "  saveToFile(content, filename, output_dir, title) — сохраняет в Markdown\n\n"
                        "Алгоритм работы:\n"
                        "1. ПОИСК. Разбей тему на ключевые слова и вызывай search многократно с разными query. "
                        "search возвращает только список файлов — не их содержимое.\n"
                        "2. ОТБОР. Получив список файлов — проанализируй пути и имена. "
                        "Читай через readFile ТОЛЬКО файлы которые явно релевантны теме по названию или пути. "
                        "ПРОПУСКАЙ без чтения: migrations/, tests/, *_test.py, test_*.py, *.pyc, "
                        "автогенерированный код, конфиги, локализацию — если тема не касается их напрямую. "
                        "Если релевантных файлов больше 10 — читай только самые важные.\n"
                        "3. СУММАРИЗАЦИЯ. Передай содержимое прочитанных файлов в summarize одним вызовом. "
                        "Перед содержимым каждого файла указывай его путь в формате: ### path/to/file.py\n"
                        "4. СОХРАНЕНИЕ. Передай результат summarize в saveToFile.\n\n"
                        f"Параметры по умолчанию: search.directory='{search_dir}', readFile.directory='{search_dir}', "

                        f"saveToFile.output_dir='{output_dir}'.\n"
                        "После сохранения сообщи какие файлы прочитал и какие query использовал."
                    ),
                },

                {"role": "user", "content": task},
            ]

            step = 0
            while True:
                step += 1
                print(f"--- Шаг {step}: запрос к LLM ---")

                choice = openai_chat(messages, tools=openai_tools, model=model)
                message = choice["message"]
                finish_reason = choice["finish_reason"]

                # Добавляем ответ ассистента в историю
                messages.append(message)

                # Если модель хочет вызвать инструменты
                if finish_reason == "tool_calls" and message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        fn_name = tc["function"]["name"]
                        fn_args_raw = tc["function"]["arguments"]
                        fn_args = json.loads(fn_args_raw)

                        print(f"  → Вызов инструмента: {fn_name}")

                        print(f"    Аргументы: {json.dumps(fn_args, ensure_ascii=False, indent=6)}")


                        # Вызов через MCP-сессию
                        result = await session.call_tool(fn_name, fn_args)

                        # Извлекаем текст из результата
                        tool_output = ""
                        for content_block in result.content:
                            if hasattr(content_block, "text"):
                                tool_output += content_block.text

                        print(f"    Результат: {tool_output[:300]}{'...' if len(tool_output) > 300 else ''}\n")

                        # Добавляем результат в историю
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_output,
                            }
                        )

                # Если модель завершила работу
                elif finish_reason == "stop":
                    final_text = message.get("content", "")

                    print(f"\n{'='*60}")
                    print("Итоговый ответ агента:")
                    print(final_text)
                    print(f"{'='*60}\n")
                    break

                # Модель упёрлась в лимит токенов — продолжаем с тем что есть
                elif finish_reason == "length":
                    partial = message.get("content", "")
                    print(f"⚠️  Достигнут лимит токенов (finish_reason=length). "
                          f"Ответ обрезан на {len(partial)} символах, продолжаем...")
                    # Добавляем обрезанный ответ и просим продолжить
                    messages.append({
                        "role": "user",
                        "content": "Ответ был обрезан из-за лимита токенов. Продолжи с того места где остановился.",
                    })

                else:
                    print(f"Неожиданный finish_reason: {finish_reason!r}")
                    break



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="День 19 — агент с пайплайном MCP-инструментов (SSE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Модель OpenAI",
    )
    parser.add_argument(
        "--task",
        default=(
            "Найди в файлах всё, что связано с темой 'pipeline', "
            "суммаризируй найденное и сохрани результат в файл 'pipeline_summary'."
        ),
        help="Задача для агента",
    )
    parser.add_argument(

        "--search-dir",
        default=".",
        help="Директория для инструмента search",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Директория для сохранения Markdown-файлов",
    )
    parser.add_argument(
        "--sse-url",
        default="http://127.0.0.1:8000/sse",
        help="URL SSE-эндпоинта MCP-сервера",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_agent(
            task=args.task,
            model=args.model,
            sse_url=args.sse_url,
            search_dir=args.search_dir,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
