"""
Минимальный MCP-клиент: подключается к публичному HTTP-серверу
и выводит список его tools.

Зависимости:
    pip install mcp

Использование:
    python mcp_list_tools.py
    python mcp_list_tools.py --url https://example.com/mcp --transport streamable-http
    python mcp_list_tools.py --url https://example.com/sse --transport sse
"""

import asyncio
import argparse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

# Публичный MCP-сервер Cloudflare (Streamable HTTP, без авторизации)
DEFAULT_URL = "https://docs.mcp.cloudflare.com/mcp"
DEFAULT_TRANSPORT = "streamable-http"


async def list_tools_streamable_http(url: str) -> None:
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            _print_tools(result.tools)


async def list_tools_sse(url: str) -> None:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            _print_tools(result.tools)


def _print_tools(tools) -> None:
    print(f"Найдено инструментов: {len(tools)}\n")
    for tool in tools:
        print(f"  • {tool.name}")
        if tool.description:
            # первая строка описания
            desc = tool.description.strip().splitlines()[0]
            print(f"    {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Список tools удалённого MCP-сервера")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL MCP-сервера")
    parser.add_argument(
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["streamable-http", "sse"],
        help="Транспорт: streamable-http (новый) или sse (deprecated)",
    )
    args = parser.parse_args()

    print(f"Подключение к {args.url} [{args.transport}]...\n")

    if args.transport == "streamable-http":
        asyncio.run(list_tools_streamable_http(args.url))
    else:
        asyncio.run(list_tools_sse(args.url))


if __name__ == "__main__":
    main()
