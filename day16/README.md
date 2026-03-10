# mcp-list-tools

Минимальный MCP-клиент на Python: подключается к удалённому HTTP-серверу и выводит список его инструментов (tools) в stdout.

## Требования

- Python 3.10+
- Пакет `mcp` (официальный Python SDK)

```bash
pip install mcp
```

## Быстрый старт

```bash
python mcp_list_tools.py
```

По умолчанию подключается к публичному серверу документации Cloudflare (`https://docs.mcp.cloudflare.com/mcp`) без какой-либо авторизации.

Пример вывода:

```
Подключение к https://docs.mcp.cloudflare.com/mcp [streamable-http]...

Найдено инструментов: 3

  • search
    Search Cloudflare's developer documentation
  • get_page
    Fetch a specific documentation page by URL
  • list_sections
    List all top-level documentation sections
```

## Использование

```
python mcp_list_tools.py [--url URL] [--transport {streamable-http,sse}]
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--url` | `https://docs.mcp.cloudflare.com/mcp` | URL MCP-сервера |
| `--transport` | `streamable-http` | Транспортный протокол |

### Примеры

```bash
# Подключиться к произвольному Streamable HTTP серверу
python mcp_list_tools.py --url https://example.com/mcp

# Подключиться к старому SSE серверу
python mcp_list_tools.py --url https://example.com/sse --transport sse
```

## Транспорты

MCP поддерживает два HTTP-транспорта.

**Streamable HTTP** — актуальный стандарт (с марта 2025). Общение через один endpoint по HTTP POST, при необходимости сервер апгрейдит соединение до SSE-стрима. Используйте этот вариант для новых серверов.

**SSE** — устаревший транспорт (deprecated с марта 2025). Два отдельных endpoint: `/sse` для получения ответов и `/messages` для отправки запросов. Поддерживается для совместимости со старыми серверами.

## Публичные серверы для тестирования

| Сервер | URL | Транспорт | Авторизация |
|---|---|---|---|
| Cloudflare Docs | `https://docs.mcp.cloudflare.com/mcp` | Streamable HTTP | Нет |
| Cloudflare API | `https://api.mcp.cloudflare.com/mcp` | Streamable HTTP | OAuth |

## Структура кода

```
mcp_list_tools.py
├── list_tools_streamable_http()  # клиент для нового транспорта
├── list_tools_sse()              # клиент для SSE
├── _print_tools()                # форматирование вывода
└── main()                        # CLI-обёртка (argparse)
```

Основной паттерн подключения одинаков для обоих транспортов:

```python
async with streamablehttp_client(url) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()          # MCP handshake
        result = await session.list_tools() # запрос списка tools
```
