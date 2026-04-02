# Day 33 — AI-ассистент поддержки пользователей

## Структура проекта

```
support-agent/
├── agent.py          # Точка входа, CLI-чат
├── crm_server.py     # MCP stdio-сервер для CRM
├── crm.json          # База пользователей и тикетов
├── docs/
│   ├── faq.txt       # FAQ
│   └── technical.txt # Техническая документация
└── src/
    ├── mcp_client.py  # MCP stdio-клиент
    ├── openai_client.py # Обёртка над OpenAI chat completions
    └── rag.py         # RAG: индексация и поиск
```

## Запуск

```bash
export OPENAI_API_URL="https://openai.api.proxyapi.ru/v1"
export OPENAI_API_KEY="sk-..."

python agent.py --model gpt-4o
```

## Флоу работы

1. **RAG-индексация** — при запуске все `docs/*.txt` нарезаются на чанки и
   индексируются через `text-embedding-3-small`.

2. **MCP CRM** — запускается `crm_server.py` как subprocess. Агент общается с
   ним по JSON-RPC 2.0 (stdio).

3. **Идентификация клиента** — агент спрашивает email, ищет через MCP и
   подтягивает все тикеты. Контекст вшивается в system prompt.

4. **Чат-цикл** — каждый запрос:
   - обогащается top-3 релевантными чанками из RAG
   - LLM может вызывать CRM-инструменты (tool calling loop)
   - возвращает финальный текстовый ответ

## MCP-инструменты CRM

| Инструмент            | Описание                            |
|-----------------------|-------------------------------------|
| `get_user`            | Данные пользователя по ID           |
| `search_user_by_email`| Поиск пользователя по email         |
| `get_ticket`          | Тикет по ID                         |
| `list_user_tickets`   | Все тикеты пользователя             |

## Добавление документации

Просто кладите `.txt`-файлы в папку `docs/` — они подхватятся при следующем
запуске автоматически.

## Тестовые данные

**Пользователи:**
- `alex@example.com` — Pro, 2 тикета (1 открытый: авторизация)
- `maria@example.com` — Free, 1 тикет (запрос экспорта в PDF)
- `dima@example.com` — Enterprise, 1 тикет (API 500 ошибка, critical)
