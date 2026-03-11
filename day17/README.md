# Todo MCP Agent

Два файла, нулевые зависимости — только стандартная библиотека Python 3.10+.

```
todo_agent.py          ← агент: читает ввод, дёргает OpenAI API, вызывает инструменты
todo_mcp_server.py     ← MCP-сервер: управляет задачами, хранит данные в todos.json
```

---

## Архитектура


```
Пользователь
    │  input()

    ▼

todo_agent.py
    │  urllib → POST /v1/chat/completions
    ▼
OpenAI-совместимый API
    │  finish_reason: "tool_calls"
    ▼
todo_agent.py  (agentic loop)
    │  subprocess stdin/stdout  JSON-RPC 2.0
    ▼
todo_mcp_server.py
    │  read/write
    ▼
todos.json
```

Агент крутит **agentic loop**: после каждого ответа модели проверяет `finish_reason`.
Если модель хочет вызвать инструмент — выполняет его через MCP-сервер и отправляет результат обратно.
Цикл продолжается до тех пор, пока модель не вернёт обычный текстовый ответ.

---

## Быстрый старт

```bash
# 1. Задать переменные окружения

export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # опционально, это значение по умолчанию

# 2. Запустить агента (сервер стартует автоматически как subprocess)
python todo_agent.py
```

### Параметры запуска


| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--model` | `gpt-4o-mini` | Модель OpenAI-совместимого API |
| `--server` | `todo_mcp_server.py` | Путь к MCP-серверу |

```bash
python todo_agent.py --model gpt-4o --server ./todo_mcp_server.py
```

---


## Переменные окружения

| Переменная | Обязательная | Описание |
|-----------|-------------|----------|
| `OPENAI_API_KEY` | ✅ | API-ключ |
| `OPENAI_BASE_URL` | ❌ | Base URL API (по умолчанию `https://api.openai.com/v1`) |

### Примеры с разными провайдерами

```bash
# Groq (быстрый, бесплатный тир)

export OPENAI_BASE_URL=https://api.groq.com/openai/v1

export OPENAI_API_KEY=gsk-...
python todo_agent.py --model llama-3.3-70b-versatile

# Локальный Ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
python todo_agent.py --model qwen2.5-coder

# OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-...
python todo_agent.py --model anthropic/claude-3.5-haiku
```

---

## Инструменты MCP-сервера

| Инструмент | Описание |
|-----------|----------|
| `todo_add` | Добавить задачу (поле `title`, опционально `priority`: low / medium / high) |
| `todo_list` | Список задач (фильтр `all` / `active` / `done`) |
| `todo_done` | Отметить задачу выполненной по `id` |
| `todo_delete` | Удалить задачу по `id` |
| `todo_clear_done` | Удалить все выполненные задачи |

Данные хранятся в `todos.json` рядом с `todo_mcp_server.py`.


---

## Пример диалога

```
🤖 Todo Agent (модель: gpt-4o-mini)
   Доступно инструментов: 5
   Base URL: https://api.openai.com/v1

Вы: добавь задачу "написать тесты" с высоким приоритетом
   🔧 todo_add({"title": "написать тесты", "priority": "high"})
   📋 ✅ Задача добавлена: [a3f1c2b0] написать тесты (приоритет: high)
🤖 Агент: Готово! Задача «написать тесты» добавлена с высоким приоритетом.

Вы: покажи активные задачи
   🔧 todo_list({"filter": "active"})
   📋 Активные задачи (1):
       ☐ 🔴 [a3f1c2b0] написать тесты
🤖 Агент: У тебя одна активная задача: «написать тесты» (высокий приоритет).

Вы: отметь a3f1c2b0 как выполненную
   🔧 todo_done({"id": "a3f1c2b0"})
   📋 ✅ Задача [a3f1c2b0] «написать тесты» отмечена как выполненная.
🤖 Агент: Готово, задача выполнена!
```

---


## Что демонстрирует этот пример


**MCP-протокол**
- Транспорт stdio: общение через `subprocess` stdin/stdout

- JSON-RPC 2.0: структура запросов и ответов
- Рукопожатие: `initialize` → `notifications/initialized` → `tools/list` → `tools/call`
- `inputSchema` (JSON Schema) — как модель понимает аргументы инструмента

**Agentic loop**
- `finish_reason: "tool_calls"` — сигнал что модель хочет вызвать инструмент
- Несколько вызовов инструментов в одном ответе (`tool_calls` — массив)

- Роль `tool` в истории сообщений — как передавать результаты обратно модели


**OpenAI-совместимый API**
- Чистые HTTP-запросы через `urllib` без сторонних библиотек
- Конвертация MCP tools → OpenAI function calling format
