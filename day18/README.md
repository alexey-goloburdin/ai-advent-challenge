# Day 18 — MCP Reminder Server

Пример MCP-инструмента с отложенным выполнением задач.  
Сервер хранит напоминания в JSON, агент опрашивает сервер каждые N секунд и печатает сработавшие напоминания.

## Структура проекта

```
.
├── reminder_server.py   # MCP-сервер (stdio transport)
├── reminder_agent.py    # Агент-клиент (24/7 polling loop)
├── reminders.json       # База данных (создаётся автоматически)
└── README.md
```

## Установка

```bash
pip install mcp
```

Требуется Python 3.10+.

## Переменные окружения

| Переменная        | Обязательная | Описание                                      |
|-------------------|:------------:|-----------------------------------------------|

| `OPENAI_BASE_URL` | ✅           | Базовый URL API, например `https://api.openai.com/v1` |
| `OPENAI_API_KEY`  | ✅           | Ключ доступа к API                            |

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...

```

Агент завершится с ошибкой при старте, если хотя бы одна из переменных не задана.


## Аргументы командной строки

```
python reminder_agent.py [--model MODEL]
```

| Аргумент  | По умолчанию   | Описание                  |
|-----------|----------------|---------------------------|
| `--model` | `gpt-4o-mini`  | Название LLM-модели       |

Примеры:

```bash
python reminder_agent.py
python reminder_agent.py --model gpt-4o
python reminder_agent.py --model qwen/qwen-2.5-72b-instruct
```

## Запуск

```bash
python reminder_agent.py --model gpt-4o-mini
```

Агент сам запускает `reminder_server.py` как дочерний процесс — отдельно запускать сервер не нужно.

Остановка: `Ctrl+C`

## Пример вывода

```
$ python reminder_agent.py --model gpt-4o-mini

[agent]  12:00:00 Подключаемся к MCP-серверу: reminder_server.py
[agent]  12:00:00 Доступные tools: ['add_reminder', 'get_due_reminders', 'list_reminders']
[agent]  12:00:00 Агент запущен. Модель: gpt-4o-mini. Интервал проверки: 10 сек.
[agent]  12:00:00 ✅ Напоминание #1 добавлено. Сработает в 12:00:15.
[agent]  12:00:10 Нет новых напоминаний.
[agent]  12:00:20 Сработало напоминание #1, отправляю в LLM...

======================================================
🔔  НАПОМИНАНИЕ #1: Выпить стакан воды 💧
    ⏰  12:00:15
    🤖  Не забудь выпить стакан воды — отличный способ поддержать себя в тонусе! 💧
======================================================
```

## Архитектура

```
reminder_agent.py          reminder_server.py
┌──────────────────┐       ┌──────────────────────────┐
│                  │ stdio │                          │
│  ClientSession   │◄─────►│  Server (FastMCP)        │
│                  │       │                          │
│  agent_loop()    │       │  tools:                  │
│  ┌─────────────┐ │       │  • add_reminder()        │
│  │ while True: │ │       │  • get_due_reminders()   │
│  │  sleep(10)  │ │       │  • list_reminders()      │
│  │  get_due()  │ │       │                          │
│  │  print 🔔   │ │       │  reminders.json          │
│  └─────────────┘ │       │  (персистентное хранение)│
└──────────────────┘       └──────────────────────────┘
```

**Сервер** — пассивный. Хранит напоминания в `reminders.json`, отвечает на tool-вызовы по MCP-протоколу.

**Агент** — активный. Каждые `POLL_INTERVAL` секунд вызывает `get_due_reminders`. Если список не пуст — печатает напоминания в консоль.

## MCP Tools

### `add_reminder`

Добавить напоминание с отложенным сроком.

| Параметр        | Тип    | Описание                                |
|-----------------|--------|-----------------------------------------|
| `text`          | string | Текст напоминания                       |
| `delay_seconds` | number | Через сколько секунд должно сработать   |

```python
await session.call_tool(
    "add_reminder",
    arguments={"text": "Позвонить другу", "delay_seconds": 3600},
)
```

### `get_due_reminders`

Вернуть список напоминаний, у которых наступило время. Отработавшие помечаются как `done: true`.

Возвращает JSON-массив:

```json
[
  {
    "id": 1,
    "text": "Позвонить другу",
    "fire_at": 1700000000.0,
    "fire_at_human": "2024-11-14 15:00:00",
    "done": true,
    "created_at": 1699996400.0
  }
]
```


### `list_reminders`

Вернуть все напоминания — активные и выполненные.

## Формат хранилища

`reminders.json` — обычный JSON-массив объектов:

```json
[
  {
    "id": 1,
    "text": "Выпить стакан воды 💧",
    "fire_at": 1700000015.0,
    "fire_at_human": "2024-11-14 12:00:15",
    "done": true,
    "created_at": 1700000000.0
  }
]
```

Файл создаётся автоматически при добавлении первого напоминания.  
Данные сохраняются между перезапусками — невыполненные напоминания не теряются.

## Настройки агента

В начале `reminder_agent.py`:

```python
POLL_INTERVAL = 10   # секунд между проверками get_due_reminders
```

## Важно: stdout в MCP stdio-сервере

В `reminder_server.py` все логи направлены в `sys.stderr`.  
`stdout` зарезервирован для JSON-RPC сообщений MCP-протокола — любой случайный `print()` в stdout сломает парсинг и уронит соединение.

```python
# ❌ Нельзя в сервере
print("debug info")

# ✅ Правильно
import sys
print("debug info", file=sys.stderr)
# или
import logging
logging.info("debug info")  # по умолчанию идёт в stderr
```
