# Day 20 — Research Agent (Orchestration MCP)

CLI-агент, который проводит исследование заданной темы, используя два MCP-сервера,
и сохраняет структурированный отчёт в Markdown-файл.

## Структура проекта

```
day20_research_agent/
├── agent.py              # Оркестратор: CLI, agentic loop, маршрутизация
├── server_search.py      # MCP-сервер: поиск в интернете (DuckDuckGo)
├── server_filesystem.py  # MCP-сервер: работа с локальными файлами
├── requirements.txt
└── README.md
```


## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                     agent.py                        │

│                  (оркестратор)                      │
│                                                     │
│  messages[] ──► OpenAI API ──► tool_calls           │
│                                    │                │
│              ┌─────────────────────┤                │
│              ▼                     ▼                │
│   ┌──────────────────┐  ┌──────────────────────┐   │
│   │  server_search   │  │  server_filesystem   │   │
│   │   (stdio MCP)    │  │     (stdio MCP)      │   │
│   │                  │  │                      │   │
│   │  search_web()    │  │  write_file()        │   │
│   │  search_news()   │  │  read_file()         │   │
│   └──────────────────┘  │  list_files()        │   │
│                          │  append_file()       │   │
│                          └──────────────────────┘   │
└─────────────────────────────────────────────────────┘

```

Оба сервера запускаются агентом как дочерние процессы через **stdio-транспорт**.

Агент обнаруживает инструменты на старте и строит реестр `tool_name → server`,
по которому маршрутизирует каждый вызов.

## Флоу исследования

1. Агент получает тему от пользователя
2. `search_web` — несколько поисковых запросов по теме
3. `search_news` — поиск свежих новостей
4. LLM анализирует найденное и формирует отчёт
5. `write_file` — сохраняет `report_<тема>.md`

6. Агент сообщает пользователю о готовности

## Установка

```bash
pip install mcp duckduckgo-search
```

## Настройка


Переменные окружения:

| Переменная       | Описание                               | По умолчанию                                          |

|------------------|----------------------------------------|-------------------------------------------------------|
| `OPENAI_API_KEY` | Токен доступа к API **обязательно**    | —                                                     |
| `OPENAI_API_URL` | URL эндпоинта                          | `https://api.openai.com/v1/chat/completions`          |
| `FS_WORKSPACE`   | Директория для сохранения файлов       | `.` (текущая директория)                              |

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_API_URL=https://api.openai.com/v1/chat/completions  # или другой совместимый
export FS_WORKSPACE=/home/user/research                           # опционально
```

## Запуск

```bash
# Тема задаётся интерактивно

python agent.py --model gpt-4o-mini

# Тема передаётся аргументом
python agent.py --model gpt-4o --topic "Квантовые компьютеры 2025"


# Другой совместимый провайдер
OPENAI_API_URL=https://api.together.xyz/v1/chat/completions \
OPENAI_API_KEY=... \
python agent.py --model meta-llama/Llama-3-70b-chat-hf --topic "LLM fine-tuning"
```

## Пример вывода

```
🚀 Запуск MCP-серверов...

  ✓ search-server готов
  ✓ filesystem-server готов

📚 Доступные инструменты: search_web, search_news, write_file, read_file, list_files, append_file
🔍 Тема исследования: Квантовые компьютеры 2025
────────────────────────────────────────────────────────────

[Шаг 1] Запрос к модели...
  → Вызовов инструментов: 2
  🔧 [search] search_web(query='квантовые компьютеры 2025')
     ↳ ### IBM Quantum System Two...
  🔧 [search] search_news(query='quantum computing 2025 breakthroughs')
     ↳ ### Google Willow chip sets new record...

[Шаг 2] Запрос к модели...
  → Вызовов инструментов: 1
  🔧 [search] search_web(query='квантовые компьютеры применение промышленность')

     ↳ ### Практическое применение квантовых вычислений...

[Шаг 3] Запрос к модели...
  → Вызовов инструментов: 1
  🔧 [filesystem] write_file(filename='report_квантовые_компьютеры_2025.md', ...)
     ↳ Файл сохранён: /home/user/research/report_квантовые_компьютеры_2025.md

[Шаг 4] Запрос к модели...


✅ Агент завершил работу:


Исследование завершено. Отчёт сохранён в файл:
report_квантовые_компьютеры_2025.md
```

## Расширение


Добавить новый MCP-сервер — три шага:

1. Создать `server_xxx.py` с `FastMCP` и нужными инструментами
2. В `agent.py` добавить запуск сервера в `main()` по аналогии с существующими
3. Добавить сессию в словарь `sessions`

Агент автоматически обнаружит новые инструменты и добавит их в маршрутизатор.
