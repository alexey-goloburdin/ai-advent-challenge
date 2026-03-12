# Композиция MCP-инструментов

Автоматический пайплайн из трёх MCP-инструментов: поиск по локальным файлам → суммаризация через LLM → сохранение результата в Markdown.

## Структура

```
day19/
├── server.py   # MCP-сервер (FastMCP, транспорт SSE)
├── agent.py    # Агент (agentic loop, OpenAI tool calling)
└── README.md
```

## Требования

```bash
pip install mcp uvicorn starlette
```

Переменные окружения:

| Переменная | Обязательная | По умолчанию | Описание |
|---|---|---|---|
| `OPENAI_API_KEY` | да | — | Ключ для доступа к LLM |
| `OPENAI_BASE_URL` | нет | `https://api.openai.com` | Base URL API (для совместимых провайдеров) |

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Или любой OpenAI-совместимый провайдер (OpenRouter, LM Studio, Ollama...)

export OPENAI_BASE_URL=https://api.your-provider.com
export OPENAI_API_KEY=your-key
```

## Запуск


**Терминал 1 — MCP-сервер:**


```bash
python server.py

# Сервер поднимается на http://127.0.0.1:8000/sse
```


**Терминал 2 — агент:**


```bash
python agent.py --search-dir ./docs --output-dir ./output
```

### Аргументы агента

| Аргумент | По умолчанию | Описание |
|---|---|---|

| `--model` | `gpt-4o-mini` | Модель OpenAI |
| `--task` | поиск по теме 'pipeline' | Задача в свободной форме |
| `--search-dir` | `.` | Директория для инструмента `search` |
| `--output-dir` | `./output` | Куда сохранять Markdown-файлы |
| `--sse-url` | `http://127.0.0.1:8000/sse` | URL SSE-эндпоинта MCP-сервера |

Пример с кастомной задачей:


```bash
python agent.py \
  --model gpt-4o \
  --task "Найди всё про авторизацию, суммаризируй и сохрани в auth_summary" \
  --search-dir ./src \
  --output-dir ./reports
```

## Архитектура

```
┌─────────────────────────────────────────────┐
│                   agent.py                  │
│                                             │
│  argparse → run_agent()                     │
│      │                                      │
│      ├── sse_client + ClientSession         │  MCP (SSE/HTTP)
│      │        │ list_tools()   ─────────────┼──────────────► server.py
│      │        │ call_tool()    ─────────────┼──────────────► search / summarize / saveToFile
│      │                                      │
│      └── openai_chat() [urllib]             │
│               │                             │
│               └── agentic loop             │
│                    Шаг 1: LLM → tool_calls  │
│                    Шаг 2: выполнить инструмент через MCP
│                    Шаг 3: добавить результат в messages
│                    Шаг 4: повторить до finish_reason=stop
└─────────────────────────────────────────────┘
```

## Инструменты MCP-сервера


### `search`


Ищет строку в текстовых файлах директории (рекурсивно).

```json
{
  "query": "pipeline",
  "directory": "./docs",
  "pattern": "*.md",
  "case_sensitive": false
}
```

Возвращает:

```json
{

  "query": "pipeline",
  "directory": "/abs/path/docs",
  "matches": [
    {"file": "arch.md", "line": 4, "text": "...pipeline..."}
  ]
}
```

### `summarize`

Суммаризирует переданный текст через OpenAI API (вызов идёт на стороне сервера, тоже через `urllib`).

```json
{
  "text": "...",
  "model": "gpt-4o-mini",
  "language": "Russian"
}
```

Возвращает:

```json
{
  "summary": "Краткое резюме...",
  "model": "gpt-4o-mini"
}
```

### `saveToFile`


Сохраняет текст в `.md`-файл. Создаёт директорию при необходимости.

```json
{

  "content": "Текст резюме",
  "filename": "pipeline_summary",
  "output_dir": "./output",
  "title": "Заголовок H1"
}
```

Возвращает:

```json
{
  "saved": true,
  "path": "/abs/path/output/pipeline_summary.md",
  "bytes": 1024

}
```

## Пайплайн

LLM сам управляет порядком вызовов. System-промпт задаёт ожидаемую последовательность:

```
search → summarize → saveToFile → финальный ответ
```

Данные передаются между инструментами через историю сообщений: результат каждого инструмента добавляется в `messages` с ролью `tool`, следующий вызов LLM видит его и использует как вход для следующего шага.

## Транспорт

MCP-сервер использует **SSE (Server-Sent Events)**. Агент подключается через `mcp.client.sse.sse_client` — устанавливает постоянное SSE-соединение для получения событий, и отправляет сообщения HTTP POST на `/messages/`.


Все HTTP-запросы к OpenAI API выполняются через стандартную библиотеку `urllib` — без сторонних HTTP-клиентов.
