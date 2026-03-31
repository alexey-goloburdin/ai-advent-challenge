# Day 31 — Dev Assistant

Ассистент разработчика с RAG по документации проекта и MCP-интеграцией с git.

## Что умеет

- Индексирует `README.md` и `docs/*.md` выбранного проекта при запуске
- Отвечает на вопросы о проекте через команду `/help`
- Знает текущую git-ветку через MCP

## Требования

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) с запущенными моделями:
  - чат-модель (например, `qwen/qwen3.5-9b`)
  - эмбеддинг-модель `text-embedding-nomic-embed-text-v1.5`
- LM Studio API доступен на `http://localhost:1234`

## Установка

```bash
pip install mcp fastmcp
```

## Запуск

```bash
python main.py --project /путь/к/вашему/проекту
```

При старте агент автоматически:
1. Читает `README.md` и все `docs/*.md` из указанного проекта
2. Строит векторный индекс через LM Studio embedding API
3. Сохраняет индекс в `.dev_assistant_index.json`
4. Запускает MCP git-сервер

## Использование

```
> /help как запустить проект?
> /help какие форматы документов поддерживаются?
> /help что такое ИНН и как он извлекается?
> exit
```

## Структура проекта

```
day31-dev-assistant/
├── src/
│   ├── rag/
│   │   ├── loader.py       # Загрузка и разбивка markdown по секциям
│   │   ├── embedder.py     # Эмбеддинги через LM Studio API
│   │   └── retriever.py    # Cosine similarity поиск + сохранение индекса
│   ├── mcp/
│   │   ├── git_server.py   # FastMCP сервер: инструмент git_branch
│   │   └── client.py       # stdio клиент (JSON-RPC 2.0 через subprocess)
│   └── agent/
│       └── assistant.py    # Формирование промпта и вызов LLM
├── main.py                 # Точка входа, CLI, основной цикл
├── requirements.txt
└── README.md
```

## Как это работает

```
/help <вопрос>
      │
      ├─ RAG: embed(вопрос) → cosine similarity → top-3 чанка из документации
      │
      ├─ MCP: git_branch() → текущая ветка
      │
      └─ LLM: system prompt + документация + ветка + вопрос → ответ
```

Индекс хранится в `.dev_assistant_index.json` и пересоздаётся при каждом запуске. Файл добавьте в `.gitignore`.

## Конфигурация

Модели и URL задаются константами в исходниках:

| Файл | Константа | Значение по умолчанию |
|------|-----------|-----------------------|
| `src/rag/embedder.py` | `LM_STUDIO_URL` | `http://localhost:1234/v1/embeddings` |
| `src/rag/embedder.py` | `EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` |
| `src/agent/assistant.py` | `LM_STUDIO_CHAT_URL` | `http://localhost:1234/v1/chat/completions` |
| `src/agent/assistant.py` | `CHAT_MODEL` | `qwen/qwen3.5-9b` |

## Зависимости

```
mcp
fastmcp
```

Стандартная библиотека Python (`urllib`, `argparse`, `subprocess`, `json`, `math`, `pathlib`) — без дополнительных пакетов для основной логики.
