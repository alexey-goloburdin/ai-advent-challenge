# PR Review Agent — Day 32


CLI-агент для автоматического AI-ревью GitHub Pull Request'ов.

## Установка

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```


## Переменные окружения

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_URL="https://openai.api.proxyapi.ru/v1"

# Опционально — GitHub токен, чтобы не упереться в rate limit
export GITHUB_TOKEN="ghp_..."

```

## Запуск

```bash
python main.py <PR_URL> <MODEL> [--output review.md] [--repo-dir /tmp/myrepo]
```

### Примеры

```bash
# Базовый запуск
python main.py https://github.com/fastapi/fastapi/pull/15269 gpt-4o

# Указать выходной файл явно

python main.py https://github.com/fastapi/fastapi/pull/15269 gpt-4o-mini -o my_review.md

# Переиспользовать уже скачанный репозиторий (не клонировать заново)
python main.py https://github.com/fastapi/fastapi/pull/15269 gpt-4o --repo-dir /tmp/fastapi
```

## Что делает агент

```
[1/4] Fetch PR info      — GitHub API: метаданные PR, diff, список файлов
[2/4] Clone repo         — shallow clone базовой ветки (--depth 1)
[3/4] Build RAG index    — AST-чанкинг .py файлов + Markdown docs → ChromaDB
[4/4] Run AI review      — RAG-запросы по diff → промпт → LLM → Markdown
```

### Что индексируется в RAG

| Источник | Метод разбивки |
|---|---|
| Все `.py` файлы репо | AST: каждая функция/класс = отдельный чанк |
| `README*.md` | По заголовкам `##` |
| `docs/ru/docs/**/*.md` | По заголовкам `##` |

### Структура ревью

- **Summary** — что делает PR
- **Potential Bugs** — конкретные баги со ссылками на файлы/строки
- **Architectural Issues** — проблемы дизайна (с учётом RAG-контекста)
- **Recommendations** — улучшения: производительность, читаемость, тесты
- **Conclusion** — итог: Approve / Request Changes / Needs Discussion

## Архитектура

```
main.py
├── src/github.py   — парсинг URL, GitHub API, git clone
├── src/rag.py      — AST-чанкинг, ChromaDB in-memory, поиск
└── src/agent.py    — сборка контекста, промпты, вызов LLM
```
