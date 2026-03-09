# Research Agent

CLI-агент для исследования тем с конечным автоматом состояний (Task State Machine).

## Концепция

Агент проводит структурированное исследование темы по шагам:

```
IDLE → CLARIFICATION → PLANNING → SEARCHING → ANALYZING → SYNTHESIZING → REVIEW → DONE
                                      ↑            │
                                      └────────────┘
                                   (нужно больше данных)
```

Ключевая особенность — **персистентное состояние**. Можно остановить агента на любом этапе, закрыть программу и продолжить позже без потери контекста.

## Зачем это нужно

**Без state machine:** агент при перезапуске не знает, где остановился — начинает сначала или путается.

**С state machine:** состояние явно хранит фазу, собранные данные и следующее действие. Перезапуск продолжает с того же места.

## Установка

Зависимости — только стандартная библиотека Python 3.10+.

```bash
# Клонировать или скопировать файл
curl -O https://example.com/research_agent.py

# Установить API ключ OpenAI
export OPENAI_API_KEY='sk-...'
```

## Запуск

```bash
python3 research_agent.py
```

### Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OPENAI_API_KEY` | — | API ключ OpenAI (обязательно) |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | Базовый URL API |
| `OPENAI_MODEL` | `gpt-4o-mini` | Модель для запросов |
| `USE_MOCK_SEARCH` | `false` | Использовать мок вместо реального поиска |

### Пример с локальной моделью

```bash
export OPENAI_API_BASE='http://localhost:1234/v1'
export OPENAI_API_KEY='not-needed'
export OPENAI_MODEL='local-model'
python3 research_agent.py
```

## Использование

### Команды

| Команда | Действие |
|---------|----------|
| `/new` | Начать новое исследование |
| `/ready` | Подтвердить переход к следующему этапу |
| `/back` | Вернуться к предыдущему этапу |
| `/status` | Показать текущее состояние |
| `/clear` | Сбросить состояние |
| `/quit` | Выйти (состояние сохранится) |
| `/done` | Завершить исследование (в фазе REVIEW) |
| `/export` | Сохранить результаты в `research_output.md` |

### Пример сессии

```
> Сравни подходы к кэшированию в Python веб-приложениях

❓ Состояние: clarification

Уточню scope исследования:
1. Интересует серверное кэширование, клиентское или оба?
2. Какой уровень глубины — обзор или глубокое погружение?
3. Есть ли конкретный стек (Django, FastAPI, Flask)?

💡 Ответьте на вопросы. Когда закончите — введите /ready

> Серверное, FastAPI

✓ Принято.

Текущий scope:
Серверное, FastAPI

Добавьте ещё уточнений или введите /ready для продолжения

> Redis vs in-memory, нужно понять когда что использовать

✓ Принято.

Текущий scope:
Серверное, FastAPI
Redis vs in-memory, нужно понять когда что использовать

Добавьте ещё уточнений или введите /ready для продолжения

> /ready

📋 Сформированы поисковые запросы:
  • redis vs in-memory cache python
  • fastapi caching strategies
  • when to use redis python
  • lru_cache vs redis performance

💡 Проверьте запросы. Введите /ready для начала поиска, 
или добавьте свои запросы текстом

> python functools caching

✓ Добавлен запрос: python functools caching

Текущие запросы:
  • redis vs in-memory cache python
  • fastapi caching strategies
  • when to use redis python
  • lru_cache vs redis performance
  • python functools caching

Добавьте ещё или /ready для начала поиска

> /ready

🔍 Начинаю поиск по 5 запросам...

Нажмите Enter для продолжения

>

🔍 Выполняю поиск...
  Ищу: redis vs in-memory cache python
  Ищу: fastapi caching strategies
  ...

Найдено 12 уникальных источников.

>

🔬 Анализирую источники...
...

> /back   # если хотим вернуться к поиску

↩️  Вернулись к формированию выводов.
...
```

### Пауза и продолжение

```bash
# Сессия 1: начали исследование, вышли на середине
python3 research_agent.py
> Микросервисы vs монолит
...
> /quit
💾 Состояние сохранено. До свидания!

# Сессия 2: продолжаем с того же места
python3 research_agent.py
📂 Загружено сохранённое исследование: Микросервисы vs монолит
🔬 Состояние: analyzing
   Источников: 8

>  # продолжаем с фазы анализа
```

## Архитектура

### Структура состояния

```python
@dataclass
class ResearchState:
    phase: ResearchPhase          # текущая фаза
    topic: str                    # исходная тема
    clarified_scope: str          # уточнённый scope
    research_questions: list[str] # поисковые запросы
    sources: list[dict]           # найденные источники
    findings: list[str]           # результаты анализа
    contradictions: list[str]     # выявленные противоречия
    draft_conclusion: str         # финальный вывод
    conversation_history: list    # история диалога
    created_at: str
    updated_at: str
```

### Переходы между состояниями

```
CLARIFICATION  ──/ready──▶  PLANNING  ──/ready──▶  SEARCHING
      │                         │                       │
      │◀────────/back───────────│                       │
      │                                                 ▼
      │                                            ANALYZING
      │                                                 │
      │                         ┌─────/back─────────────┤
      │                         ▼                       │
      │                    SEARCHING ◀──(нужно больше)──┤
      │                                                 │
      │                                                 ▼
      │                                           SYNTHESIZING
      │                                                 │
      │                         ┌─────/back─────────────┤
      │                         ▼                       │
      │                    ANALYZING                    │
      │                                                 ▼
      │                                              REVIEW
      │                                                 │
      │          ┌──────────────┼──────────────┐        │
      │          ▼              ▼              ▼        │
      │    (углубление)      /done         /export      │
      │          │              │              │        │
      │          ▼              ▼              │        │
      │      PLANNING         DONE            │        │
      │                                       │        │
      └──────────────────────────────────────/back─────┘
```

Ключевые команды перехода:
- `/ready` — подтвердить и перейти к следующему этапу
- `/back` — вернуться к предыдущему этапу
- `/done` — завершить исследование

### Файлы

| Файл | Назначение |
|------|------------|
| `research_agent.py` | Основной код агента |
| `research_state.json` | Персистентное состояние (создаётся автоматически) |
| `research_output.md` | Экспорт результатов (по команде `/export`) |

## Тестирование без API

Для отладки без реальных запросов к OpenAI и поиску:

```bash
export USE_MOCK_SEARCH=true
```

Мок-поиск возвращает заготовленные данные для ключевых слов: `redis`, `cache`, `python`.

## Расширение

### Добавление новых фаз

1. Добавить значение в `ResearchPhase`
2. Создать метод `_handle_новая_фаза()` в `ResearchAgent`
3. Добавить case в `process_input()`
4. Обновить переходы в существующих фазах

### Смена поискового движка

Заменить реализацию `real_web_search()`:

```python
def real_web_search(query: str, max_results: int = 5) -> list[Source]:
    # Ваша реализация: SerpAPI, Tavily, Brave Search, etc.
    ...
```

### Интеграция с другими LLM

Изменить `OPENAI_API_BASE` и `OPENAI_MODEL`:

```bash
# Anthropic через прокси
export OPENAI_API_BASE='https://anthropic-proxy.example.com/v1'
export OPENAI_MODEL='claude-3-haiku-20240307'

# Ollama
export OPENAI_API_BASE='http://localhost:11434/v1'
export OPENAI_MODEL='llama3'
```

## Ограничения

- Поиск через DuckDuckGo может блокироваться (есть fallback на мок)
- Нет параллельных запросов (поиск последовательный)
- История диалога растёт неограниченно (нет обрезки контекста)
- Один активный research за раз
