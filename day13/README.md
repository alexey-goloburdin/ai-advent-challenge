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
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Базовый URL API |
| `OPENAI_MODEL` | `gpt-4o-mini` | Модель для запросов |
| `USE_MOCK_SEARCH` | `false` | Использовать мок вместо реального поиска |

### Пример с локальной моделью

```bash
export OPENAI_BASE_URL='http://localhost:1234/v1'
export OPENAI_API_KEY='not-needed'
export OPENAI_MODEL='local-model'
python3 research_agent.py
```

## Использование

### Команды

| Команда | Действие |
|---------|----------|
| `/new` | Начать новое исследование |
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

> Серверное, FastAPI, Redis vs in-memory, нужно понять когда что использовать

📋 Состояние: planning

Сформированы поисковые запросы:
  • redis vs in-memory cache python
  • fastapi caching strategies
  • when to use redis python
  • lru_cache vs redis performance

Начинаю поиск... (нажмите Enter)

>

🔍 Выполняю поиск...
  Ищу: redis vs in-memory cache python
  Ищу: fastapi caching strategies
  ...

Найдено 12 уникальных источников.

Анализирую... (нажмите Enter)

>

🔬 Анализирую источники...

ДОСТАТОЧНО

Ключевые идеи:
- Redis нужен для распределённого кэша (несколько процессов/серверов)
- In-memory (lru_cache, dict) достаточно для single-process
- Redis добавляет latency ~1ms, но даёт персистентность
...

Формирую выводы... (нажмите Enter)

>

============================================================
РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ
============================================================

Тема: Сравни подходы к кэшированию в Python веб-приложениях
Scope: Серверное, FastAPI, Redis vs in-memory

## Краткое резюме
...

Что дальше?
  • Введите уточняющий вопрос для углубления
  • /done — завершить исследование
  • /export — сохранить в файл

> Подробнее про инвалидацию кэша

📋 Состояние: planning

Углубляю исследование: инвалидация кэша
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

```python
class ResearchPhase(Enum):
    IDLE = "idle"                # → CLARIFICATION (при вводе темы)
    CLARIFICATION = "clarification"  # → PLANNING (после уточнений)
    PLANNING = "planning"        # → SEARCHING (запросы сформированы)
    SEARCHING = "searching"      # → ANALYZING (источники собраны)
    ANALYZING = "analyzing"      # → SYNTHESIZING | SEARCHING (нужно больше)
    SYNTHESIZING = "synthesizing"    # → REVIEW (выводы готовы)
    REVIEW = "review"            # → DONE | PLANNING (углубление)
    DONE = "done"                # финал
```

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

Изменить `OPENAI_BASE_URL` и `OPENAI_MODEL`:

```bash
# Anthropic через прокси
export OPENAI_BASE_URL='https://anthropic-proxy.example.com/v1'
export OPENAI_MODEL='claude-3-haiku-20240307'

# Ollama
export OPENAI_BASE_URL='http://localhost:11434/v1'
export OPENAI_MODEL='llama3'
```

## Ограничения

- Поиск через DuckDuckGo может блокироваться (есть fallback на мок)
- Нет параллельных запросов (поиск последовательный)
- История диалога растёт неограниченно (нет обрезки контекста)
- Один активный research за раз
