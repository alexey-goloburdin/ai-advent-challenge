#!/usr/bin/env python3
"""
Research Agent — CLI-утилита для исследования тем с конечным автоматом состояний.

Состояния:
    CLARIFICATION → PLANNING → SEARCHING → ANALYZING → SYNTHESIZING → REVIEW → DONE

Особенности:
    - Персистентное состояние в JSON-файле
    - Возможность паузы и продолжения
    - Web-поиск через DuckDuckGo (urllib)
    - OpenAI-совместимый API
"""

import json
import os
import re
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================================
# Конфигурация
# ============================================================================

STATE_FILE = Path("research_state.json")
OPENAI_API_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1") + "/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================================
# Состояния конечного автомата
# ============================================================================

class ResearchPhase(str, Enum):
    """Фазы исследования."""
    IDLE = "idle"                        # Начальное состояние
    CLARIFICATION = "clarification"      # Уточняем тему
    PLANNING = "planning"                # Формируем вопросы для поиска
    SEARCHING = "searching"              # Ищем информацию
    ANALYZING = "analyzing"              # Анализируем источники
    SYNTHESIZING = "synthesizing"        # Формируем выводы
    REVIEW = "review"                    # Показываем результат, ждём фидбек
    DONE = "done"                        # Завершено


@dataclass
class Source:
    """Источник информации."""
    title: str
    url: str
    snippet: str


@dataclass
class ResearchState:
    """Полное состояние исследования."""
    phase: ResearchPhase = ResearchPhase.IDLE
    topic: str = ""
    clarified_scope: str = ""
    research_questions: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # list[Source] as dicts
    findings: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    draft_conclusion: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        """Сериализация в словарь."""
        data = asdict(self)
        data["phase"] = self.phase.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchState":
        """Десериализация из словаря."""
        data["phase"] = ResearchPhase(data["phase"])
        return cls(**data)


# ============================================================================
# Персистентность
# ============================================================================

def save_state(state: ResearchState) -> None:
    """Сохранить состояние в файл."""
    state.updated_at = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def load_state() -> ResearchState | None:
    """Загрузить состояние из файла."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ResearchState.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def clear_state() -> None:
    """Удалить файл состояния."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ============================================================================
# Web Search через DuckDuckGo (urllib)
# ============================================================================

USE_MOCK_SEARCH = os.getenv("USE_MOCK_SEARCH", "false").lower() == "true"


# Мок-данные для тестирования без реального поиска
MOCK_SEARCH_DATA = {
    "redis": [
        Source("Redis vs In-Memory Cache", "https://redis.io/docs/compare", 
               "Redis предоставляет персистентность, репликацию и кластеризацию. In-memory кэш проще, но данные теряются при перезапуске."),
        Source("When to use Redis", "https://stackoverflow.com/questions/redis-use-cases",
               "Redis оптимален для: сессий, очередей задач, pub/sub. Простой dict подходит для кэша внутри одного процесса."),
        Source("Redis Performance Benchmarks", "https://redis.io/docs/benchmarks",
               "Redis обрабатывает 100k+ операций/сек. Latency ~0.5ms для локального, ~1-2ms для сетевого."),
    ],
    "cache": [
        Source("Caching Strategies", "https://docs.aws.amazon.com/caching",
               "Основные стратегии: Cache-Aside, Write-Through, Write-Behind. Выбор зависит от паттерна чтения/записи."),
        Source("Python functools.lru_cache", "https://docs.python.org/3/library/functools.html",
               "lru_cache — встроенный декоратор для мемоизации. Простой и эффективный для чистых функций."),
    ],
    "python": [
        Source("Python Best Practices", "https://docs.python-guide.org/",
               "Гид по написанию качественного Python-кода: структура проектов, тестирование, документация."),
        Source("AsyncIO Tutorial", "https://docs.python.org/3/library/asyncio.html",
               "asyncio — библиотека для асинхронного программирования. Используйте async/await для I/O-bound задач."),
    ],
    "default": [
        Source("Wikipedia", "https://wikipedia.org",
               "Свободная энциклопедия с базовой информацией по большинству тем."),
        Source("Stack Overflow", "https://stackoverflow.com",
               "Крупнейший ресурс вопросов и ответов для разработчиков."),
    ]
}


def mock_search(query: str, max_results: int = 5) -> list[Source]:
    """Мок-поиск для тестирования без сети."""
    query_lower = query.lower()
    
    results = []
    for key, sources in MOCK_SEARCH_DATA.items():
        if key in query_lower:
            results.extend(sources)
    
    if not results:
        results = MOCK_SEARCH_DATA["default"]
    
    return results[:max_results]


def real_web_search(query: str, max_results: int = 5) -> list[Source]:
    """
    Поиск через DuckDuckGo HTML-версию.
    Простая реализация без внешних зависимостей.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as e:
        print(f"  [!] Ошибка поиска: {e}")
        print(f"  [!] Переключаюсь на мок-режим")
        return mock_search(query, max_results)
    
    # Парсим результаты (простой regex-парсинг)
    sources = []
    
    # Альтернативный паттерн для сниппетов
    alt_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
        re.DOTALL
    )
    
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL
    )
    
    # Извлекаем URL и заголовки
    urls_titles = alt_pattern.findall(html)
    snippets = snippet_pattern.findall(html)
    
    for i, (url, title) in enumerate(urls_titles[:max_results]):
        # Очищаем URL от редиректа DuckDuckGo
        if "uddg=" in url:
            match = re.search(r'uddg=([^&]+)', url)
            if match:
                url = urllib.parse.unquote(match.group(1))
        
        # Получаем сниппет
        snippet = snippets[i] if i < len(snippets) else ""
        # Очищаем от HTML-тегов
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        
        if url.startswith("http"):
            sources.append(Source(
                title=title.strip(),
                url=url,
                snippet=snippet[:500]
            ))
    
    # Если реальный поиск ничего не вернул, используем мок
    if not sources:
        return mock_search(query, max_results)
    
    return sources


def web_search(query: str, max_results: int = 5) -> list[Source]:
    """Обёртка над поиском: мок или реальный."""
    if USE_MOCK_SEARCH:
        print(f"  [mock] Поиск: {query}")
        return mock_search(query, max_results)
    return real_web_search(query, max_results)


# ============================================================================
# OpenAI API (urllib)
# ============================================================================

def call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """Вызов OpenAI-совместимого API через urllib."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не установлен")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OPENAI_API_URL, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API Error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network Error: {e}")


# ============================================================================
# Промпты для каждой фазы
# ============================================================================

SYSTEM_PROMPT = """Ты — исследовательский ассистент. Помогаешь пользователю глубоко изучить тему.

Твой стиль:
- Структурированный и аналитический
- Выявляешь противоречия между источниками
- Формулируешь чёткие выводы
- Отвечаешь на русском языке

Текущая фаза исследования: {phase}
"""

CLARIFICATION_PROMPT = """Пользователь хочет исследовать тему: "{topic}"

Задай 2-3 уточняющих вопроса, чтобы понять:
1. Какой аспект темы интересует больше всего
2. Какой уровень глубины нужен (обзор vs глубокое погружение)
3. Есть ли конкретный контекст применения

Формат: задай вопросы списком, кратко."""

PLANNING_PROMPT = """Тема исследования: "{topic}"
Уточнённый scope: "{scope}"

Сформулируй 3-5 конкретных поисковых запросов для исследования этой темы.
Запросы должны быть на том языке, на котором больше качественных источников (обычно английский для технических тем).

Формат ответа — только JSON-массив строк:
["запрос 1", "запрос 2", "запрос 3"]"""

ANALYSIS_PROMPT = """Тема: "{topic}"
Scope: "{scope}"

Найденные источники:
{sources}

Проанализируй источники:
1. Какие ключевые идеи повторяются?
2. Есть ли противоречия между источниками?
3. Какие аспекты темы не покрыты и требуют дополнительного поиска?

Если информации достаточно, напиши "ДОСТАТОЧНО" в начале ответа.
Если нужен дополнительный поиск, напиши "НУЖЕН ПОИСК:" и укажи что искать."""

SYNTHESIS_PROMPT = """Тема: "{topic}"
Scope: "{scope}"

Источники и находки:
{sources}

Анализ:
{findings}

Сформируй структурированный вывод по теме:
1. Краткое резюме (2-3 предложения)
2. Ключевые тезисы (3-5 пунктов)
3. Противоречия и нюансы (если есть)
4. Практические рекомендации (если применимо)

Используй информацию только из предоставленных источников."""


# ============================================================================
# Логика переходов между состояниями
# ============================================================================

class ResearchAgent:
    """Агент-исследователь с конечным автоматом состояний."""
    
    def __init__(self, state: ResearchState | None = None):
        self.state = state or ResearchState(created_at=datetime.now().isoformat())
    
    def get_system_message(self) -> dict:
        """Системное сообщение с текущей фазой."""
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(phase=self.state.phase.value)
        }
    
    def process_input(self, user_input: str) -> str:
        """Обработать ввод пользователя и вернуть ответ."""
        # Добавляем в историю
        self.state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Роутинг по состояниям
        match self.state.phase:
            case ResearchPhase.IDLE:
                response = self._handle_idle(user_input)
            case ResearchPhase.CLARIFICATION:
                response = self._handle_clarification(user_input)
            case ResearchPhase.PLANNING:
                response = self._handle_planning(user_input)
            case ResearchPhase.SEARCHING:
                response = self._handle_searching()
            case ResearchPhase.ANALYZING:
                response = self._handle_analyzing()
            case ResearchPhase.SYNTHESIZING:
                response = self._handle_synthesizing()
            case ResearchPhase.REVIEW:
                response = self._handle_review(user_input)
            case ResearchPhase.DONE:
                response = "Исследование завершено. Начните новое с команды /new"
            case _:
                response = "Неизвестное состояние"
        
        # Сохраняем ответ в историю
        self.state.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Персистим состояние
        save_state(self.state)
        
        return response
    
    def _handle_idle(self, topic: str) -> str:
        """Начало нового исследования."""
        self.state.topic = topic
        self.state.phase = ResearchPhase.CLARIFICATION
        
        # Запрашиваем уточнения у LLM
        messages = [
            self.get_system_message(),
            {"role": "user", "content": CLARIFICATION_PROMPT.format(topic=topic)}
        ]
        
        response = call_llm(messages)
        return response + "\n\n💡 Ответьте на вопросы. Когда закончите — введите /ready"
    
    def _handle_clarification(self, user_input: str) -> str:
        """Обработка уточнений от пользователя. Накапливает ответы до /ready."""
        # Команда /ready — переходим к планированию
        if user_input.strip().lower() == "/ready":
            if not self.state.clarified_scope:
                return "⚠️  Сначала ответьте на уточняющие вопросы, затем введите /ready"
            
            # Переходим к планированию
            return self._transition_to_planning()
        
        # Команда /back — возвращаемся к вводу темы
        if user_input.strip().lower() == "/back":
            self.state.phase = ResearchPhase.IDLE
            self.state.clarified_scope = ""
            return "↩️  Вернулись назад. Введите тему исследования:"
        
        # Накапливаем ответы пользователя
        if self.state.clarified_scope:
            self.state.clarified_scope += "\n" + user_input
        else:
            self.state.clarified_scope = user_input
        
        # Показываем что накопили и предлагаем продолжить
        return f"✓ Принято.\n\n" \
               f"Текущий scope:\n{self.state.clarified_scope}\n\n" \
               f"Добавьте ещё уточнений или введите /ready для продолжения"
    
    def _transition_to_planning(self) -> str:
        """Переход от clarification к planning."""
        self.state.phase = ResearchPhase.PLANNING
        
        # Формируем поисковые запросы
        messages = [
            self.get_system_message(),
            {"role": "user", "content": PLANNING_PROMPT.format(
                topic=self.state.topic,
                scope=self.state.clarified_scope
            )}
        ]
        
        response = call_llm(messages, temperature=0.3)
        
        # Парсим JSON с запросами
        try:
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                self.state.research_questions = json.loads(json_match.group())
        except json.JSONDecodeError:
            self.state.research_questions = [
                line.strip().strip('"-,')
                for line in response.split('\n')
                if line.strip() and not line.strip().startswith('[')
            ][:5]
        
        return f"📋 Сформированы поисковые запросы:\n" + \
               "\n".join(f"  • {q}" for q in self.state.research_questions) + \
               "\n\n💡 Проверьте запросы. Введите /ready для начала поиска, " \
               "или добавьте свои запросы текстом"
    
    def _handle_planning(self, user_input: str) -> str:
        """Обработка этапа планирования. Можно добавлять свои запросы."""
        # Команда /ready — переходим к поиску
        if user_input.strip().lower() == "/ready":
            if not self.state.research_questions:
                return "⚠️  Нет поисковых запросов. Добавьте хотя бы один запрос текстом."
            
            self.state.phase = ResearchPhase.SEARCHING
            return f"🔍 Начинаю поиск по {len(self.state.research_questions)} запросам...\n\n" \
                   f"Нажмите Enter для продолжения"
        
        # Команда /back — возвращаемся к уточнению
        if user_input.strip().lower() == "/back":
            self.state.phase = ResearchPhase.CLARIFICATION
            self.state.research_questions = []
            return f"↩️  Вернулись к уточнению scope.\n\n" \
                   f"Текущий scope:\n{self.state.clarified_scope}\n\n" \
                   f"Добавьте уточнений или /ready для продолжения"
        
        # Добавляем пользовательский запрос
        new_query = user_input.strip()
        if new_query:
            self.state.research_questions.append(new_query)
        
        return f"✓ Добавлен запрос: {new_query}\n\n" \
               f"Текущие запросы:\n" + \
               "\n".join(f"  • {q}" for q in self.state.research_questions) + \
               f"\n\nДобавьте ещё или /ready для начала поиска"
    
    def _handle_searching(self) -> str:
        """Выполнение поиска."""
        print("\n🔍 Выполняю поиск...")
        
        all_sources = []
        for query in self.state.research_questions:
            print(f"  Ищу: {query}")
            sources = web_search(query, max_results=3)
            for s in sources:
                all_sources.append({
                    "title": s.title,
                    "url": s.url,
                    "snippet": s.snippet,
                    "query": query
                })
        
        # Дедупликация по URL
        seen_urls = set()
        unique_sources = []
        for s in all_sources:
            if s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                unique_sources.append(s)
        
        self.state.sources = unique_sources
        self.state.phase = ResearchPhase.ANALYZING
        
        return f"Найдено {len(unique_sources)} уникальных источников.\n\nАнализирую... (нажмите Enter)"
    
    def _handle_analyzing(self) -> str:
        """Анализ найденных источников."""
        print("\n🔬 Анализирую источники...")
        
        # Форматируем источники для LLM
        sources_text = "\n\n".join(
            f"[{i+1}] {s['title']}\n    URL: {s['url']}\n    {s['snippet']}"
            for i, s in enumerate(self.state.sources)
        )
        
        messages = [
            self.get_system_message(),
            {"role": "user", "content": ANALYSIS_PROMPT.format(
                topic=self.state.topic,
                scope=self.state.clarified_scope,
                sources=sources_text
            )}
        ]
        
        analysis = call_llm(messages)
        self.state.findings.append(analysis)
        
        # Проверяем, нужен ли дополнительный поиск
        if "НУЖЕН ПОИСК:" in analysis:
            # Извлекаем дополнительные запросы
            match = re.search(r'НУЖЕН ПОИСК:\s*(.+?)(?:\n|$)', analysis)
            if match:
                additional_query = match.group(1).strip()
                self.state.research_questions.append(additional_query)
                self.state.phase = ResearchPhase.SEARCHING
                return f"Требуется дополнительный поиск: {additional_query}\n\n(нажмите Enter)"
        
        self.state.phase = ResearchPhase.SYNTHESIZING
        return f"Анализ завершён.\n\n{analysis}\n\nФормирую выводы... (нажмите Enter)"
    
    def _handle_synthesizing(self) -> str:
        """Синтез финальных выводов."""
        print("\n📝 Формирую выводы...")
        
        sources_text = "\n\n".join(
            f"[{i+1}] {s['title']}: {s['snippet']}"
            for i, s in enumerate(self.state.sources)
        )
        
        findings_text = "\n\n".join(self.state.findings)
        
        messages = [
            self.get_system_message(),
            {"role": "user", "content": SYNTHESIS_PROMPT.format(
                topic=self.state.topic,
                scope=self.state.clarified_scope,
                sources=sources_text,
                findings=findings_text
            )}
        ]
        
        self.state.draft_conclusion = call_llm(messages)
        self.state.phase = ResearchPhase.REVIEW
        
        return f"""
{'='*60}
РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ
{'='*60}

Тема: {self.state.topic}
Scope: {self.state.clarified_scope}

{self.state.draft_conclusion}

{'='*60}
Источники:
{chr(10).join(f"  [{i+1}] {s['url']}" for i, s in enumerate(self.state.sources[:10]))}

{'='*60}

Что дальше?
  • Введите уточняющий вопрос для углубления
  • /done — завершить исследование
  • /export — сохранить в файл
"""
    
    def _handle_review(self, user_input: str) -> str:
        """Обработка фидбека после результатов."""
        cmd = user_input.strip().lower()
        
        if cmd == "/done":
            self.state.phase = ResearchPhase.DONE
            return "✅ Исследование завершено!"
        
        if cmd == "/export":
            self._export_results()
            return "📄 Результаты сохранены в research_output.md"
        
        if cmd == "/back":
            # Возвращаемся к синтезу для переформулировки выводов
            self.state.phase = ResearchPhase.SYNTHESIZING
            self.state.draft_conclusion = ""
            return "↩️  Вернулись к формированию выводов.\n\n" \
                   "Нажмите Enter для повторного синтеза или /back для возврата к анализу"
        
        # Углубление по запросу — возвращаемся в планирование
        self.state.topic = f"{self.state.topic} — {user_input}"
        self.state.phase = ResearchPhase.PLANNING
        
        messages = [
            self.get_system_message(),
            {"role": "user", "content": PLANNING_PROMPT.format(
                topic=self.state.topic,
                scope=f"{self.state.clarified_scope}. Дополнительный фокус: {user_input}"
            )}
        ]
        
        response = call_llm(messages, temperature=0.3)
        
        try:
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                new_questions = json.loads(json_match.group())
                self.state.research_questions.extend(new_questions)
        except json.JSONDecodeError:
            pass
        
        return f"📋 Углубляю исследование: {user_input}\n\n" \
               f"Новые запросы:\n" + \
               "\n".join(f"  • {q}" for q in self.state.research_questions[-3:]) + \
               f"\n\n/ready для начала поиска или добавьте свои запросы"
    
    def _export_results(self) -> None:
        """Экспорт результатов в Markdown."""
        output = f"""# Исследование: {self.state.topic}

**Scope:** {self.state.clarified_scope}

**Дата:** {self.state.updated_at}

---

## Результаты

{self.state.draft_conclusion}

---

## Источники

"""
        for i, s in enumerate(self.state.sources, 1):
            output += f"{i}. [{s['title']}]({s['url']})\n"
        
        output += f"""
---

## Поисковые запросы

"""
        for q in self.state.research_questions:
            output += f"- {q}\n"
        
        with open("research_output.md", "w", encoding="utf-8") as f:
            f.write(output)


# ============================================================================
# CLI
# ============================================================================

def print_status(state: ResearchState) -> None:
    """Показать текущий статус."""
    phase_icons = {
        ResearchPhase.IDLE: "⏸️",
        ResearchPhase.CLARIFICATION: "❓",
        ResearchPhase.PLANNING: "📋",
        ResearchPhase.SEARCHING: "🔍",
        ResearchPhase.ANALYZING: "🔬",
        ResearchPhase.SYNTHESIZING: "📝",
        ResearchPhase.REVIEW: "✅",
        ResearchPhase.DONE: "🏁"
    }
    
    icon = phase_icons.get(state.phase, "•")
    print(f"\n{icon} Состояние: {state.phase.value}")
    if state.topic:
        print(f"   Тема: {state.topic}")
    if state.sources:
        print(f"   Источников: {len(state.sources)}")


def main():
    """Главная функция CLI."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🔬 Research Agent v1.0                          ║
║                                                              ║
║  Команды:                                                    ║
║    /new    — начать новое исследование                       ║
║    /ready  — подтвердить переход к следующему этапу          ║
║    /back   — вернуться к предыдущему этапу                   ║
║    /status — показать текущее состояние                      ║
║    /clear  — сбросить состояние                              ║
║    /quit   — выйти (состояние сохранится)                    ║
║                                                              ║
║  Для продолжения просто введите текст или нажмите Enter      ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Проверяем API ключ
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY не установлен!")
        print("   Установите переменную окружения:")
        print("   export OPENAI_API_KEY='sk-...'")
        print()
    
    # Загружаем состояние
    state = load_state()
    if state and state.phase != ResearchPhase.IDLE:
        print(f"📂 Загружено сохранённое исследование: {state.topic}")
        print_status(state)
        agent = ResearchAgent(state)
        
        # Подсказка в зависимости от фазы
        if state.phase == ResearchPhase.SEARCHING:
            print(f"\n👉 Нажмите Enter для выполнения поиска")
        elif state.phase in (ResearchPhase.ANALYZING, ResearchPhase.SYNTHESIZING):
            print(f"\n👉 Нажмите Enter для продолжения или /back для возврата")
        elif state.phase == ResearchPhase.CLARIFICATION:
            print(f"\n👉 Ответьте на вопросы, затем /ready для продолжения (или /back)")
        elif state.phase == ResearchPhase.PLANNING:
            print(f"\n👉 Добавьте запросы или /ready для поиска (или /back)")
        elif state.phase == ResearchPhase.REVIEW:
            print(f"\n👉 Введите вопрос для углубления, /done, /export или /back")
        else:
            print(f"\n👉 Введите текст для продолжения или /new для нового исследования")
    else:
        agent = ResearchAgent()
        print("Введите тему для исследования или /new для начала:")
    
    # Главный цикл
    while True:
        try:
            user_input = input("\n> ").strip()
        except KeyboardInterrupt:
            # Ctrl+C — выходим чисто, без traceback
            print("\n\n💾 Состояние сохранено. До свидания!")
            break
        except EOFError:
            # Конец ввода (pipe, редирект)
            print("\n\n💾 Состояние сохранено. До свидания!")
            break
        
        # Команды
        if user_input.lower() == "/quit":
            print("💾 Состояние сохранено. До свидания!")
            break
        
        if user_input.lower() == "/clear":
            clear_state()
            agent = ResearchAgent()
            print("🗑️  Состояние сброшено. Введите тему:")
            continue
        
        if user_input.lower() == "/status":
            print_status(agent.state)
            continue
        
        if user_input.lower() == "/new":
            clear_state()
            agent = ResearchAgent()
            print("📝 Новое исследование. Введите тему:")
            continue
        
        # Пустой ввод — продолжаем автоматические фазы
        if not user_input and agent.state.phase in (
            ResearchPhase.SEARCHING,
            ResearchPhase.ANALYZING,
            ResearchPhase.SYNTHESIZING
        ):
            pass  # Продолжаем без ввода
        elif not user_input:
            continue
        
        # Обработка через агента
        try:
            response = agent.process_input(user_input or "")
            print(f"\n{response}")
        except KeyboardInterrupt:
            # Ctrl+C во время обработки — выходим чисто
            print("\n\n⚠️  Прервано. Состояние сохранено.")
            save_state(agent.state)
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ловим Ctrl+C на верхнем уровне
        print("\n\n💾 До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        raise SystemExit(1)
