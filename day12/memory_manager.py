"""
Модуль управления памятью агента с тремя слоями:

1. Краткосрочная (short-term) — история текущего диалога (сырые сообщения)
2. Рабочая (working) — структурированные данные текущей задачи
3. Долговременная (long-term) — профиль пользователя, решения, накопленные знания

Каждый слой хранится в отдельном JSON-файле.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class MemoryConfig:
    """Конфигурация путей к файлам памяти."""
    short_term_path: str = "memory_short_term.json"
    working_path: str = "memory_working.json"
    long_term_path: str = "memory_long_term.json"
    max_short_term_messages: int = 50


@dataclass
class ShortTermMemory:
    """
    Краткосрочная память — сырая история диалога.
    Просто список сообщений без обработки.
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    session_started: Optional[str] = None

    def add_message(self, role: str, text: str) -> None:
        if role not in ("user", "assistant"):
            return
        if not text or not text.strip():
            return
        self.messages.append({
            "role": role,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat()
        })

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        return self.messages[-n:] if n > 0 else []

    def clear(self) -> None:
        self.messages = []
        self.session_started = datetime.now().isoformat()


@dataclass
class WorkingMemory:
    """
    Рабочая память — структурированные данные текущей задачи.
    Для агента сбора реквизитов: какие поля уже собраны, какие нет.
    """
    task_type: str = "collect_requisites"
    task_started: Optional[str] = None
    task_completed: bool = False

    # Собранные данные (что уже известно)
    collected_data: Dict[str, Any] = field(default_factory=dict)

    # Какие поля ещё нужно собрать
    missing_fields: List[str] = field(default_factory=list)

    # Текущий статус / блокеры
    current_status: str = ""
    blockers: List[str] = field(default_factory=list)

    # Контекст текущего вопроса (что сейчас уточняем)
    current_question_context: str = ""

    def update_collected(self, field_name: str, value: Any) -> None:
        self.collected_data[field_name] = value
        if field_name in self.missing_fields:
            self.missing_fields.remove(field_name)

    def mark_completed(self) -> None:
        self.task_completed = True
        self.current_status = "completed"

    def reset(self) -> None:
        self.task_started = datetime.now().isoformat()
        self.task_completed = False
        self.collected_data = {}
        self.missing_fields = [
            "full_legal_name",
            "inn",
            "ogrn_or_ogrnip",
            "legal_address",
            "signatory",
            "bank_details"
        ]
        self.current_status = "in_progress"
        self.blockers = []
        self.current_question_context = ""


@dataclass
class LongTermMemory:
    """
    Долговременная память — профиль пользователя, накопленные знания.
    Сохраняется между сессиями.
    """
    # Профиль пользователя
    user_profile: Dict[str, Any] = field(default_factory=dict)

    # Предпочтения (как общаться, формат ответов и т.д.)
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Известные компании пользователя (история успешно собранных реквизитов)
    known_companies: List[Dict[str, Any]] = field(default_factory=list)

    # Решения и паттерны ("пользователь предпочитает краткие ответы")
    learned_patterns: List[str] = field(default_factory=list)

    # Факты, которые стоит помнить
    facts: List[Dict[str, str]] = field(default_factory=list)

    def add_fact(self, fact: str, category: str = "general") -> None:
        self.facts.append({
            "fact": fact,
            "category": category,
            "added_at": datetime.now().isoformat()
        })

    def add_known_company(self, company_data: Dict[str, Any]) -> None:
        company_data["saved_at"] = datetime.now().isoformat()
        self.known_companies.append(company_data)

    def update_user_profile(self, key: str, value: Any) -> None:
        self.user_profile[key] = value

    def add_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = value

    def add_learned_pattern(self, pattern: str) -> None:
        if pattern not in self.learned_patterns:
            self.learned_patterns.append(pattern)


class MemoryManager:
    """
    Менеджер памяти — управляет всеми тремя слоями.
    Отвечает за загрузку, сохранение и координацию.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        self.short_term = self._load_short_term()
        self.working = self._load_working()
        self.long_term = self._load_long_term()

    # ==================== Загрузка ====================

    def _load_short_term(self) -> ShortTermMemory:
        path = Path(self.config.short_term_path)
        if not path.exists():
            mem = ShortTermMemory()
            mem.session_started = datetime.now().isoformat()
            return mem

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ShortTermMemory(
                messages=data.get("messages", []),
                session_started=data.get("session_started")
            )
        except Exception:
            mem = ShortTermMemory()
            mem.session_started = datetime.now().isoformat()
            return mem

    def _load_working(self) -> WorkingMemory:
        path = Path(self.config.working_path)
        if not path.exists():
            mem = WorkingMemory()
            mem.reset()
            return mem

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkingMemory(
                task_type=data.get("task_type", "collect_requisites"),
                task_started=data.get("task_started"),
                task_completed=data.get("task_completed", False),
                collected_data=data.get("collected_data", {}),
                missing_fields=data.get("missing_fields", []),
                current_status=data.get("current_status", ""),
                blockers=data.get("blockers", []),
                current_question_context=data.get("current_question_context", "")
            )
        except Exception:
            mem = WorkingMemory()
            mem.reset()
            return mem

    def _load_long_term(self) -> LongTermMemory:
        path = Path(self.config.long_term_path)
        if not path.exists():
            return LongTermMemory()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LongTermMemory(
                user_profile=data.get("user_profile", {}),
                preferences=data.get("preferences", {}),
                known_companies=data.get("known_companies", []),
                learned_patterns=data.get("learned_patterns", []),
                facts=data.get("facts", [])
            )
        except Exception:
            return LongTermMemory()

    # ==================== Сохранение ====================

    def _save_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        """Атомарное сохранение через временный файл."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def save_short_term(self) -> None:
        # Обрезаем до максимума перед сохранением
        max_msgs = self.config.max_short_term_messages
        if len(self.short_term.messages) > max_msgs:
            self.short_term.messages = self.short_term.messages[-max_msgs:]

        self._save_atomic(
            Path(self.config.short_term_path),
            asdict(self.short_term)
        )

    def save_working(self) -> None:
        self._save_atomic(
            Path(self.config.working_path),
            asdict(self.working)
        )

    def save_long_term(self) -> None:
        self._save_atomic(
            Path(self.config.long_term_path),
            asdict(self.long_term)
        )

    def save_all(self) -> None:
        self.save_short_term()
        self.save_working()
        self.save_long_term()

    # ==================== Публичный API ====================

    def add_user_message(self, text: str) -> None:
        """Добавить сообщение пользователя в краткосрочную память."""
        self.short_term.add_message("user", text)
        self.save_short_term()

    def add_assistant_message(self, text: str) -> None:
        """Добавить ответ ассистента в краткосрочную память."""
        self.short_term.add_message("assistant", text)
        self.save_short_term()

    def update_working_memory(self, updates: Dict[str, Any]) -> None:
        """
        Обновить рабочую память структурированными данными.
        updates может содержать:
        - collected_data: Dict[str, Any]
        - missing_fields: List[str]
        - current_status: str
        - blockers: List[str]
        - current_question_context: str
        - task_completed: bool
        """
        if "collected_data" in updates:
            for k, v in updates["collected_data"].items():
                self.working.update_collected(k, v)

        if "missing_fields" in updates:
            self.working.missing_fields = updates["missing_fields"]

        if "current_status" in updates:
            self.working.current_status = updates["current_status"]

        if "blockers" in updates:
            self.working.blockers = updates["blockers"]

        if "current_question_context" in updates:
            self.working.current_question_context = updates["current_question_context"]

        if updates.get("task_completed"):
            self.working.mark_completed()

        self.save_working()

    def update_long_term(self, updates: Dict[str, Any]) -> None:
        """
        Обновить долговременную память.
        updates может содержать:
        - user_profile: Dict[str, Any]
        - preferences: Dict[str, Any]
        - new_fact: str
        - new_pattern: str
        - completed_company: Dict[str, Any]
        """
        if "user_profile" in updates:
            for k, v in updates["user_profile"].items():
                self.long_term.update_user_profile(k, v)

        if "preferences" in updates:
            for k, v in updates["preferences"].items():
                self.long_term.add_preference(k, v)

        if "new_fact" in updates:
            self.long_term.add_fact(updates["new_fact"])

        if "new_pattern" in updates:
            self.long_term.add_learned_pattern(updates["new_pattern"])

        if "completed_company" in updates:
            self.long_term.add_known_company(updates["completed_company"])

        self.save_long_term()

    def get_context_for_llm(self) -> Dict[str, Any]:
        """
        Собрать контекст из всех слоёв памяти для передачи в LLM.
        """
        return {
            "short_term": {
                "recent_messages": self.short_term.get_last_n(10),
                "session_started": self.short_term.session_started
            },
            "working": {
                "task_type": self.working.task_type,
                "collected_data": self.working.collected_data,
                "missing_fields": self.working.missing_fields,
                "current_status": self.working.current_status,
                "blockers": self.working.blockers,
                "current_question_context": self.working.current_question_context
            },
            "long_term": {
                "user_profile": self.long_term.user_profile,
                "preferences": self.long_term.preferences,
                "known_companies_count": len(self.long_term.known_companies),
                "learned_patterns": self.long_term.learned_patterns,
                "recent_facts": self.long_term.facts[-5:] if self.long_term.facts else []
            }
        }

    def reset_working_memory(self) -> None:
        """Сбросить рабочую память (начать новую задачу)."""
        self.working.reset()
        self.save_working()

    def clear_short_term(self) -> None:
        """Очистить краткосрочную память (новая сессия)."""
        self.short_term.clear()
        self.save_short_term()

    def get_full_history(self) -> List[Dict[str, str]]:
        """Получить всю историю из краткосрочной памяти."""
        return self.short_term.messages

    def debug_dump(self) -> str:
        """Вывести состояние всех слоёв памяти для отладки."""
        return json.dumps({
            "short_term": asdict(self.short_term),
            "working": asdict(self.working),
            "long_term": asdict(self.long_term)
        }, ensure_ascii=False, indent=2)
