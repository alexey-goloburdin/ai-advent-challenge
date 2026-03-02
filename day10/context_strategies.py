# context_strategies.py
"""
Три стратегии управления контекстом:
1. SlidingWindowStrategy — храним только последние N сообщений
2. StickyFactsStrategy — facts (key-value) + последние N сообщений  
3. BranchingStrategy — checkpoint + независимые ветки диалога
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy


# ============================================================================
# Base Strategy Interface
# ============================================================================

@dataclass
class Message:
    """Единое представление сообщения."""
    role: str  # "user" | "assistant"
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Message":
        return cls(role=d["role"], text=d["text"])


class ContextStrategy(ABC):
    """Базовый интерфейс для стратегий управления контекстом."""

    @abstractmethod
    def add_message(self, msg: Message) -> None:
        """Добавить сообщение в историю."""
        pass

    @abstractmethod
    def get_context_messages(self) -> List[Message]:
        """Получить сообщения для отправки в LLM."""
        pass

    @abstractmethod
    def get_all_messages(self) -> List[Message]:
        """Получить ВСЮ историю (для отображения)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Сохранить состояние в файл."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Загрузить состояние из файла."""
        pass

    @abstractmethod
    def get_strategy_info(self) -> str:
        """Информация о текущем состоянии стратегии (для отладки)."""
        pass

    def get_extra_system_content(self) -> Optional[str]:
        """Дополнительный контент для system prompt (например, facts)."""
        return None


# ============================================================================
# Strategy 1: Sliding Window
# ============================================================================

class SlidingWindowStrategy(ContextStrategy):
    """
    Простейшая стратегия: храним только последние N сообщений.
    Всё остальное безвозвратно теряется.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = max(2, window_size)
        self._messages: List[Message] = []
        self._full_history: List[Message] = []  # для статистики

    def add_message(self, msg: Message) -> None:
        self._full_history.append(msg)
        self._messages.append(msg)
        # Обрезаем до window_size
        if len(self._messages) > self.window_size:
            self._messages = self._messages[-self.window_size:]

    def get_context_messages(self) -> List[Message]:
        return list(self._messages)

    def get_all_messages(self) -> List[Message]:
        return list(self._full_history)

    def save(self, path: Path) -> None:
        data = {
            "strategy": "sliding_window",
            "window_size": self.window_size,
            "messages": [m.to_dict() for m in self._messages],
            "full_history": [m.to_dict() for m in self._full_history],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("strategy") != "sliding_window":
                return
            self._messages = [Message.from_dict(m) for m in data.get("messages", [])]
            self._full_history = [Message.from_dict(m) for m in data.get("full_history", [])]
        except Exception:
            pass

    def get_strategy_info(self) -> str:
        total = len(self._full_history)
        in_window = len(self._messages)
        lost = total - in_window
        return (
            f"[Sliding Window] window={self.window_size}, "
            f"в окне: {in_window}, всего было: {total}, потеряно: {lost}"
        )


# ============================================================================
# Strategy 2: Sticky Facts (Key-Value Memory)
# ============================================================================

# Поля реквизитов, которые мы хотим извлекать
REQUISITE_FIELDS = [
    "full_legal_name",      # полное название юрлица
    "inn",                  # ИНН
    "ogrn",                 # ОГРН/ОГРНИП
    "legal_address",        # юридический адрес
    "signatory_name",       # ФИО подписанта
    "signatory_position",   # должность подписанта
    "bank_name",            # наименование банка
    "bik",                  # БИК
    "account_number",       # расчётный счёт
    "correspondent_account" # корр. счёт
]


class StickyFactsStrategy(ContextStrategy):
    """
    Стратегия с извлечением фактов:
    - Отдельный блок facts (key-value) с важными данными
    - facts обновляются после каждого сообщения пользователя
    - В контекст идёт: facts + последние N сообщений
    """

    def __init__(self, window_size: int = 6, llm_caller: Optional[Any] = None):
        self.window_size = max(2, window_size)
        self._messages: List[Message] = []
        self._full_history: List[Message] = []
        self._facts: Dict[str, str] = {}
        self._llm_caller = llm_caller  # функция для вызова LLM (извлечение фактов)
        
        # Статистика токенов на извлечение фактов
        self.facts_extraction_tokens: Tuple[int, int] = (0, 0)

    def set_llm_caller(self, caller: Any) -> None:
        """Установить функцию для вызова LLM."""
        self._llm_caller = caller

    def add_message(self, msg: Message) -> None:
        self._full_history.append(msg)
        self._messages.append(msg)

        # После сообщения пользователя — извлекаем/обновляем факты
        if msg.role == "user" and self._llm_caller:
            self._extract_facts_from_dialogue()

        # Обрезаем окно
        if len(self._messages) > self.window_size:
            self._messages = self._messages[-self.window_size:]

    def _extract_facts_from_dialogue(self) -> None:
        """Извлекает факты из последних сообщений через LLM."""
        if not self._llm_caller:
            return

        # Берём последние сообщения для анализа
        recent = self._messages[-4:] if len(self._messages) >= 4 else self._messages
        dialogue = "\n".join([f"{m.role}: {m.text}" for m in recent])

        current_facts_str = json.dumps(self._facts, ensure_ascii=False) if self._facts else "{}"

        extraction_prompt = f"""Проанализируй диалог и извлеки реквизиты компании.

Текущие известные факты:
{current_facts_str}

Последние сообщения диалога:
{dialogue}

Верни JSON с обновлёнными фактами. Используй ТОЛЬКО эти ключи (если значение известно):
- full_legal_name: полное название юрлица
- inn: ИНН (10 или 12 цифр)
- ogrn: ОГРН или ОГРНИП
- legal_address: юридический адрес
- signatory_name: ФИО подписанта
- signatory_position: должность подписанта
- bank_name: название банка
- bik: БИК банка
- account_number: расчётный счёт
- correspondent_account: корреспондентский счёт

Правила:
- Сохраняй ранее известные значения, если они не были явно изменены
- Добавляй новые значения, если они появились в диалоге
- Если значение неизвестно — НЕ включай ключ
- Верни ТОЛЬКО валидный JSON, без markdown и пояснений
"""

        try:
            result, p_tok, c_tok = self._llm_caller(extraction_prompt)
            self.facts_extraction_tokens = (
                self.facts_extraction_tokens[0] + p_tok,
                self.facts_extraction_tokens[1] + c_tok
            )
            
            # Парсим JSON
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            
            new_facts = json.loads(result)
            if isinstance(new_facts, dict):
                # Мержим с существующими
                for k, v in new_facts.items():
                    if k in REQUISITE_FIELDS and v and str(v).strip():
                        self._facts[k] = str(v).strip()
        except Exception:
            pass  # Не удалось извлечь — не страшно

    def get_facts(self) -> Dict[str, str]:
        """Получить текущие факты."""
        return dict(self._facts)

    def set_facts(self, facts: Dict[str, str]) -> None:
        """Установить факты вручную."""
        self._facts = dict(facts)

    def get_context_messages(self) -> List[Message]:
        return list(self._messages)

    def get_all_messages(self) -> List[Message]:
        return list(self._full_history)

    def get_extra_system_content(self) -> Optional[str]:
        """Возвращает блок фактов для вставки в system prompt."""
        if not self._facts:
            return None
        
        lines = ["ИЗВЕСТНЫЕ РЕКВИЗИТЫ (из предыдущих сообщений):"]
        field_names = {
            "full_legal_name": "Название",
            "inn": "ИНН",
            "ogrn": "ОГРН/ОГРНИП",
            "legal_address": "Юр. адрес",
            "signatory_name": "Подписант (ФИО)",
            "signatory_position": "Подписант (должность)",
            "bank_name": "Банк",
            "bik": "БИК",
            "account_number": "Расчётный счёт",
            "correspondent_account": "Корр. счёт",
        }
        for key, value in self._facts.items():
            name = field_names.get(key, key)
            lines.append(f"  {name}: {value}")
        
        lines.append("\nНе спрашивай повторно то, что уже известно. Уточняй только недостающее.")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        data = {
            "strategy": "sticky_facts",
            "window_size": self.window_size,
            "facts": self._facts,
            "messages": [m.to_dict() for m in self._messages],
            "full_history": [m.to_dict() for m in self._full_history],
            "facts_extraction_tokens": list(self.facts_extraction_tokens),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("strategy") != "sticky_facts":
                return
            self._facts = data.get("facts", {})
            self._messages = [Message.from_dict(m) for m in data.get("messages", [])]
            self._full_history = [Message.from_dict(m) for m in data.get("full_history", [])]
            tokens = data.get("facts_extraction_tokens", [0, 0])
            self.facts_extraction_tokens = (tokens[0], tokens[1])
        except Exception:
            pass

    def get_strategy_info(self) -> str:
        facts_count = len(self._facts)
        in_window = len(self._messages)
        total = len(self._full_history)
        p, c = self.facts_extraction_tokens
        return (
            f"[Sticky Facts] window={self.window_size}, "
            f"facts: {facts_count}, в окне: {in_window}, всего: {total}, "
            f"токены на извлечение: {p}+{c}={p+c}"
        )


# ============================================================================
# Strategy 3: Branching (Ветки диалога)
# ============================================================================

@dataclass
class Branch:
    """Одна ветка диалога."""
    name: str
    messages: List[Message] = field(default_factory=list)
    created_from_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "messages": [m.to_dict() for m in self.messages],
            "created_from_checkpoint": self.created_from_checkpoint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Branch":
        return cls(
            name=d["name"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            created_from_checkpoint=d.get("created_from_checkpoint"),
        )


@dataclass
class Checkpoint:
    """Точка сохранения для создания веток."""
    name: str
    messages: List[Message]  # состояние на момент checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            name=d["name"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
        )


class BranchingStrategy(ContextStrategy):
    """
    Стратегия с ветвлением:
    - Можно создавать checkpoint в любой момент
    - От checkpoint можно создавать независимые ветки
    - Можно переключаться между ветками
    """

    def __init__(self, window_size: int = 20):
        self.window_size = max(2, window_size)
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._branches: Dict[str, Branch] = {"main": Branch(name="main")}
        self._current_branch: str = "main"

    def add_message(self, msg: Message) -> None:
        branch = self._branches[self._current_branch]
        branch.messages.append(msg)

    def get_context_messages(self) -> List[Message]:
        branch = self._branches[self._current_branch]
        msgs = branch.messages
        if len(msgs) > self.window_size:
            return msgs[-self.window_size:]
        return list(msgs)

    def get_all_messages(self) -> List[Message]:
        branch = self._branches[self._current_branch]
        return list(branch.messages)

    # --- Checkpoint operations ---

    def create_checkpoint(self, name: str) -> bool:
        """Создать checkpoint с текущим состоянием."""
        if name in self._checkpoints:
            return False  # уже существует
        
        branch = self._branches[self._current_branch]
        self._checkpoints[name] = Checkpoint(
            name=name,
            messages=deepcopy(branch.messages)
        )
        return True

    def list_checkpoints(self) -> List[str]:
        """Список всех checkpoint'ов."""
        return list(self._checkpoints.keys())

    def delete_checkpoint(self, name: str) -> bool:
        """Удалить checkpoint."""
        if name not in self._checkpoints:
            return False
        del self._checkpoints[name]
        return True

    # --- Branch operations ---

    def create_branch(self, branch_name: str, from_checkpoint: str) -> bool:
        """Создать новую ветку от checkpoint."""
        if branch_name in self._branches:
            return False  # ветка уже есть
        if from_checkpoint not in self._checkpoints:
            return False  # checkpoint не существует
        
        cp = self._checkpoints[from_checkpoint]
        self._branches[branch_name] = Branch(
            name=branch_name,
            messages=deepcopy(cp.messages),
            created_from_checkpoint=from_checkpoint,
        )
        return True

    def switch_branch(self, branch_name: str) -> bool:
        """Переключиться на другую ветку."""
        if branch_name not in self._branches:
            return False
        self._current_branch = branch_name
        return True

    def get_current_branch(self) -> str:
        """Текущая ветка."""
        return self._current_branch

    def list_branches(self) -> List[str]:
        """Список всех веток."""
        return list(self._branches.keys())

    def delete_branch(self, branch_name: str) -> bool:
        """Удалить ветку (нельзя удалить main или текущую)."""
        if branch_name == "main":
            return False
        if branch_name == self._current_branch:
            return False
        if branch_name not in self._branches:
            return False
        del self._branches[branch_name]
        return True

    def get_branch_info(self, branch_name: str) -> Optional[Dict[str, Any]]:
        """Информация о ветке."""
        if branch_name not in self._branches:
            return None
        branch = self._branches[branch_name]
        return {
            "name": branch.name,
            "message_count": len(branch.messages),
            "created_from": branch.created_from_checkpoint,
        }

    # --- Persistence ---

    def save(self, path: Path) -> None:
        data = {
            "strategy": "branching",
            "window_size": self.window_size,
            "current_branch": self._current_branch,
            "checkpoints": {k: v.to_dict() for k, v in self._checkpoints.items()},
            "branches": {k: v.to_dict() for k, v in self._branches.items()},
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("strategy") != "branching":
                return
            
            self._current_branch = data.get("current_branch", "main")
            
            self._checkpoints = {}
            for k, v in data.get("checkpoints", {}).items():
                self._checkpoints[k] = Checkpoint.from_dict(v)
            
            self._branches = {}
            for k, v in data.get("branches", {}).items():
                self._branches[k] = Branch.from_dict(v)
            
            # Гарантируем main
            if "main" not in self._branches:
                self._branches["main"] = Branch(name="main")
            if self._current_branch not in self._branches:
                self._current_branch = "main"
        except Exception:
            pass

    def get_strategy_info(self) -> str:
        branch = self._branches[self._current_branch]
        return (
            f"[Branching] ветка: '{self._current_branch}', "
            f"сообщений: {len(branch.messages)}, "
            f"веток всего: {len(self._branches)}, "
            f"checkpoints: {len(self._checkpoints)}"
        )


# ============================================================================
# Factory
# ============================================================================

def create_strategy(
    strategy_type: str,
    window_size: int = 10,
    llm_caller: Optional[Any] = None
) -> ContextStrategy:
    """
    Фабрика для создания стратегий.
    
    strategy_type: "sliding_window" | "sticky_facts" | "branching"
    """
    if strategy_type == "sliding_window":
        return SlidingWindowStrategy(window_size=window_size)
    elif strategy_type == "sticky_facts":
        return StickyFactsStrategy(window_size=window_size, llm_caller=llm_caller)
    elif strategy_type == "branching":
        return BranchingStrategy(window_size=window_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")
