"""
Управление памятью задачи (task state).
Хранит цель диалога, уточнения и ограничения, накопленные по ходу беседы.
"""

import json
import re

from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


class TaskMemory:
    """
    Хранит и обновляет task_state: цель, уточнения, ограничения.
    Обновление происходит через LLM после каждого сообщения пользователя.
    """

    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

        self.state: dict = {
            "goal": None,
            "clarified": [],       # Что уже выяснили/уточнили
            "constraints": [],     # Ограничения и зафиксированные термины
            "open_questions": [],  # Что ещё не прояснено
        }

    def update(self, user_message: str, assistant_response: str) -> None:
        """Обновить task_state на основе нового сообщения и ответа ассистента."""
        prompt = f"""Ты анализируешь диалог и обновляешь структуру задачи.


Текущее состояние задачи:
{json.dumps(self.state, ensure_ascii=False, indent=2)}

Новое сообщение пользователя:
{user_message}

Ответ ассистента:
{assistant_response}

Верни ТОЛЬКО JSON (без пояснений, без ```json) с обновлённым состоянием задачи:
{{
  "goal": "цель диалога одним предложением или null если не ясна",
  "clarified": ["список фактов, которые пользователь уточнил или подтвердил"],
  "constraints": ["ограничения, требования, зафиксированные термины"],
  "open_questions": ["вопросы, которые ещё не прояснены"]
}}

Правила:
- goal: формулируй кратко, сохраняй из предыдущего состояния если не изменилась
- clarified и constraints: только накапливай, не удаляй старые пункты
- open_questions: удаляй если уже прояснили, добавляй новые
- Если ничего не изменилось — верни текущее состояние без изменений"""

        try:
            data = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 500,
            }).encode()

            req = Request(
                f"{self.api_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            raw = result["choices"][0]["message"]["content"].strip()
            # Убираем ```json ... ``` если LLM всё же добавил
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            new_state = json.loads(raw)
            # Валидируем ключи
            for key in ("goal", "clarified", "constraints", "open_questions"):

                if key in new_state:
                    self.state[key] = new_state[key]

        except (URLError, json.JSONDecodeError, KeyError):

            # Не критично — просто оставляем предыдущее состояние
            pass

    def build_rag_query(self, user_message: str) -> str:
        """
        Формирует расширенный запрос для RAG.
        Объединяет цель + ограничения + текущее сообщение.
        """
        parts = []

        if self.state["goal"]:
            parts.append(self.state["goal"])
        if self.state["constraints"]:
            parts.append(" ".join(self.state["constraints"]))
        parts.append(user_message)
        return " ".join(parts)

    def format_for_prompt(self) -> str:
        """Форматирует task_state для вставки в системный промпт."""

        if not self.state["goal"] and not self.state["clarified"]:
            return ""

        lines = ["=== Память задачи ==="]
        if self.state["goal"]:
            lines.append(f"Цель: {self.state['goal']}")
        if self.state["clarified"]:
            lines.append("Уточнено:")
            lines.extend(f"  • {item}" for item in self.state["clarified"])
        if self.state["constraints"]:
            lines.append("Ограничения/термины:")
            lines.extend(f"  • {item}" for item in self.state["constraints"])
        if self.state["open_questions"]:
            lines.append("Открытые вопросы:")
            lines.extend(f"  • {item}" for item in self.state["open_questions"])
        lines.append("====================")
        return "\n".join(lines)

    def summary(self) -> str:
        """Краткая сводка для вывода пользователю."""
        goal = self.state["goal"] or "не определена"
        n_facts = len(self.state["clarified"]) + len(self.state["constraints"])
        return f"Цель: {goal} | Фактов накоплено: {n_facts}"
