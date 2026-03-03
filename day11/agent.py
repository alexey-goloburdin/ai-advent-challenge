"""
Агент для сбора реквизитов компании с трёхслойной моделью памяти.

Слои памяти:
1. Краткосрочная (short-term) — сырая история диалога
2. Рабочая (working) — структурированные данные текущей задачи
3. Долговременная (long-term) — профиль пользователя, накопленные знания

Каждый слой хранится в отдельном JSON-файле.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from memory_manager import MemoryManager, MemoryConfig
from memory_extractor import MemoryExtractor


JSON_STOP_SEQUENCE = "<JSON_END>"


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    extractor_model: str = "gpt-4o-mini"  # Дешёвая модель для экстрактора
    max_output_tokens: int = 1024
    json_format: bool = False

    # Memory paths
    short_term_path: str = "memory_short_term.json"
    working_path: str = "memory_working.json"
    long_term_path: str = "memory_long_term.json"

    # Сколько сообщений из краткосрочной памяти включать в контекст
    context_messages_count: int = 10


def build_system_prompt(
    json_format: bool,
    working_memory: Dict[str, Any],
    long_term_memory: Dict[str, Any]
) -> str:
    """
    Строит системный промпт с учётом рабочей и долговременной памяти.
    """
    base_prompt = (
        "Ты — агент для получения реквизитов компании.\n"
        "Веди диалог с пользователем так, чтобы собрать реквизиты и затем вернуть их ОДНИМ финальным сообщением.\n"
        "\n"
        "Нужно собрать и вернуть ТОЛЬКО следующий состав реквизитов:\n"
        "1) полное название юридического лица\n"
        "2) ИНН\n"
        "3) ОГРН или ОГРНИП (в зависимости от типа)\n"
        "4) юридический адрес\n"
        "5) подписант (ФИО + должность; кто подписывает договор)\n"
        "6) банковские реквизиты: наименование банка, БИК, расчётный счёт, корреспондентский счёт\n"
        "\n"
        "Правила:\n"
        "- Если каких-то полей не хватает, задавай уточняющие вопросы, пока не соберёшь всё.\n"
        "- Не выдумывай значения. Если пользователь не знает поле — попроси уточнить/проверить.\n"
        "- Когда все поля собраны, верни их одним финальным сообщением и НЕ задавай больше вопросов.\n"
        "- Не добавляй лишних пояснений, дисклеймеров или общих фраз.\n"
    )

    # Добавляем контекст из рабочей памяти
    if working_memory.get("collected_data"):
        base_prompt += (
            "\n"
            "=== РАБОЧАЯ ПАМЯТЬ (уже собранные данные) ===\n"
            f"{json.dumps(working_memory['collected_data'], ensure_ascii=False, indent=2)}\n"
        )

    missing = [f for f in working_memory.get("missing_fields", []) if f]
    if missing:
        base_prompt += (
            "\n"
            "=== ЕЩЁ НУЖНО СОБРАТЬ ===\n"
            f"{', '.join(missing)}\n"
        )

    blockers = [b for b in working_memory.get("blockers", []) if b]
    if blockers:
        base_prompt += (
            "\n"
            "=== БЛОКЕРЫ ===\n"
            f"{', '.join(blockers)}\n"
        )

    # Добавляем контекст из долговременной памяти
    if long_term_memory.get("user_profile"):
        base_prompt += (
            "\n"
            "=== ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ===\n"
            f"{json.dumps(long_term_memory['user_profile'], ensure_ascii=False)}\n"
        )

    if long_term_memory.get("preferences"):
        base_prompt += (
            "\n"
            "=== ПРЕДПОЧТЕНИЯ ПОЛЬЗОВАТЕЛЯ ===\n"
            f"{json.dumps(long_term_memory['preferences'], ensure_ascii=False)}\n"
        )

    patterns = [p for p in long_term_memory.get("learned_patterns", []) if p]
    if patterns:
        base_prompt += (
            "\n"
            "=== ПАТТЕРНЫ ОБЩЕНИЯ ===\n"
            f"{', '.join(patterns)}\n"
        )

    if long_term_memory.get("known_companies_count", 0) > 0:
        base_prompt += (
            "\n"
            f"Пользователь уже сохранял реквизиты {long_term_memory['known_companies_count']} компаний ранее.\n"
        )

    # JSON-формат финального ответа
    if json_format:
        base_prompt += (
            "\n"
            "ФИНАЛЬНЫЙ ОТВЕТ: строго валидный JSON без Markdown и без комментариев.\n"
            f"После JSON добавь стоп-последовательность {JSON_STOP_SEQUENCE}.\n"
            "До стоп-последовательности должен быть только JSON.\n"
            "\n"
            "JSON-ключи используй строго такие:\n"
            "{\n"
            '  "full_legal_name": "...",\n'
            '  "inn": "...",\n'
            '  "ogrn_or_ogrnip": "...",\n'
            '  "legal_address": "...",\n'
            '  "signatory": {"name": "...", "position": "..."},\n'
            '  "bank_details": {\n'
            '    "bank_name": "...",\n'
            '    "bik": "...",\n'
            '    "account_number": "...",\n'
            '    "correspondent_account": "..."\n'
            "  }\n"
            "}\n"
        )

    return base_prompt


def extract_assistant_text(resp: Dict[str, Any]) -> str:
    out = resp.get("output")
    if not isinstance(out, list) or not out:
        raise ValueError("Нет поля output в ответе или оно пустое")

    texts: List[str] = []
    for item in out:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t)

    if not texts:
        raise ValueError("Не удалось извлечь текст из output/content")

    return "\n".join(texts).strip()


class CompanyRequisitesAgent:
    """
    Агент с трёхслойной моделью памяти:
    - short_term: история диалога (memory_short_term.json)
    - working: данные текущей задачи (memory_working.json)
    - long_term: профиль пользователя (memory_long_term.json)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: AgentConfig,
    ) -> None:
        if not base_url:
            raise ValueError("base_url пустой")
        if not api_key:
            raise ValueError("api_key пустой")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.config = config

        # Инициализируем менеджер памяти
        memory_config = MemoryConfig(
            short_term_path=config.short_term_path,
            working_path=config.working_path,
            long_term_path=config.long_term_path,
            max_short_term_messages=50
        )
        self.memory = MemoryManager(memory_config)

        # Инициализируем экстрактор памяти
        self.extractor = MemoryExtractor(
            base_url=base_url,
            api_key=api_key,
            model=config.extractor_model
        )

    # ==================== Public API ====================

    def reply(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        # 1. Сохраняем сообщение пользователя в краткосрочную память
        self.memory.add_user_message(user_text)

        # 2. Строим системный промпт с учётом рабочей и долговременной памяти
        context = self.memory.get_context_for_llm()
        system_prompt = build_system_prompt(
            json_format=self.config.json_format,
            working_memory=context["working"],
            long_term_memory=context["long_term"]
        )

        # 3. Готовим сообщения для LLM (системный + история из краткосрочной памяти)
        messages = self._build_messages(system_prompt)

        # 4. Вызываем основную модель
        assistant_text = self._call_llm(messages)

        # 5. Сохраняем ответ в краткосрочную память
        clean_text = self._clean_output(assistant_text)
        self.memory.add_assistant_message(clean_text)

        # 6. Извлекаем данные и обновляем рабочую/долговременную память
        self._update_memories_from_dialog()

        return clean_text

    def get_memory_state(self) -> str:
        """Вернуть текущее состояние всех слоёв памяти (для отладки)."""
        return self.memory.debug_dump()

    def get_short_term_history(self) -> List[Dict[str, str]]:
        """Вернуть историю из краткосрочной памяти."""
        return self.memory.get_full_history()

    def reset_task(self) -> None:
        """Сбросить рабочую память (начать новую задачу)."""
        self.memory.reset_working_memory()

    def new_session(self) -> None:
        """Начать новую сессию (очистить краткосрочную память)."""
        self.memory.clear_short_term()

    # ==================== Private ====================

    def _build_messages(self, system_prompt: str) -> List[Dict[str, Any]]:
        """Собрать сообщения для LLM из системного промпта и краткосрочной памяти."""
        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}
        ]

        # Берём последние N сообщений из краткосрочной памяти
        recent = self.memory.short_term.get_last_n(self.config.context_messages_count)

        for msg in recent:
            role = msg["role"]
            text = msg["text"]
            if role == "user":
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                })
            elif role == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]
                })

        return messages

    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Вызов основной модели."""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": messages,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.json_format:
            payload["stop"] = [JSON_STOP_SEQUENCE]

        json_data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json_data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                return extract_assistant_text(resp_json)

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise SystemExit(f"HTTP-ошибка {e.code}: {body or '(без тела ответа)'}") from None
        except urllib.error.URLError as e:
            raise SystemExit(f"Ошибка сети: {e.reason}") from None
        except json.JSONDecodeError:
            raise SystemExit("Не удалось распарсить JSON-ответ от LLM") from None
        except Exception as e:
            raise SystemExit(f"Ошибка при отправке запроса к LLM: {type(e).__name__}: {e}") from None

    def _update_memories_from_dialog(self) -> None:
        """
        Извлечь данные из последних сообщений и обновить рабочую/долговременную память.
        Это отдельный вызов LLM (экстрактор).
        """
        recent = self.memory.short_term.get_last_n(4)
        if not recent:
            return

        context = self.memory.get_context_for_llm()

        working_updates, long_term_updates, task_completed = self.extractor.extract(
            recent_messages=recent,
            current_working_memory=context["working"]
        )

        # Обновляем рабочую память
        if working_updates:
            if task_completed:
                working_updates["task_completed"] = True
            self.memory.update_working_memory(working_updates)

        # Обновляем долговременную память (если есть что)
        if long_term_updates:
            # Если задача завершена, сохраняем собранные реквизиты в долговременную память
            if task_completed and context["working"].get("collected_data"):
                long_term_updates["completed_company"] = context["working"]["collected_data"]
            self.memory.update_long_term(long_term_updates)

    def _clean_output(self, text: str) -> str:
        """Очистить вывод от стоп-последовательности."""
        if self.config.json_format and JSON_STOP_SEQUENCE in text:
            return text.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()
        return text
