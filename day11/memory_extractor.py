"""
Экстрактор памяти — извлекает структурированные данные из диалога.

Использует LLM для анализа сообщений и обновления рабочей/долговременной памяти.
Один вызов LLM -> распределение по двум слоям (рабочая и долговременная).
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple


EXTRACTION_SYSTEM_PROMPT = """Ты — модуль извлечения данных из диалога.

Твоя задача: проанализировать последние сообщения диалога и извлечь:
1. Данные для РАБОЧЕЙ памяти (текущая задача сбора реквизитов)
2. Данные для ДОЛГОВРЕМЕННОЙ памяти (факты о пользователе, предпочтения)

КОНТЕКСТ ЗАДАЧИ:
Агент собирает реквизиты компании. Нужные поля:
- full_legal_name (полное название юрлица)
- inn (ИНН)
- ogrn_or_ogrnip (ОГРН или ОГРНИП)
- legal_address (юридический адрес)
- signatory (подписант: name + position)
- bank_details (bank_name, bik, account_number, correspondent_account)

ТЕКУЩЕЕ СОСТОЯНИЕ РАБОЧЕЙ ПАМЯТИ:
{working_memory}

АНАЛИЗИРУЙ последнее сообщение пользователя и ответ ассистента.

ВЕРНИ СТРОГО JSON (без markdown, без пояснений):
{{
    "working_memory_updates": {{
        "collected_data": {{
            // только НОВЫЕ данные, которые пользователь сообщил
            // например: "inn": "7707083893"
            // НЕ включай поля, которые уже есть в текущем состоянии
        }},
        "missing_fields": [
            // список полей, которые ЕЩЁ нужно собрать (обнови на основе collected_data)
        ],
        "current_status": "in_progress | completed | blocked",
        "current_question_context": "что сейчас уточняем",
        "blockers": []
    }},
    "long_term_updates": {{
        "user_profile": {{
            // если пользователь сообщил что-то о себе (имя, роль, компания)
        }},
        "preferences": {{
            // если понятны предпочтения (краткость, формат)
        }},
        "new_fact": null,  // или строка с важным фактом
        "new_pattern": null,  // или строка с паттерном поведения
        "save_long_term": false  // true только если есть что-то для долговременной памяти
    }},
    "task_completed": false  // true если ВСЕ реквизиты собраны
}}

ВАЖНО:
- В collected_data включай ТОЛЬКО поля, значения которых ЯВНО сообщил пользователь в последнем сообщении
- Не выдумывай значения
- signatory должен быть объектом {{"name": "...", "position": "..."}}
- bank_details должен быть объектом с 4 полями
- save_long_term = true только если узнали что-то важное о пользователе (не о компании)
"""


class MemoryExtractor:
    """
    Извлекает структурированные данные из диалога через LLM.
    """

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def extract(
        self,
        recent_messages: List[Dict[str, str]],
        current_working_memory: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Извлечь данные из последних сообщений.

        Args:
            recent_messages: последние сообщения диалога
            current_working_memory: текущее состояние рабочей памяти

        Returns:
            (working_updates, long_term_updates, task_completed)
        """
        if not recent_messages:
            return {}, {}, False

        # Формируем промпт с текущим состоянием
        system = EXTRACTION_SYSTEM_PROMPT.format(
            working_memory=json.dumps(current_working_memory, ensure_ascii=False, indent=2)
        )

        # Берём последние 4 сообщения для анализа
        msgs_to_analyze = recent_messages[-4:]
        user_content = self._format_messages_for_analysis(msgs_to_analyze)

        try:
            response_text = self._call_llm(system, user_content)
            parsed = self._parse_response(response_text)

            working_updates = parsed.get("working_memory_updates", {})
            long_term_updates = parsed.get("long_term_updates", {})
            task_completed = parsed.get("task_completed", False)

            # Если save_long_term = false, очищаем long_term_updates
            if not long_term_updates.get("save_long_term", False):
                long_term_updates = {}

            return working_updates, long_term_updates, task_completed

        except Exception as e:
            print(f"[MemoryExtractor] Ошибка извлечения: {e}")
            return {}, {}, False

    def _format_messages_for_analysis(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", "")
            prefix = "USER" if role == "user" else "ASSISTANT"
            lines.append(f"{prefix}: {text}")
        return "\n\n".join(lines)

    def _call_llm(self, system: str, user_content: str) -> str:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_content}]}
            ],
            "max_output_tokens": 1024
        }

        json_data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json_data,
            method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=30) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
            return self._extract_text(resp_json)

    def _extract_text(self, resp: Dict[str, Any]) -> str:
        out = resp.get("output", [])
        texts = []
        for item in out:
            content = item.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    texts.append(block["text"])
        return "\n".join(texts).strip()

    def _parse_response(self, text: str) -> Dict[str, Any]:
        # Убираем возможные markdown-обёртки
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Убираем первую и последнюю строки с ```
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
