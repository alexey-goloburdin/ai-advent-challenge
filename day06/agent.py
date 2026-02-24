import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

JSON_STOP_SEQUENCE = "<JSON_END>"


@dataclass
class AgentConfig:
    model: str = "gpt-5.2"
    max_output_tokens: int = 1024
    json_format: bool = False

    # Context management
    # Сколько последних "ходов" хранить в контексте (1 ход = user+assistant)
    max_history_turns: int = 50

    # Memory management (очень простая long-term память)
    memory_path: str = "memory.json"


def build_system_prompt(json_format: bool) -> str:
    prompt = (
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

    if json_format:
        prompt += (
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

    return prompt


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
    Агент = отдельная сущность со своим состоянием:
    - short-term context: history сообщений
    - system prompt (правила)
    - long-term memory: JSON-файл (очень простой пример)
    - инкапсулированный вызов LLM (Responses API через HTTP-клиент)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: AgentConfig,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url пустой")
        if not api_key:
            raise ValueError("api_key пустой")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.config = config

        self.system_prompt = system_prompt or build_system_prompt(config.json_format)

        # Состояние диалога (short-term memory)
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]}
        ]

        # Long-term memory
        self.memory_path = Path(self.config.memory_path)
        self.memory: Dict[str, Any] = self._load_memory()

    # -------- Public API --------

    def reply(self, user_text: str) -> str:
        """
        Один агентный шаг:
        1) принять user_text
        2) обновить контекст (и подмешать память)
        3) вызвать LLM
        4) обновить контекст результатом
        5) (опционально) обновить память
        6) вернуть текст ответа
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        self._append_user(user_text)
        self._trim_history()

        # Собираем input: system + memory + history
        messages = self._build_messages_with_memory()

        assistant_text = self._call_openai_responses(messages)

        # Запишем в историю именно то, что вернула модель (включая стоп, если он есть)
        self._append_assistant(assistant_text)

        return self._postprocess_output_for_display(assistant_text)

    # -------- Context management --------

    def _append_user(self, text: str) -> None:
        self._messages.append(
            {"role": "user", "content": [{"type": "input_text", "text": text}]}
        )

    def _append_assistant(self, text: str) -> None:
        self._messages.append(
            {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
        )

    def _trim_history(self) -> None:
        """
        Оставляем:
        - 1 системное сообщение
        - + максимум N ходов user+assistant
        """
        max_turns = max(1, int(self.config.max_history_turns))
        max_msgs = 1 + max_turns * 2  # system + (user+assistant)*N
        if len(self._messages) > max_msgs:
            system_msg = self._messages[0]
            tail = self._messages[-(max_turns * 2):]
            self._messages = [system_msg] + tail

    # -------- Memory management --------

    def _load_memory(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                return json.loads(self.memory_path.read_text(encoding="utf-8"))
            except Exception:
                # если файл битый — начинаем с чистого
                return {}
        return {}

    def _save_memory(self) -> None:
        self.memory_path.write_text(
            json.dumps(self.memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _render_memory(self) -> str:
        if not self.memory:
            return ""
        lines: List[str] = []
        for k, v in self.memory.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _build_messages_with_memory(self) -> List[Dict[str, Any]]:
        """
        Подмешиваем память отдельным developer-сообщением между system и history.
        Это демонстрирует memory management и отделение "памяти" от истории.
        """
        if not self.memory:
            return list(self._messages)

        system = self._messages[0]
        history = self._messages[1:]

        mem_text = self._render_memory()
        mem_msg = {
            "role": "developer",
            "content": [{"type": "input_text", "text": f"Память о пользователе/контексте:\n{mem_text}"}],
        }
        return [system, mem_msg, *history]

    # -------- LLM call (encapsulated) --------

    def _call_openai_responses(self, messages: List[Dict[str, Any]]) -> str:
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
            raise SystemExit(
                f"Ошибка при отправке запроса к LLM: {type(e).__name__}: {e}"
            ) from None

    # -------- Output helpers --------

    def _postprocess_output_for_display(self, assistant_text: str) -> str:
        """
        Для CLI-вывода:
        - если json_format и модель вернула стоп — отрежем при печати
        """
        to_print = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in to_print:
            to_print = to_print.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()
        return to_print
