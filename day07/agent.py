# agent.py
import json
import os
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

JSON_STOP_SEQUENCE = "<JSON_END>"


@dataclass
class AgentConfig:
    model: str = "gpt-5.2"
    max_output_tokens: int = 1024
    json_format: bool = False


    # Context management
    max_history_turns: int = 50

    # Memory management: теперь это файл-журнал сообщений
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
    Агент со своим состоянием:

    - system prompt
    - история сообщений (в памяти процесса)
    - персистентный журнал сообщений в JSON-файле (--memory_path)
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

        self.memory_path = Path(self.config.memory_path)


        # Загружаем журнал сообщений из файла и подмешиваем в текущую историю
        persisted_msgs = self._load_persisted_messages()

        # Текущая история (short-term), но инициализируется сохранёнными сообщениями
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]}
        ]

        self._messages.extend(self._persisted_to_api_messages(persisted_msgs))

        # Подрежем сразу (если файл огромный), чтобы не улететь в контекст
        self._trim_history()

    # -------- Public helpers --------

    def get_persisted_messages(self) -> List[Dict[str, str]]:
        """Для рендера в CLI: возвращает сохранённые сообщения (role/text) из memory.json."""
        return self._load_persisted_messages()

    # -------- Public API --------

    def reply(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""


        self._append_user(user_text)
        self._append_persisted_message({"role": "user", "text": user_text})
        self._trim_history()

        messages = list(self._messages)

        assistant_text = self._call_openai_responses(messages)

        # В контекст кладём сырой текст (как вернула модель)
        self._append_assistant(assistant_text)

        # А в журнал сохраняем очищенную версию
        clean_text = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in clean_text:
            clean_text = clean_text.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()

        self._append_persisted_message({"role": "assistant", "text": clean_text})
        self._trim_history()

        return self._postprocess_output_for_display(assistant_text)

    # -------- Context management --------


    def _append_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": [{"type": "input_text", "text": text}]})

    def _append_assistant(self, text: str) -> None:

        self._messages.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})

    def _trim_history(self) -> None:
        """
        Оставляем:

        - 1 системное сообщение
        - + максимум N ходов user+assistant
        """
        max_turns = max(1, int(self.config.max_history_turns))
        max_msgs = 1 + max_turns * 2
        if len(self._messages) > max_msgs:
            system_msg = self._messages[0]
            tail = self._messages[-(max_turns * 2):]
            self._messages = [system_msg] + tail

    # -------- Persisted memory (JSON journal) --------

    def _load_persisted_messages(self) -> List[Dict[str, str]]:
        """
        Читает memory.json и возвращает список {"role": "...", "text": "..."}.
        Если файла нет или он битый — возвращает [].
        """
        if not self.memory_path.exists():
            return []

        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
            msgs = raw.get("messages", [])

            if not isinstance(msgs, list):
                return []
            out: List[Dict[str, str]] = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                text = m.get("text")
                if role in ("user", "assistant") and isinstance(text, str) and text.strip():
                    out.append({"role": role, "text": text})
            return out

        except Exception:
            return []

    def _save_persisted_messages(self, msgs: List[Dict[str, str]]) -> None:
        """
        Сохраняет журнал атомарно (через .tmp -> replace).
        """
        data = {"messages": msgs}
        tmp = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.memory_path)

    def _append_persisted_message(self, msg: Dict[str, str]) -> None:
        """
        Добавляет 1 сообщение в memory.json.
        """
        role = msg.get("role")
        text = msg.get("text")
        if role not in ("user", "assistant"):
            return
        if not isinstance(text, str) or not text.strip():
            return

        msgs = self._load_persisted_messages()
        msgs.append({"role": role, "text": text})
        self._save_persisted_messages(msgs)

    def _persisted_to_api_messages(self, msgs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """

        Переводит persisted-журнал в формат Responses API input.
        """
        api_msgs: List[Dict[str, Any]] = []
        for m in msgs:
            role = m["role"]
            text = m["text"]
            if role == "user":

                api_msgs.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
            elif role == "assistant":
                api_msgs.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
        return api_msgs

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

            raise SystemExit(f"Ошибка при отправке запроса к LLM: {type(e).__name__}: {e}") from None

    # -------- Output helpers --------

    def _postprocess_output_for_display(self, assistant_text: str) -> str:
        to_print = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in to_print:
            to_print = to_print.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()
        return to_print
