import json
import os
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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



def _extract_usage_tokens(resp_json: Dict[str, Any]) -> Tuple[int, int]:
    """
    Возвращает (prompt_tokens, completion_tokens) для текущего вызова.
    Поддерживает разные варианты названий полей usage.
    Если usage нет — вернёт (0, 0).
    """
    usage = resp_json.get("usage")

    if not isinstance(usage, dict):
        return (0, 0)

    # На практике встречаются разные ключи
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")

    if prompt is None:
        prompt = usage.get("prompt_tokens")
    if completion is None:
        completion = usage.get("completion_tokens")

    try:
        prompt_i = int(prompt) if prompt is not None else 0
    except Exception:
        prompt_i = 0

    try:
        completion_i = int(completion) if completion is not None else 0

    except Exception:
        completion_i = 0

    return (max(0, prompt_i), max(0, completion_i))


class CompanyRequisitesAgent:
    """
    Агент со своим состоянием:


    - system prompt
    - история сообщений (в памяти процесса)
    - персистентный журнал сообщений в JSON-файле (--memory_path)
    - накопительные usage_totals в memory.json
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

        # last token stats: (prompt_tokens, completion_tokens, history_total_tokens)
        self._last_token_stats: Optional[Tuple[int, int, int]] = None


        # Загружаем журнал сообщений + totals из файла
        mem = self._load_memory()
        persisted_msgs = mem["messages"]

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
        return self._load_memory()["messages"]

    def get_last_token_stats(self) -> Optional[Tuple[int, int, int]]:
        """
        Возвращает (prompt_tokens_current, completion_tokens_current, history_total_tokens).
        Может быть None, если usage не пришёл.
        """
        return self._last_token_stats

    # -------- Public API --------

    def reply(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        self._append_user(user_text)
        self._append_persisted_message({"role": "user", "text": user_text})
        self._trim_history()


        messages = list(self._messages)


        assistant_text, prompt_tokens, completion_tokens = self._call_openai_responses(messages)

        # В контекст кладём сырой текст (как вернула модель)

        self._append_assistant(assistant_text)

        # А в журнал сохраняем очищенную версию
        clean_text = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in clean_text:
            clean_text = clean_text.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()

        self._append_persisted_message({"role": "assistant", "text": clean_text})
        self._trim_history()


        # Обновим totals и last stats
        self._update_usage_totals(prompt_tokens, completion_tokens)

        # "Сколько влезает в контекстное окно" — это input_tokens текущего запроса
        context_tokens_now = prompt_tokens

        # Оценка "диалог после ответа" (полезно понимать, что уйдёт в историю)
        dialog_tokens_after_reply_est = prompt_tokens + completion_tokens

        after_est = prompt_tokens + completion_tokens
        self._last_token_stats = (prompt_tokens, completion_tokens, after_est)

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

    def _load_memory(self) -> Dict[str, Any]:
        """
        Читает memory.json и возвращает структуру:
        {
          "messages": [ {"role": "...", "text": "..."}, ... ],
          "usage_totals": {"prompt_tokens": int, "completion_tokens": int}
        }
        Если файла нет/битый — вернёт дефолт.
        """
        default = {"messages": [], "usage_totals": {"prompt_tokens": 0, "completion_tokens": 0}}

        if not self.memory_path.exists():
            return default

        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default

            msgs_raw = raw.get("messages", [])
            msgs: List[Dict[str, str]] = []
            if isinstance(msgs_raw, list):
                for m in msgs_raw:

                    if not isinstance(m, dict):
                        continue

                    role = m.get("role")
                    text = m.get("text")
                    if role in ("user", "assistant") and isinstance(text, str) and text.strip():
                        msgs.append({"role": role, "text": text})

            totals_raw = raw.get("usage_totals", {})
            prompt_total = 0
            completion_total = 0
            if isinstance(totals_raw, dict):
                try:
                    prompt_total = int(totals_raw.get("prompt_tokens", 0))
                except Exception:
                    prompt_total = 0
                try:
                    completion_total = int(totals_raw.get("completion_tokens", 0))
                except Exception:
                    completion_total = 0

            return {
                "messages": msgs,
                "usage_totals": {
                    "prompt_tokens": max(0, prompt_total),
                    "completion_tokens": max(0, completion_total),
                },

            }

        except Exception:
            return default

    def _save_memory(self, msgs: List[Dict[str, str]], usage_totals: Dict[str, int]) -> None:
        """
        Сохраняет журнал атомарно (через .tmp -> replace).

        """
        data = {
            "messages": msgs,
            "usage_totals": {
                "prompt_tokens": int(usage_totals.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage_totals.get("completion_tokens", 0) or 0),
            },
        }
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

        mem = self._load_memory()
        msgs = mem["messages"]
        totals = mem["usage_totals"]

        msgs.append({"role": role, "text": text})
        self._save_memory(msgs, totals)

    def _update_usage_totals(self, prompt_tokens: int, completion_tokens: int) -> int:
        """
        Обновляет totals в memory.json и возвращает history_total_tokens (prompt+completion).
        """
        mem = self._load_memory()
        msgs = mem["messages"]
        totals = mem["usage_totals"]

        totals["prompt_tokens"] = int(totals.get("prompt_tokens", 0) or 0) + int(prompt_tokens or 0)
        totals["completion_tokens"] = int(totals.get("completion_tokens", 0) or 0) + int(completion_tokens or 0)

        # Атомарно сохраним вместе с messages, чтобы структура не расходилась
        self._save_memory(msgs, totals)

        return int(totals["prompt_tokens"]) + int(totals["completion_tokens"])

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

    def _call_openai_responses(self, messages: List[Dict[str, Any]]) -> Tuple[str, int, int]:
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

                assistant_text = extract_assistant_text(resp_json)
                prompt_tokens, completion_tokens = _extract_usage_tokens(resp_json)

                return assistant_text, prompt_tokens, completion_tokens

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
