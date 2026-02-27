# agent.py
import json
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JSON_STOP_SEQUENCE = "<JSON_END>"


# -------- Config --------

@dataclass
class AgentConfig:
    model: str = "gpt-5.2"
    max_output_tokens: int = 1024
    json_format: bool = False

    # Context management
    max_history_turns: int = 50

    # Memory management: file-journal
    memory_path: str = "memory.json"

    # --- Summarization experiment ---
    # Keep last N persisted messages "as is"
    keep_last_messages: int = 10

    # Summarize older history in batches of N messages
    summarize_batch_size: int = 10

    # Summarization model settings (can be same as main)
    summary_model: Optional[str] = None
    summary_max_output_tokens: int = 4096



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


def build_summary_system_prompt() -> str:
    # Суммаризация должна помогать именно агенту реквизитов:

    # фиксируем уже известные поля, какие вопросы заданы, чего не хватает, и любые важные уточнения.
    return (

        "Ты — помощник, который сжимает историю диалога в краткую сводку для последующей подстановки в контекст.\n"
        "Задача: кратко и точно суммировать диалог, НЕ выдумывая фактов.\n"
        "\n"
        "Правила сводки:\n"
        "- Пиши по-русски.\n"
        "- Сохраняй только полезную информацию для продолжения сбора реквизитов.\n"
        "- Если какие-то реквизиты уже сообщены — перечисли их как 'Известно: ...'.\n"
        "- Если чего-то не хватает — перечисли как 'Нужно уточнить: ...'.\n"
        "- Если были договоренности/ограничения/исключения/особые условия — включи.\n"
        "- Не добавляй общих фраз. Не добавляй дисклеймеров.\n"
        "- Формат: короткие пункты (маркированный список).\n"
    )


# -------- Response parsing --------

def extract_assistant_text(resp: Dict[str, Any]) -> str:
    """
    Более устойчивый экстрактор текста из Responses API.

    Поддерживает:
    - верхнеуровневое поле output_text (если присутствует)
    - output: list[...], где текст может быть:
      - item["content"][...]["text"]
      - item["text"]
    """
    # 1) Самый простой и частый вариант
    ot = resp.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = resp.get("output")
    texts: List[str] = []

    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue

            # Иногда текст кладут прямо в item["text"]
            t_item = item.get("text")
            if isinstance(t_item, str) and t_item.strip():
                texts.append(t_item.strip())

            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    # Иногда блок может быть строкой (редко, но бывает)
                    if isinstance(block, str) and block.strip():
                        texts.append(block.strip())

                        continue

                    if not isinstance(block, dict):
                        continue

                    t = block.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
                        continue

                    # На всякий случай: если вдруг "text" лежит вложенно

                    # (не всегда нужно, но дешево)
                    maybe = block.get("content")
                    if isinstance(maybe, str) and maybe.strip():
                        texts.append(maybe.strip())

    if texts:
        return "\n".join(texts).strip()


    # 2) Ничего не нашли — это не обязательно “фатал” модели, но нам нужен дебаг.
    raise ValueError(
        "Не удалось извлечь текст из ответа. "
        f"Верхние ключи: {sorted(list(resp.keys()))}"
    )


def _extract_usage_tokens(resp_json: Dict[str, Any]) -> Tuple[int, int]:
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)

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



# -------- Agent --------

class CompanyRequisitesAgent:
    """
    Агент со своим состоянием:

    - system prompt
    - история сообщений (в памяти процесса)
    - персистентный журнал сообщений в JSON-файле (--memory_path)
    - summary chunks (компрессия истории): хранится отдельно в memory.json

    - usage_totals и summary_usage_totals в memory.json
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

        # last token stats: (prompt_tokens, completion_tokens, after_est)
        self._last_token_stats: Optional[Tuple[int, int, int]] = None


        # Загружаем memory: messages + summaries + totals
        mem = self._load_memory()
        persisted_msgs = mem["messages"]
        summary_chunks = mem["summary"]["chunks"]

        # Собираем текущую историю (short-term) для API:
        # system + (summary as system) + last messages
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]}
        ]

        summary_text = self._format_summary_for_context(summary_chunks)
        if summary_text:
            # Важно: summary подставляем в контекст как system-сообщение

            self._messages.append(
                {"role": "system", "content": [{"type": "input_text", "text": summary_text}]}
            )

        self._messages.extend(self._persisted_to_api_messages(persisted_msgs))

        # Подрежем short-term историю, если кто-то руками засунул слишком много в messages
        self._trim_history()

    # -------- Public helpers --------

    def get_persisted_messages(self) -> List[Dict[str, str]]:
        """Для рендера в CLI: возвращает сохранённые сообщения (role/text) из memory.json (без summary)."""
        return self._load_memory()["messages"]

    def get_last_token_stats(self) -> Optional[Tuple[int, int, int]]:

        return self._last_token_stats

    # -------- Public API --------

    def reply(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""


        # --- user ---
        self._append_user(user_text)
        self._append_persisted_message({"role": "user", "text": user_text})


        # ВАЖНО: сворачиваем сразу после записи user

        self._maybe_summarize_history()

        self._trim_history()

        # --- main LLM call ---
        messages = list(self._messages)
        assistant_text, prompt_tokens, completion_tokens = self._call_openai_responses(
            model=self.config.model,
            messages=messages,
            max_output_tokens=self.config.max_output_tokens,
            stop=[JSON_STOP_SEQUENCE] if self.config.json_format else None,
        )

        # --- assistant ---
        self._append_assistant(assistant_text)

        clean_text = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in clean_text:
            clean_text = clean_text.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()

        self._append_persisted_message({"role": "assistant", "text": clean_text})

        # ВАЖНО: и сразу после записи assistant тоже сворачиваем
        self._maybe_summarize_history()

        self._trim_history()

        self._update_usage_totals(prompt_tokens, completion_tokens, kind="main")
        self._last_token_stats = (prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)

        return self._postprocess_output_for_display(assistant_text)

    # -------- Context management --------

    def _append_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": [{"type": "input_text", "text": text}]})


    def _append_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})

    def _trim_history(self) -> None:
        """
        Оставляем:
        - system (1)
        - + (optional) summary system (0/1)

        - + максимум N ходов user+assistant
        """
        max_turns = max(1, int(self.config.max_history_turns))
        max_msgs_tail = max_turns * 2

        # заголовок: system + возможный summary-system
        head: List[Dict[str, Any]] = []
        if self._messages:
            head.append(self._messages[0])
        if len(self._messages) >= 2 and self._messages[1].get("role") == "system":
            # мы именно так добавляем summary: вторым system-сообщением
            head.append(self._messages[1])

        tail = self._messages[len(head):]
        if len(tail) > max_msgs_tail:

            tail = tail[-max_msgs_tail:]

        self._messages = head + tail

    # -------- Persisted memory (JSON journal + summaries) --------

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "summary": {"chunks": []},  # list[{text, message_count}]
            "messages": [],  # last messages only (rolling)
            "usage_totals": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "summary_prompt_tokens": 0,
                "summary_completion_tokens": 0,
            },
        }

    def _load_memory(self) -> Dict[str, Any]:
        """

        Читает memory.json и возвращает структуру:
        {
          "summary": {"chunks": [{"text": "...", "message_count": 10}, ...]},
          "messages": [ {"role": "...", "text": "..."}, ... ],
          "usage_totals": {
              "prompt_tokens": int, "completion_tokens": int,
              "summary_prompt_tokens": int, "summary_completion_tokens": int
          }
        }
        Если файла нет/битый — вернёт дефолт.
        """

        default = self._default_memory()

        if not self.memory_path.exists():
            return default

        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default

            # messages
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


            # summary chunks

            summary_raw = raw.get("summary", {})
            chunks: List[Dict[str, Any]] = []
            if isinstance(summary_raw, dict):
                ch = summary_raw.get("chunks", [])
                if isinstance(ch, list):
                    for item in ch:
                        if not isinstance(item, dict):
                            continue
                        t = item.get("text")
                        mc = item.get("message_count", 0)

                        if isinstance(t, str) and t.strip():
                            try:
                                mc_i = int(mc)
                            except Exception:
                                mc_i = 0
                            chunks.append({"text": t.strip(), "message_count": max(0, mc_i)})

            # usage totals (backward-compatible)
            totals_raw = raw.get("usage_totals", {})
            totals = dict(default["usage_totals"])
            if isinstance(totals_raw, dict):
                for k in list(totals.keys()):
                    try:

                        totals[k] = max(0, int(totals_raw.get(k, totals[k]) or 0))
                    except Exception:
                        totals[k] = totals[k]

                # старые ключи prompt_tokens/completion_tokens тоже поддержим
                # (у нас они уже есть)

                # ничего больше не нужно

            return {
                "summary": {"chunks": chunks},
                "messages": msgs,
                "usage_totals": totals,
            }

        except Exception:
            return default

    def _save_memory(self, mem: Dict[str, Any]) -> None:
        """
        Атомарное сохранение memory.json (через .tmp -> replace).
        """
        # нормализуем структуру
        data = self._default_memory()

        # summary
        chunks = []
        summary = mem.get("summary", {})
        if isinstance(summary, dict):
            ch = summary.get("chunks", [])
            if isinstance(ch, list):
                for item in ch:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("text")
                    mc = item.get("message_count", 0)
                    if isinstance(t, str) and t.strip():
                        try:
                            mc_i = int(mc)
                        except Exception:
                            mc_i = 0
                        chunks.append({"text": t.strip(), "message_count": max(0, mc_i)})
        data["summary"]["chunks"] = chunks

        # messages
        msgs = []
        msgs_raw = mem.get("messages", [])
        if isinstance(msgs_raw, list):
            for m in msgs_raw:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")

                text = m.get("text")
                if role in ("user", "assistant") and isinstance(text, str) and text.strip():
                    msgs.append({"role": role, "text": text})
        data["messages"] = msgs

        # totals
        totals = data["usage_totals"]
        totals_raw = mem.get("usage_totals", {})

        if isinstance(totals_raw, dict):
            for k in list(totals.keys()):

                try:

                    totals[k] = max(0, int(totals_raw.get(k, totals[k]) or 0))

                except Exception:
                    pass
        data["usage_totals"] = totals

        tmp = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.memory_path)


    def _append_persisted_message(self, msg: Dict[str, str]) -> None:
        role = msg.get("role")
        text = msg.get("text")
        if role not in ("user", "assistant"):
            return

        if not isinstance(text, str) or not text.strip():
            return

        mem = self._load_memory()
        mem["messages"].append({"role": role, "text": text})
        self._save_memory(mem)

    def _update_usage_totals(self, prompt_tokens: int, completion_tokens: int, kind: str) -> None:
        """
        kind:
          - "main"    -> prompt_tokens/completion_tokens

          - "summary" -> summary_prompt_tokens/summary_completion_tokens
        """
        mem = self._load_memory()
        totals = mem["usage_totals"]


        if kind == "summary":
            totals["summary_prompt_tokens"] = int(totals.get("summary_prompt_tokens", 0) or 0) + int(prompt_tokens or 0)
            totals["summary_completion_tokens"] = int(totals.get("summary_completion_tokens", 0) or 0) + int(completion_tokens or 0)
        else:
            totals["prompt_tokens"] = int(totals.get("prompt_tokens", 0) or 0) + int(prompt_tokens or 0)
            totals["completion_tokens"] = int(totals.get("completion_tokens", 0) or 0) + int(completion_tokens or 0)

        mem["usage_totals"] = totals
        self._save_memory(mem)

    def _persisted_to_api_messages(self, msgs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        api_msgs: List[Dict[str, Any]] = []
        for m in msgs:
            role = m["role"]
            text = m["text"]
            if role == "user":
                api_msgs.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
            elif role == "assistant":
                api_msgs.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
        return api_msgs

    # -------- Summarization logic --------

    def _format_summary_for_context(self, chunks: List[Dict[str, Any]]) -> str:

        if not chunks:
            return ""
        parts = []
        for i, ch in enumerate(chunks, start=1):
            t = (ch.get("text") or "").strip()
            if not t:
                continue
            parts.append(f"Сводка #{i}:\n{t}")
        if not parts:
            return ""
        return "СВОДКА РАНЕЕШНЕГО ДИАЛОГА (сжатая история):\n\n" + "\n\n".join(parts)

    def _refresh_summary_in_short_term_context(self) -> None:
        """
        Пересобирает summary-system сообщение (2-е system) в self._messages
        на основе memory.json.
        """
        mem = self._load_memory()

        summary_text = self._format_summary_for_context(mem["summary"]["chunks"])

        # self._messages[0] = основной system
        # self._messages[1] = summary system (если есть)
        if summary_text:
            summary_msg = {"role": "system", "content": [{"type": "input_text", "text": summary_text}]}
            if len(self._messages) >= 2 and self._messages[1].get("role") == "system":
                self._messages[1] = summary_msg
            else:
                self._messages.insert(1, summary_msg)
        else:
            # summary нет — уберём, если он был
            if len(self._messages) >= 2 and self._messages[1].get("role") == "system":
                self._messages.pop(1)


    def _maybe_summarize_history(self) -> None:
        """
        Правило:
        - хранить последние keep_last_messages сообщений как есть
        - всё остальное сворачивать в summary батчами по summarize_batch_size
        """
        keep_n = max(0, int(self.config.keep_last_messages))
        batch_n = max(1, int(self.config.summarize_batch_size))

        mem = self._load_memory()
        msgs = mem["messages"]

        chunks = mem["summary"]["chunks"]


        # Пока "старой" части хватает минимум на один батч
        # и при этом мы сохраняем keep_n последних сообщений нетронутыми.
        changed = False
        while len(msgs) >= keep_n + batch_n and batch_n > 0:
            batch = msgs[:batch_n]
            # сформируем текст батча для суммаризации
            dialogue = self._format_dialogue_batch(batch)


            summary_text, p_tok, c_tok = self._summarize_text(dialogue)

            # учёт токенов на суммаризацию отдельно
            self._update_usage_totals(p_tok, c_tok, kind="summary")

            chunks.append({"text": summary_text, "message_count": batch_n})

            msgs = msgs[batch_n:]
            changed = True

        if changed:
            mem["messages"] = msgs
            mem["summary"]["chunks"] = chunks
            self._save_memory(mem)

            # важный момент: short-term контекст должен обновить summary-сообщение
            self._refresh_summary_in_short_term_context()

            # и short-term messages должны соответствовать persisted tail:
            # проще всего пересобрать хвост в self._messages (без system + summary).
            self._rebuild_short_term_tail_from_persisted()

    def _rebuild_short_term_tail_from_persisted(self) -> None:
        """
        Пересобирает хвост self._messages (user/assistant) из memory.json.messages.
        Это гарантирует, что после удаления старых сообщений при суммаризации
        short-term контекст не содержит "призраков" удалённой истории.
        """
        mem = self._load_memory()

        persisted_msgs = mem["messages"]

        head: List[Dict[str, Any]] = []
        if self._messages:
            head.append(self._messages[0])
        if len(self._messages) >= 2 and self._messages[1].get("role") == "system":
            head.append(self._messages[1])

        self._messages = head + self._persisted_to_api_messages(persisted_msgs)


    def _format_dialogue_batch(self, batch: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for m in batch:
            role = m.get("role")
            text = (m.get("text") or "").strip()
            if not text:
                continue
            prefix = "Пользователь" if role == "user" else "Агент"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines).strip()

    def _summarize_text(self, dialogue_text: str) -> Tuple[str, int, int]:
        """
        Делает 1 вызов LLM для суммаризации батча.
        Возвращает (summary_text, prompt_tokens, completion_tokens).

        """
        dialogue_text = (dialogue_text or "").strip()
        if not dialogue_text:
            return ("", 0, 0)

        model = self.config.summary_model or self.config.model
        max_out = max(64, int(self.config.summary_max_output_tokens or 256))

        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": build_summary_system_prompt()}]},
            {"role": "user", "content": [{"type": "input_text", "text": dialogue_text}]},
        ]

        text, p_tok, c_tok = self._call_openai_responses(

            model=model,
            messages=messages,
            max_output_tokens=max_out,

            stop=None,
        )
        return (text.strip(), p_tok, c_tok)

    # -------- LLM call (encapsulated) --------

    def _call_openai_responses(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_output_tokens: int,
        stop: Optional[List[str]],
    ) -> Tuple[str, int, int]:
        payload: Dict[str, Any] = {

            "model": model,
            "input": messages,
            "max_output_tokens": int(max_output_tokens),
        }
        if stop:
            payload["stop"] = stop

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
