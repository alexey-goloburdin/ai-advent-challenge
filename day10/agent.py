# agent.py
"""
Агент для сбора реквизитов компании с поддержкой трёх стратегий управления контекстом:
1. Sliding Window — только последние N сообщений
2. Sticky Facts — facts (key-value) + последние N сообщений
3. Branching — checkpoint'ы и независимые ветки диалога
"""

import json
import urllib.request
import urllib.error

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from context_strategies import (
    ContextStrategy,
    Message,
    create_strategy,
    StickyFactsStrategy,
    BranchingStrategy,
)

JSON_STOP_SEQUENCE = "<JSON_END>"


# ============================================================================
# Config
# ============================================================================

@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 1024
    json_format: bool = False

    # Стратегия: "sliding_window" | "sticky_facts" | "branching"
    strategy: str = "sliding_window"

    # Размер окна (для всех стратегий)
    window_size: int = 10

    # Путь для сохранения состояния
    memory_path: str = "memory.json"


# ============================================================================
# System Prompt Builder
# ============================================================================

def build_system_prompt(json_format: bool, extra_content: Optional[str] = None) -> str:
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

    # Вставляем дополнительный контент (например, facts)
    if extra_content:
        prompt += f"\n{extra_content}\n"

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


# ============================================================================
# Response Parsing
# ============================================================================

def extract_assistant_text(resp: Dict[str, Any]) -> str:
    """Извлекает текст ответа из Responses API."""
    ot = resp.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = resp.get("output")
    texts: List[str] = []

    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue

            t_item = item.get("text")
            if isinstance(t_item, str) and t_item.strip():
                texts.append(t_item.strip())

            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, str) and block.strip():
                        texts.append(block.strip())
                        continue
                    if not isinstance(block, dict):
                        continue
                    t = block.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())

    if texts:
        return "\n".join(texts).strip()

    raise ValueError(f"Не удалось извлечь текст из ответа. Ключи: {sorted(list(resp.keys()))}")


def extract_usage_tokens(resp_json: Dict[str, Any]) -> Tuple[int, int]:
    """Извлекает статистику токенов из ответа."""
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return (0, 0)

    prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0

    try:
        return (max(0, int(prompt)), max(0, int(completion)))
    except Exception:
        return (0, 0)


# ============================================================================
# Agent
# ============================================================================

class CompanyRequisitesAgent:
    """
    Агент для сбора реквизитов компании.
    Поддерживает три стратегии управления контекстом.
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
        self.memory_path = Path(config.memory_path)

        # Статистика токенов
        self._last_token_stats: Optional[Tuple[int, int, int]] = None
        self._total_tokens: Tuple[int, int] = (0, 0)  # (prompt, completion)

        # Создаём стратегию
        self._strategy: ContextStrategy = create_strategy(
            strategy_type=config.strategy,
            window_size=config.window_size,
            llm_caller=self._llm_call_for_facts if config.strategy == "sticky_facts" else None,
        )

        # Загружаем состояние
        self._strategy.load(self.memory_path)

        # Для sticky_facts нужно установить llm_caller после создания
        if isinstance(self._strategy, StickyFactsStrategy):
            self._strategy.set_llm_caller(self._llm_call_for_facts)

    # ========== Public API ==========

    def reply(self, user_text: str) -> str:
        """Обработать сообщение пользователя и вернуть ответ."""
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        # Добавляем сообщение пользователя
        self._strategy.add_message(Message(role="user", text=user_text))

        # Собираем контекст для LLM
        extra_content = self._strategy.get_extra_system_content()
        system_prompt = build_system_prompt(self.config.json_format, extra_content)

        messages = self._build_api_messages(system_prompt)

        # Вызываем LLM
        assistant_text, prompt_tokens, completion_tokens = self._call_openai_responses(
            model=self.config.model,
            messages=messages,
            max_output_tokens=self.config.max_output_tokens,
            stop=[JSON_STOP_SEQUENCE] if self.config.json_format else None,
        )

        # Обрабатываем ответ
        clean_text = assistant_text
        if self.config.json_format and JSON_STOP_SEQUENCE in clean_text:
            clean_text = clean_text.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()

        # Добавляем ответ ассистента
        self._strategy.add_message(Message(role="assistant", text=clean_text))

        # Сохраняем состояние
        self._strategy.save(self.memory_path)

        # Обновляем статистику
        self._total_tokens = (
            self._total_tokens[0] + prompt_tokens,
            self._total_tokens[1] + completion_tokens,
        )
        self._last_token_stats = (prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)

        return clean_text

    def get_last_token_stats(self) -> Optional[Tuple[int, int, int]]:
        """Статистика токенов последнего запроса."""
        return self._last_token_stats

    def get_total_tokens(self) -> Tuple[int, int]:
        """Суммарная статистика токенов."""
        return self._total_tokens

    def get_persisted_messages(self) -> List[Dict[str, str]]:
        """Все сообщения для отображения."""
        return [m.to_dict() for m in self._strategy.get_all_messages()]

    def get_strategy_info(self) -> str:
        """Информация о текущей стратегии."""
        return self._strategy.get_strategy_info()

    def get_strategy(self) -> ContextStrategy:
        """Доступ к стратегии напрямую (для branching операций)."""
        return self._strategy

    def reset(self) -> None:
        """Сбросить состояние агента."""
        self._strategy = create_strategy(
            strategy_type=self.config.strategy,
            window_size=self.config.window_size,
            llm_caller=self._llm_call_for_facts if self.config.strategy == "sticky_facts" else None,
        )
        self._total_tokens = (0, 0)
        self._last_token_stats = None
        if self.memory_path.exists():
            self.memory_path.unlink()

    # ========== Branching Operations (delegates) ==========

    def create_checkpoint(self, name: str) -> bool:
        """Создать checkpoint (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return False
        result = self._strategy.create_checkpoint(name)
        if result:
            self._strategy.save(self.memory_path)
        return result

    def create_branch(self, branch_name: str, from_checkpoint: str) -> bool:
        """Создать ветку от checkpoint (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return False
        result = self._strategy.create_branch(branch_name, from_checkpoint)
        if result:
            self._strategy.save(self.memory_path)
        return result

    def switch_branch(self, branch_name: str) -> bool:
        """Переключиться на ветку (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return False
        result = self._strategy.switch_branch(branch_name)
        if result:
            self._strategy.save(self.memory_path)
        return result

    def get_current_branch(self) -> Optional[str]:
        """Текущая ветка (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return None
        return self._strategy.get_current_branch()

    def list_branches(self) -> List[str]:
        """Список веток (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return []
        return self._strategy.list_branches()

    def list_checkpoints(self) -> List[str]:
        """Список checkpoint'ов (только для branching)."""
        if not isinstance(self._strategy, BranchingStrategy):
            return []
        return self._strategy.list_checkpoints()

    # ========== Facts Operations (delegates) ==========

    def get_facts(self) -> Dict[str, str]:
        """Получить факты (только для sticky_facts)."""
        if not isinstance(self._strategy, StickyFactsStrategy):
            return {}
        return self._strategy.get_facts()

    # ========== Internal ==========

    def _build_api_messages(self, system_prompt: str) -> List[Dict[str, Any]]:
        """Собирает сообщения для API."""
        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}
        ]

        for msg in self._strategy.get_context_messages():
            if msg.role == "user":
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.text}]
                })
            elif msg.role == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg.text}]
                })

        return messages

    def _llm_call_for_facts(self, prompt: str) -> Tuple[str, int, int]:
        """LLM вызов для извлечения фактов (используется StickyFactsStrategy)."""
        messages = [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ]
        return self._call_openai_responses(
            model=self.config.model,
            messages=messages,
            max_output_tokens=512,
            stop=None,
        )

    def _call_openai_responses(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_output_tokens: int,
        stop: Optional[List[str]],
    ) -> Tuple[str, int, int]:
        """Вызов OpenAI Responses API."""
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
                prompt_tokens, completion_tokens = extract_usage_tokens(resp_json)
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
