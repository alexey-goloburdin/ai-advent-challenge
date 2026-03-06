#!/usr/bin/env python3
"""
CLI-агент с инвариантами и двухэтапной проверкой.
Коммуникация с OpenAI API через urllib.
"""

import argparse
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Invariants:
    """Инварианты проекта, загружаемые из файла."""
    
    project_name: str
    stack: dict
    architecture: dict
    forbidden: list[str]
    code_rules: list[dict]
    business_rules: list[dict]
    
    @classmethod
    def from_file(cls, path: str | Path) -> "Invariants":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            project_name=data.get("project_name", "Проект"),
            stack=data.get("stack", {}),
            architecture=data.get("architecture", {}),
            forbidden=data.get("forbidden_technologies", data.get("forbidden", [])),
            code_rules=data.get("code_rules", []),
            business_rules=data.get("business_rules", []),
        )
    
    def format_for_prompt(self) -> str:
        lines = [f"## Инварианты проекта: {self.project_name}"]
        lines.append("(ОБЯЗАТЕЛЬНЫ К СОБЛЮДЕНИЮ)")
        
        if self.stack:
            lines.append("\n### Стек технологий:")
            for key, value in self.stack.items():
                if isinstance(value, dict):
                    formatted = ", ".join(f"{k}: {v}" for k, v in value.items())
                    lines.append(f"- {key}: {formatted}")
                elif isinstance(value, bool):
                    lines.append(f"- {key}: {'да' if value else 'нет'}")
                else:
                    lines.append(f"- {key}: {value}")
        
        if self.architecture:
            lines.append("\n### Архитектура:")
            for key, value in self.architecture.items():
                if isinstance(value, list):
                    lines.append(f"- {key}: {', '.join(value)}")
                else:
                    lines.append(f"- {key}: {value}")
        
        if self.code_rules:
            lines.append("\n### Правила кода:")
            for rule in self.code_rules:
                desc = rule.get("description", rule.get("id", ""))
                lines.append(f"- {desc}")
        
        if self.business_rules:
            lines.append("\n### Бизнес-правила:")
            for rule in self.business_rules:
                desc = rule.get("description", rule.get("id", ""))
                lines.append(f"- {desc}")
        
        if self.forbidden:
            lines.append("\n### ЗАПРЕЩЕНО:")
            for item in self.forbidden:
                lines.append(f"- ❌ {item}")
        
        return "\n".join(lines)


class OpenAIClient:
    """Клиент для OpenAI API через urllib."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
    
    def chat_completion(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        data = json.dumps(payload).encode("utf-8")
        
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"API error {e.code}: {error_body}")


class InvariantViolation(Exception):
    """Исключение при нарушении инварианта."""
    
    def __init__(self, violated: list[str], explanation: str):
        self.violated = violated
        self.explanation = explanation
        super().__init__(explanation)


class InvariantAgent:
    """Агент с двухэтапной проверкой инвариантов."""
    
    def __init__(self, client: OpenAIClient, invariants: Invariants, model: str = "gpt-4o-mini"):
        self.client = client
        self.invariants = invariants
        self.model = model
        self.conversation: list[dict] = []
    
    def _build_system_prompt(self) -> str:
        return f"""Ты — технический ассистент проекта.

{self.invariants.format_for_prompt()}

ВАЖНО: 
- Все твои предложения ДОЛЖНЫ соответствовать инвариантам выше
- Если запрос противоречит инвариантам — откажи и объясни почему
- Не предлагай технологии из списка ЗАПРЕЩЕНО
"""
    
    def _check_response(self, user_request: str, response: str) -> None:
        """
        Вторая линия защиты: проверяем ответ LLM на соответствие инвариантам.
        Используем отдельный вызов LLM для анализа.
        """
        check_prompt = f"""Проверь, нарушает ли ответ ассистента инварианты проекта.

## Инварианты:
{self.invariants.format_for_prompt()}

## Запрос пользователя:
{user_request}

## Ответ ассистента для проверки:
{response}

## Задача:
Проанализируй, предлагает ли ответ что-то из списка ЗАПРЕЩЕНО или нарушает правила/стек/архитектуру.

Ответь СТРОГО в формате JSON (без markdown):
{{"violates": true/false, "violated_invariants": ["список нарушенных инвариантов"], "explanation": "объяснение"}}

Если нарушений нет, верни:
{{"violates": false, "violated_invariants": [], "explanation": "Ответ соответствует инвариантам"}}
"""
        
        messages = [
            {"role": "system", "content": "Ты — валидатор ответов. Отвечай только JSON."},
            {"role": "user", "content": check_prompt},
        ]
        
        check_result = self.client.chat_completion(messages, model=self.model, temperature=0)
        
        # Парсим JSON ответ
        try:
            # Убираем возможные markdown-обёртки
            clean_result = check_result.strip()
            if clean_result.startswith("```"):
                clean_result = clean_result.split("\n", 1)[1]
                clean_result = clean_result.rsplit("```", 1)[0]
            
            result = json.loads(clean_result)
        except json.JSONDecodeError:
            # Если не удалось распарсить — пропускаем проверку
            print(f"[DEBUG] Не удалось распарсить проверку: {check_result[:200]}")
            return
        
        if result.get("violates", False):
            raise InvariantViolation(
                violated=result.get("violated_invariants", []),
                explanation=result.get("explanation", "Нарушение инварианта"),
            )
    
    def chat(self, user_message: str) -> str:
        """Обрабатывает сообщение пользователя с двухэтапной проверкой."""
        
        # Формируем сообщения
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_message})
        
        # Этап 1: Генерация ответа
        print("[...] Генерация ответа...")
        response = self.client.chat_completion(messages, model=self.model)
        
        # Этап 2: Проверка на соответствие инвариантам
        print("[...] Проверка инвариантов...")
        try:
            self._check_response(user_message, response)
        except InvariantViolation as e:
            # Ответ нарушает инварианты — возвращаем отказ
            violation_msg = f"""⚠️ **Обнаружено нарушение инвариантов!**

**Нарушены:** {', '.join(e.violated)}

**Причина:** {e.explanation}

Не могу предложить это решение, так как оно противоречит установленным ограничениям проекта."""
            
            # Сохраняем в историю отказ
            self.conversation.append({"role": "user", "content": user_message})
            self.conversation.append({"role": "assistant", "content": violation_msg})
            
            return violation_msg
        
        # Всё ок — сохраняем в историю и возвращаем
        self.conversation.append({"role": "user", "content": user_message})
        self.conversation.append({"role": "assistant", "content": response})
        
        return response
    
    def show_invariants(self) -> str:
        """Показывает текущие инварианты."""
        return self.invariants.format_for_prompt()


def main():
    parser = argparse.ArgumentParser(description="CLI-агент с инвариантами")
    parser.add_argument(
        "-m", "--model",
        default="gpt-4o-mini",
        help="Модель для использования (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "-i", "--invariants",
        default=None,
        help="Путь к файлу инвариантов (default: invariants.json рядом со скриптом)"
    )
    args = parser.parse_args()
    
    # Загружаем API ключ
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: установите переменную OPENAI_API_KEY")
        return
    
    # Загружаем инварианты
    if args.invariants:
        invariants_path = Path(args.invariants)
    else:
        invariants_path = Path(__file__).parent / "invariants.json"
    
    if not invariants_path.exists():
        print(f"Ошибка: файл инвариантов не найден: {invariants_path}")
        return
    
    invariants = Invariants.from_file(invariants_path)
    
    # Опционально: кастомный base_url (например, для polza.ai)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    client = OpenAIClient(api_key, base_url)
    agent = InvariantAgent(client, invariants, model=args.model)
    
    print("=" * 60)
    print(f"🔒 CLI-агент с инвариантами | Модель: {args.model}")
    print("=" * 60)
    print("\nКоманды:")
    print("  /invariants — показать текущие инварианты")
    print("  /clear      — очистить историю диалога")
    print("  /quit       — выход")
    print()
    print(agent.show_invariants())
    print("\n" + "=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            print("Выход.")
            break
        
        if user_input == "/invariants":
            print("\n" + agent.show_invariants() + "\n")
            continue
        
        if user_input == "/clear":
            agent.conversation.clear()
            print("История очищена.\n")
            continue
        
        try:
            response = agent.chat(user_input)
            print(f"\nАссистент: {response}\n")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
