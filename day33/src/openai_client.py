"""
Обёртка над OpenAI chat completions с поддержкой tool_calls (function calling).
Использует только urllib.
"""

import json
import urllib.request

import urllib.error
from typing import Any


def chat_completion(
    messages: list[dict],
    model: str,
    api_url: str,
    api_key: str,
    tools: list[dict] | None = None,
    temperature: float = 0.3,
) -> dict:
    """
    Возвращает dict с полями:
      - content: str | None  (текстовый ответ)
      - tool_calls: list     (вызовы инструментов, если есть)
      - finish_reason: str
    """
    payload: dict[str, Any] = {

        "model": model,
        "messages": messages,

        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {e.code}: {body}") from e

    message = result["choices"][0]["message"]
    finish_reason = result["choices"][0]["finish_reason"]

    return {
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": finish_reason,

        "raw_message": message,
    }
