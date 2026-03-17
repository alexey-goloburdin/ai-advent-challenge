import json
import os
import urllib.request


def get_api_config() -> tuple[str, str]:
    api_url = os.environ.get(
        "OPENAI_API_URL",

        "https://api.openai.com/v1/chat/completions",
    )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Переменная окружения OPENAI_API_KEY не задана")
    return api_url, api_key


def chat(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
) -> str:
    api_url, api_key = get_api_config()

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from None

    return data["choices"][0]["message"]["content"].strip()
