import json
import urllib.error
import urllib.request


def chat(
    messages: list[dict],
    model: str = "qwen/qwen3.5-9b",
    max_tokens: int = 1024,
    api_url: str = "http://localhost:1234",
) -> str:
    """Chat via LM Studio native /api/v1/chat endpoint with reasoning disabled."""
    base = api_url.rstrip("/")
    # Strip any path the caller may have appended — we always use /api/v1/chat
    for suffix in ("/api/v1/chat", "/v1/chat/completions", "/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    url = base + "/api/v1/chat"

    role_labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    input_text = "\n\n".join(
        f"{role_labels.get(m['role'], m['role'])}: {m['content']}"
        for m in messages
    )

    payload = json.dumps({
        "model": model,
        "input": input_text,
        "reasoning": "off",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from None

    # /api/v1/chat returns {"output": [{"type": "message", "content": "..."}]}
    if "output" in data:
        return data["output"][0]["content"].strip()
    raise RuntimeError(f"Unexpected response from LM Studio: {data}")
