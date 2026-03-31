import json
import urllib.request

from src.config import LM_STUDIO_BASE

LM_STUDIO_URL = f"{LM_STUDIO_BASE}/v1/embeddings"
LM_STUDIO_CHAT_URL = f"{LM_STUDIO_BASE}/v1/chat/completions"
CHAT_MODEL = "qwen/qwen3.5-9b"  # или ваша модель в LM Studio


def ask(
    question: str,
    rag_chunks: list[dict],
    git_branch: str,
) -> str:
    """Формирует промпт с RAG-контекстом и git-веткой, возвращает ответ LLM."""

    docs_context = "\n\n---\n\n".join(
        f"[{c['source']} / {c['title']}]\n{c['content']}" for c in rag_chunks
    )

    system = (
        "Ты — ассистент разработчика. Отвечай на вопросы о проекте, "
        "опираясь на предоставленную документацию и контекст репозитория. "
        "Если ответа в документации нет — скажи об этом явно."
    )

    user = (
        f"Текущая git-ветка: {git_branch}\n\n"
        f"Документация проекта:\n{docs_context}\n\n"
        f"Вопрос: {question}"
    )

    payload = json.dumps({
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "reasoning": "off",
    }).encode()

    req = urllib.request.Request(
        LM_STUDIO_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"].strip()
