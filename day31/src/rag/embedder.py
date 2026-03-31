import json
import urllib.request

from src.config import LM_STUDIO_BASE


LM_STUDIO_URL = f"{LM_STUDIO_BASE}/v1/embeddings"
LM_STUDIO_CHAT_URL = f"{LM_STUDIO_BASE}/api/v1/chat/completions"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"


def embed(texts: list[str]) -> list[list[float]]:
    """Получает эмбеддинги через LM Studio."""
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    # data["data"] — список объектов с "embedding"
    return [item["embedding"] for item in data["data"]]
