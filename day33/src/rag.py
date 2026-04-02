"""
RAG-модуль: загружает docs/*.txt, индексирует через OpenAI embeddings,
ищет топ-k релевантных чанков по косинусному сходству.
"""

import json
import math
import os
import urllib.request
import urllib.error
from pathlib import Path


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(texts: list[str], api_url: str, api_key: str, model: str = "text-embedding-3-small") -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # data["data"] отсортирован по "index"
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Делит текст на чанки по ~chunk_size символов с перекрытием."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class RAGIndex:
    def __init__(self, docs_dir: Path, api_url: str, api_key: str):
        self._api_url = api_url
        self._api_key = api_key
        self._chunks: list[str] = []
        self._embeddings: list[list[float]] = []
        self._build(docs_dir)

    def _build(self, docs_dir: Path) -> None:
        txt_files = sorted(docs_dir.glob("*.txt"))
        if not txt_files:
            print(f"[RAG] Предупреждение: нет .txt файлов в {docs_dir}")
            return

        all_chunks: list[str] = []
        for path in txt_files:
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            all_chunks.extend(chunks)
            print(f"[RAG] {path.name}: {len(chunks)} чанков")

        print(f"[RAG] Индексируем {len(all_chunks)} чанков через OpenAI embeddings...")

        # батчами по 32, чтобы не превышать лимиты
        batch_size = 32
        embeddings: list[list[float]] = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings.extend(_embed(batch, self._api_url, self._api_key))

        self._chunks = all_chunks
        self._embeddings = embeddings
        print(f"[RAG] Индекс готов: {len(self._chunks)} чанков\n")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if not self._chunks:
            return []
        query_emb = _embed([query], self._api_url, self._api_key)[0]
        scored = [
            (_cosine_similarity(query_emb, emb), chunk)
            for emb, chunk in zip(self._embeddings, self._chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
