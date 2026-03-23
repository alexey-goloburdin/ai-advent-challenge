"""
Загрузка индекса и поиск по нему через cosine similarity.
Поддерживает расширение контекста: к каждому найденному чанку
автоматически подтягиваются соседние чанки из того же документа.
"""

import json
import math
from collections import defaultdict

from urllib.request import urlopen, Request
from urllib.error import URLError


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self, index_path: str, ollama_host: str):
        self.ollama_host = ollama_host.rstrip("/")
        self.chunks: list[dict] = []
        # source -> список позиций в self.chunks, отсортированных по chunk_index

        self._source_index: dict[str, list[int]] = {}
        self._load(index_path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.chunks = data
        elif isinstance(data, dict) and "chunks" in data:
            self.chunks = data["chunks"]
        else:
            raise ValueError(f"Неизвестный формат индекса в {path}")

        if self.chunks:
            self.embedding_model = self.chunks[0].get(
                "embedding_model", "nomic-embed-text"
            )
        else:
            self.embedding_model = "nomic-embed-text"

        self._build_source_index()

        print(
            f"[VectorStore] Загружено {len(self.chunks)} чанков, "
            f"модель эмбеддингов: {self.embedding_model}"
        )

    def _build_source_index(self) -> None:
        """source -> [pos, ...] отсортированные по chunk_index."""
        source_map: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for pos, chunk in enumerate(self.chunks):
            source = chunk.get("source", "")
            chunk_index = chunk.get("chunk_index", pos)
            source_map[source].append((chunk_index, pos))
        self._source_index = {
            source: [pos for _, pos in sorted(pairs)]
            for source, pairs in source_map.items()
        }

    def _embed_query(self, text: str) -> list[float]:
        """Получить эмбеддинг текста через Ollama."""
        data = json.dumps(
            {"model": self.embedding_model, "prompt": text}
        ).encode()
        req = Request(
            f"{self.ollama_host}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            return result["embedding"]
        except URLError as e:

            raise RuntimeError(
                f"Ошибка подключения к Ollama ({self.ollama_host}): {e}"
            ) from e

    def _chunk_to_result(self, chunk: dict, score: float, is_neighbor: bool) -> dict:
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", ""),
            "score": round(score, 4),
            "is_neighbor": is_neighbor,
        }

    def search(self, query: str, top_k: int = 5, neighbors: int = 3) -> list[dict]:
        """
        Найти top_k релевантных чанков + neighbors соседей с каждой стороны.

        Порядок в результате:
          1. Найденные чанки (по убыванию score)
          2. Соседние чанки (по chunk_id, дедуплицированы)

        Поле is_neighbor=True помечает подтянутых соседей.

        """
        query_emb = self._embed_query(query)

        scored: list[tuple[float, int]] = []
        for pos, chunk in enumerate(self.chunks):
            emb = chunk.get("embedding")
            if not emb:
                continue
            score = _cosine(query_emb, emb)
            scored.append((score, pos))

        scored.sort(reverse=True)
        top_hits = scored[:top_k]

        included: dict[int, dict] = {}

        for score, pos in top_hits:
            chunk = self.chunks[pos]
            included[pos] = self._chunk_to_result(chunk, score, is_neighbor=False)

            source = chunk.get("source", "")
            ordered = self._source_index.get(source, [])
            if pos in ordered:
                idx = ordered.index(pos)

                start = max(0, idx - neighbors)
                end = min(len(ordered), idx + neighbors + 1)
                for neighbor_pos in ordered[start:end]:
                    if neighbor_pos not in included:
                        neighbor = self.chunks[neighbor_pos]
                        included[neighbor_pos] = self._chunk_to_result(
                            neighbor, score=0.0, is_neighbor=True
                        )

        hits = sorted(
            (r for r in included.values() if not r["is_neighbor"]),
            key=lambda x: x["score"],
            reverse=True,
        )
        neighbor_chunks = sorted(
            (r for r in included.values() if r["is_neighbor"]),
            key=lambda x: x["chunk_id"],
        )
        return hits + neighbor_chunks
