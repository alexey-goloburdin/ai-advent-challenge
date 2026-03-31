import json
import math
import pathlib

from .embedder import embed


INDEX_FILE = ".dev_assistant_index.json"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(documents: list[dict], index_path: str = INDEX_FILE) -> None:
    """Эмбеддит все документы и сохраняет индекс в JSON."""
    print(f"  Индексируем {len(documents)} чанков...")
    texts = [f"{d['title']}\n{d['content']}" for d in documents]
    vectors = embed(texts)

    index = [
        {"source": d["source"], "title": d["title"], "content": d["content"], "vector": v}
        for d, v in zip(documents, vectors)
    ]

    pathlib.Path(index_path).write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Индекс сохранён → {index_path}")


def load_index(index_path: str = INDEX_FILE) -> list[dict]:
    return json.loads(pathlib.Path(index_path).read_text(encoding="utf-8"))


def search(query: str, index: list[dict], top_k: int = 3) -> list[dict]:
    """Возвращает top_k наиболее релевантных чанков."""
    [query_vec] = embed([query])
    scored = [(chunk, _cosine(query_vec, chunk["vector"])) for chunk in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]
