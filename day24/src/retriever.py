import json
import math


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_index(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_lookup(index: list[dict]) -> dict[tuple[str, int], dict]:
    return {
        (record["source"], record["chunk_index"]): record
        for record in index
    }


def _expand_with_neighbors(
    chunk: dict,
    lookup: dict[tuple[str, int], dict],
    window: int,
) -> str:
    source = chunk["source"]
    idx = chunk["chunk_index"]
    parts = []
    for i in range(idx - window, idx + window + 1):
        neighbor = lookup.get((source, i))

        if neighbor:
            parts.append(neighbor["text"])
    return "\n\n".join(parts)


def search(
    query_embedding: list[float],
    index: list[dict],
    top_k: int = 20,
    window: int = 3,
) -> list[dict]:
    scored = [
        (record, _cosine_similarity(query_embedding, record["embedding"]))
        for record in index
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    lookup = _build_lookup(index) if window > 0 else {}

    results = []

    for record, score in scored[:top_k]:
        expanded_text = (
            _expand_with_neighbors(record, lookup, window)
            if window > 0
            else record["text"]
        )
        results.append({**record, "text": expanded_text, "score": score})


    return results
