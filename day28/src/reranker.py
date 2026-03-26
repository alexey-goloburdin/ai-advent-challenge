import re

from .llm import chat

RERANK_SYSTEM = (
    "Оцени насколько текст помогает ответить на вопрос. "
    "Шкала: 0 = текст вообще не связан с вопросом, "
    "5 = текст частично связан, "
    "10 = текст прямо отвечает на вопрос. "
    "Верни ТОЛЬКО целое число от 0 до 10 — без пояснений, без текста."
)


def rerank_llm(
    query: str,
    chunks: list[dict],
    model: str,
    api_url: str,
    min_score: float | None = None,
) -> list[dict]:
    """

    LLM-реранкинг: для каждого чанка запрашивает оценку релевантности 0-10.
    Возвращает отсортированный список с полем rerank_score.
    Если min_score задан — отфильтровывает чанки ниже порога.
    """
    if not chunks:
        return []

    scored = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        print(f"    LLM rerank {i}/{total}...")
        prompt = f"Вопрос: {query}\n\nТекст: {chunk['text']}"
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            response = chat(messages, model=model, api_url=api_url)
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
            score = float(m.group(1)) if m else 0.0
            print(f"      → {score} (raw: {response[:80].strip()!r})")
        except Exception:
            score = 0.0

        scored.append({**chunk, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    if min_score is not None:
        scored = [c for c in scored if c["rerank_score"] >= min_score]

    return scored
