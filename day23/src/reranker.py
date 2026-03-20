from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model_cache: dict[str, CrossEncoder] = {}


def get_model(model_name: str = MODEL_NAME) -> CrossEncoder:
    if model_name not in _model_cache:
        print(f"  Загрузка reranker модели: {model_name}...")
        _model_cache[model_name] = CrossEncoder(model_name)

    return _model_cache[model_name]


def rerank(
    query: str,
    chunks: list[dict],
    model_name: str = MODEL_NAME,
    min_score: float | None = None,
) -> list[dict]:
    """
    Cross-encoder реранкинг.
    Хорошо работает для английского, хуже для русского.
    Для русского используй многоязычную модель:
      cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    """
    if not chunks:
        return []

    model = get_model(model_name)
    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = model.predict(pairs)

    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    if min_score is not None:

        scored = [c for c in scored if c["rerank_score"] >= min_score]

    return scored



def rerank_llm(
    query: str,
    chunks: list[dict],
    model: str,
    min_score: float | None = None,
) -> list[dict]:
    """
    LLM-реранкинг через OpenAI.
    Понимает русский отлично, но делает N запросов к API (по одному на чанк).
    """
    from .llm import chat


    RERANK_SYSTEM = (
        "Оцени релевантность текста к вопросу по шкале от 0 до 10. "
        "Верни ТОЛЬКО число — без пояснений, без единиц, без текста."
    )

    if not chunks:
        return []


    scored = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):

        print(f"    LLM rerank {i}/{total}...")
        prompt = f"Вопрос: {query}\n\nТекст: {chunk['text'][:1000]}"
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:

            response = chat(messages, model=model, max_tokens=10)
            score = float(response.strip())
        except (ValueError, Exception):
            score = 0.0

        scored.append({**chunk, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)


    if min_score is not None:
        scored = [c for c in scored if c["rerank_score"] >= min_score]

    return scored
