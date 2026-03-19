from sentence_transformers import CrossEncoder
from .llm import chat


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


_model: CrossEncoder | None = None



def get_model(model_name: str = MODEL_NAME) -> CrossEncoder:
    global _model
    if _model is None:
        print(f"  Загрузка reranker модели: {model_name}...")
        _model = CrossEncoder(model_name)
    return _model



def rerank(
    query: str,
    chunks: list[dict],

    model_name: str = MODEL_NAME,

    min_score: float | None = None,
) -> list[dict]:
    """
    Переранжирует чанки по релевантности к запросу.
    Возвращает отсортированный список с добавленным полем rerank_score.
    Если min_score задан — отфильтровывает чанки ниже порога.
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



RERANK_SYSTEM = """\
Оцени релевантность текста к вопросу по шкале от 0 до 10.
Верни ТОЛЬКО число — без пояснений, без единиц, без текста.\
"""


def rerank_llm(
    query: str,
    chunks: list[dict],
    model: str,
    min_score: float | None = None,
) -> list[dict]:
    scored = []
    for chunk in chunks:
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
