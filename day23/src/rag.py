from .llm import chat
from .retriever import search
from .reranker import rerank

from .query_rewrite import rewrite_query


SYSTEM_PROMPT = "Ты helpful ассистент. Отвечай на русском языке."

SYSTEM_PROMPT_RAG = (

    "Ты helpful ассистент. Отвечай на русском языке. "
    "Используй только информацию из предоставленного контекста. "
    "Если в контексте нет ответа — так и скажи."
)

RAG_TEMPLATE = """\
Контекст:
{context}

Вопрос: {question}
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "")
        section = chunk.get("section", "")
        label = f"[{i}] {source}"
        if section:
            label += f" / {section}"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer_without_rag(question: str, model: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},

    ]
    return chat(messages, model=model)


def answer_with_rag(
    question: str,
    query_embedding: list[float],
    index: list[dict],
    model: str,
    top_k: int = 20,
    window: int = 3,
    reranker_model: str | None = None,
    rerank_top_k: int | None = None,
    min_rerank_score: float | None = None,
    rewrite: bool = False,
) -> dict:
    """
    Возвращает dict с полями:
      answer          — ответ модели

      chunks          — финальные чанки после всех фильтров
      rewritten_query — переформулированный запрос (или None)
    """
    rewritten_query = None

    if rewrite:
        rewritten_query = rewrite_query(question, model=model)

    # Первичный поиск по cosine similarity
    candidates = search(query_embedding, index, top_k=top_k, window=window)

    # Реранкинг cross-encoder'ом
    if reranker_model is not None:
        query_for_rerank = rewritten_query if rewritten_query else question

        candidates = rerank(

            query=query_for_rerank,
            chunks=candidates,
            model_name=reranker_model,
            min_score=min_rerank_score,
        )
        if rerank_top_k is not None:
            candidates = candidates[:rerank_top_k]

    context = _build_context(candidates)
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": prompt},
    ]
    answer = chat(messages, model=model)


    return {
        "answer": answer,
        "chunks": candidates,
        "rewritten_query": rewritten_query,

    }
