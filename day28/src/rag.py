import json

from .llm import chat
from .retriever import search
from .reranker import rerank_llm

# ── Промпты ───────────────────────────────────────────────────────────────────

SYSTEM_NO_RAG = "Ты helpful ассистент. Отвечай на русском языке."

SYSTEM_RAG = """\
Ты helpful ассистент. Отвечай ТОЛЬКО на основе предоставленного контекста.

Твой ответ ОБЯЗАН быть валидным JSON следующей структуры:
{
  "answer": "развёрнутый ответ на вопрос",
  "sources": ["salesbeat.md / ## Раздел (chunk: salesbeat.md::structural::3)", ...],
  "quotes": ["дословная цитата из контекста подтверждающая ответ", ...]
}

Правила:
- answer: подробный ответ на русском языке
- sources: список источников — копируй метку чанка ПОЛНОСТЬЮ включая часть "(chunk: ...)"
- quotes: 1-3 дословные цитаты из контекста которые ПРЯМО содержат ответ на вопрос или ключевые факты ответа
- Если в контексте нет ответа на вопрос — верни:
  {"answer": "НЕ ЗНАЮ", "sources": [], "quotes": []}
- Никакого текста вне JSON. Только JSON.\
"""

RAG_TEMPLATE = """\
Контекст:

{context}

Вопрос: {question}
"""

# ── Вспомогательные функции ───────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source  = chunk.get("source", "")
        section = chunk.get("section", "")
        chunk_id = chunk.get("chunk_id", "")
        label = f"[{i}] {source}"
        if section:
            label += f" / {section}"
        if chunk_id:
            label += f" (chunk: {chunk_id})"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _parse_structured(raw: str) -> dict:
    """Парсит JSON из ответа модели. При ошибке возвращает fallback."""

    raw = raw.strip()
    # убираем возможные markdown-блоки ```json ... ```
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        return {

            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "quotes":  data.get("quotes", []),
        }
    except (json.JSONDecodeError, Exception):
        return {"answer": raw, "sources": [], "quotes": []}


# ── Основные функции ──────────────────────────────────────────────────────────

def answer_without_rag(question: str, model: str, api_url: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_NO_RAG},
        {"role": "user", "content": question},
    ]
    return chat(messages, model=model, api_url=api_url)


def answer_with_rag(
    question: str,
    query_embedding: list[float],
    index: list[dict],
    model: str,
    api_url: str,
    top_k: int = 20,
    window: int = 3,
    rerank_top_k: int = 5,
    dont_know_threshold: float | None = None,
) -> dict:
    """
    RAG с LLM-реранкингом и структурированным выводом.

    Возвращает dict:
      answer          — текст ответа
      sources         — список источников ["файл / раздел", ...]

      quotes          — список цитат из контекста
      chunks          — финальные чанки после реранкинга
      dont_know       — True если реранкер не нашёл ничего выше порога
    """
    # Первичный поиск
    candidates = search(query_embedding, index, top_k=top_k, window=window)

    # LLM реранкинг
    candidates = rerank_llm(query=question, chunks=candidates, model=model, api_url=api_url)
    candidates = candidates[:rerank_top_k]

    # Режим "не знаю" — если лучший чанк ниже порога
    if dont_know_threshold is not None:

        best_score = candidates[0]["rerank_score"] if candidates else 0.0

        if best_score < dont_know_threshold:
            return {
                "answer": "Не могу ответить на этот вопрос на основе имеющихся документов. "
                          "Пожалуйста, уточните вопрос или обратитесь к другому источнику.",
                "sources": [],
                "quotes": [],

                "chunks": candidates,
                "dont_know": True,
            }

    # Генерация ответа
    context = _build_context(candidates)
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": prompt},
    ]
    raw = chat(messages, model=model, max_tokens=2048, api_url=api_url)

    structured = _parse_structured(raw)

    return {
        **structured,
        "chunks": candidates,
        "dont_know": structured["answer"].strip().upper() == "НЕ ЗНАЮ",
    }
