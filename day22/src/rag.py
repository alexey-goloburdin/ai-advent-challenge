from .llm import chat
from .retriever import search

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
    top_k: int = 5,
    window: int = 0, 
) -> tuple[str, list[dict]]:
    chunks = search(query_embedding, index, top_k=top_k, window=window)
    context = _build_context(chunks)
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": prompt},
    ]
    answer = chat(messages, model=model)
    return answer, chunks
