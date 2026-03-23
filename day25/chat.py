"""
Основной движок чата: история диалога, RAG-поиск, формирование промпта,
вызов LLM, вывод источников.
"""

import json
from urllib.request import urlopen, Request
from urllib.error import URLError

from vector_store import VectorStore
from task_memory import TaskMemory


SYSTEM_BASE = """Ты ассистент, который отвечает на основе предоставленного контекста из базы знаний.

Правила:
1. Используй информацию из раздела "Контекст из базы знаний".
   Если контекст содержит частично релевантные данные — используй их и укажи что именно нашёл.
   Пиши "В документации этого не нашёл" ТОЛЬКО если контекст совсем не касается вопроса.
   НЕ используй общие знания или догадки сверх того что есть в контексте.
2. НЕ добавляй в конец ответа блок с источниками — это делает система автоматически.
3. Учитывай историю диалога и память задачи при ответе."""


def _format_context(chunks: list[dict]) -> str:
    """Форматирует найденные чанки для вставки в промпт."""
    if not chunks:

        return "Релевантные фрагменты не найдены."
    parts = []
    for i, c in enumerate(chunks, 1):
        source_label = c["source"]
        if c.get("section"):
            source_label += f" / {c['section'].strip()}"
        parts.append(
            f"[{i}] {source_label} (score: {c['score']})\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)



def _format_sources(chunks: list[dict]) -> str:
    """Краткий список источников для отображения под ответом."""
    seen = []
    for c in chunks:
        label = c["source"]
        if c.get("section"):
            label += f" ({c['section'].strip()})"
        if label not in seen:
            seen.append(label)
    return ", ".join(seen) if seen else "источники не найдены"


class ChatEngine:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        vector_store: VectorStore,
        task_memory: TaskMemory,
        top_k: int = 5,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.vector_store = vector_store
        self.task_memory = task_memory
        self.top_k = top_k


        # История диалога в формате OpenAI: [{role, content}, ...]
        self.history: list[dict] = []


    def _call_llm(self, system: str, messages: list[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": 0.7,
                "max_tokens": 1500,
            }

        ).encode()


        req = Request(
            f"{self.api_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },

        )
        try:
            with urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        except URLError as e:
            raise RuntimeError(f"Ошибка обращения к LLM API: {e}") from e

    def send(self, user_message: str) -> tuple[str, list[dict]]:
        """
        Обработать сообщение пользователя.
        Возвращает (ответ_ассистента, список_использованных_чанков).

        """
        # 1. Расширяем запрос для RAG с учётом task_state
        rag_query = self.task_memory.build_rag_query(user_message)

        # 2. Ищем релевантные чанки (без порога — всегда передаём контекст)
        chunks = self.vector_store.search(rag_query, top_k=self.top_k)

        # 3. Формируем системный промпт

        task_block = self.task_memory.format_for_prompt()

        system_parts = [SYSTEM_BASE]
        if task_block:
            system_parts.append(task_block)


        context_block = _format_context(chunks)
        system_parts.append(

            f"=== Контекст из базы знаний ===\n{context_block}\n========================"
        )
        system = "\n\n".join(system_parts)


        # 4. Добавляем сообщение пользователя в историю
        self.history.append({"role": "user", "content": user_message})

        # 5. Вызываем LLM с полной историей
        response = self._call_llm(system, self.history)

        # 6. Добавляем ответ в историю
        self.history.append({"role": "assistant", "content": response})

        # 7. Обновляем task_memory в фоне (не блокирует UX)
        self.task_memory.update(user_message, response)

        return response, chunks

    def reset(self) -> None:
        """Сбросить историю и task_state."""
        self.history.clear()
        self.task_memory.state = {
            "goal": None,
            "clarified": [],
            "constraints": [],
            "open_questions": [],
        }

    @property
    def turn_count(self) -> int:

        """Количество завершённых пар user/assistant."""
        return sum(1 for m in self.history if m["role"] == "user")
