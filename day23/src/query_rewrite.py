from .llm import chat

SYSTEM_PROMPT = """\
Ты — ассистент для улучшения поисковых запросов.
Твоя задача — переформулировать вопрос пользователя так, чтобы он лучше \
совпадал с языком технической документации.
Верни ТОЛЬКО переформулированный запрос — без пояснений, без кавычек, без вводных фраз.\
"""


def rewrite_query(question: str, model: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},

    ]
    return chat(messages, model=model, max_tokens=256)
