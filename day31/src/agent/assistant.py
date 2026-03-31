import json
import urllib.request


LM_STUDIO_CHAT_URL = "http://192.168.56.1:1234/v1/chat/completions"
CHAT_MODEL = "qwen/qwen3.5-9b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Поиск по документации проекта. Используй когда вопрос касается структуры проекта, API, установки, использования, архитектуры.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском или английском"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "Возвращает текущую git-ветку проекта.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def _chat(messages: list[dict], tools: list = None) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "reasoning": "off",
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        LM_STUDIO_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def ask(
    question: str,
    index: list[dict],
    git_client,
    retriever_search,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — ассистент разработчика. Отвечай на вопросы о проекте. "
                "Используй инструменты чтобы получить нужную информацию. "
                "Если ответа нет в документации — скажи об этом явно."
            )
        },
        {"role": "user", "content": question}
    ]

    # tool calling цикл
    while True:
        response = _chat(messages, tools=TOOLS)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        messages.append(message)

        if finish_reason == "tool_calls":
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"] or "{}")
                tool_id = tool_call["id"]

                print(f"  [tool] {name}({args})")

                if name == "search_docs":
                    results = retriever_search(args["query"], index, top_k=3)
                    result_text = "\n\n---\n\n".join(
                        f"[{c['source']} / {c['title']}]\n{c['content']}"
                        for c in results
                    )
                elif name == "git_branch":
                    result_text = git_client.git_branch()
                else:
                    result_text = f"Неизвестный инструмент: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                })

        else:
            # finish_reason == "stop" — финальный ответ
            return message["content"].strip()
