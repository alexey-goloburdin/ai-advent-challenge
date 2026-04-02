#!/usr/bin/env python3
"""
Day 33 — AI-ассистент поддержки пользователей.

Запуск:
    python agent.py --model gpt-4o

Флоу:
1. Индексируем docs/*.txt через OpenAI embeddings (RAG).
2. Запускаем MCP-сервер CRM как subprocess.
3. Агент спрашивает email клиента, подтягивает его данные и тикеты через MCP.
4. Чат-цикл: каждый запрос обогащается RAG-контекстом + tool calls к CRM.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Добавляем src/ в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_client import MCPClient
from src.rag import RAGIndex
from src.openai_client import chat_completion


# ── конфигурация ───────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
CRM_SERVER = Path(__file__).parent / "crm_server.py"

SYSTEM_PROMPT_TEMPLATE = """Ты — вежливый и компетентный ИИ-ассистент службы поддержки пользователей.

{customer_block}

Правила работы:
- Отвечай на русском языке, чётко и по делу.
- Используй инструменты CRM (get_user, search_user_by_email, get_ticket, list_user_tickets) когда нужно уточнить данные о клиенте или его тикетах.
- При ответе на технические вопросы опирайся на документацию (она будет передана в контексте каждого запроса).
- Если вопрос выходит за рамки твоей компетенции — предложи создать тикет или обратиться к специалисту.
- Не выдумывай информацию, которой нет в документации или CRM.
"""

CUSTOMER_KNOWN = """Текущий клиент:
- Имя: {name}
- Email: {email}
- Тариф: {plan}
- Дата регистрации: {registered}
- Открытые тикеты: {open_tickets}
"""

CUSTOMER_UNKNOWN = "Клиент не идентифицирован. При необходимости используй инструмент search_user_by_email."


# ── построение tools для OpenAI ───────────────────────────────────────────────

def mcp_tools_to_openai(mcp_tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            }
        }
        for t in mcp_tools
    ]


# ── обработка tool calls ───────────────────────────────────────────────────────

def execute_tool_calls(
    tool_calls: list[dict],
    mcp: MCPClient,
    messages: list[dict],
) -> None:
    """Выполняет все tool_calls и добавляет результаты в messages."""
    for tc in tool_calls:
        func = tc["function"]
        name = func["name"]
        args = json.loads(func["arguments"])
        tool_call_id = tc["id"]

        print(f"\n  🔧 [{name}] {args}", flush=True)

        result = mcp.call_tool(name, args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })


# ── tool calling loop ──────────────────────────────────────────────────────────

def run_agent_turn(
    user_input: str,
    messages: list[dict],
    mcp: MCPClient,
    rag: RAGIndex,
    openai_tools: list[dict],
    model: str,
    api_url: str,
    api_key: str,
) -> str:
    """Один полный цикл агента (с возможными несколькими tool calls)."""

    # RAG: добавляем релевантные фрагменты документации в текущий запрос
    rag_chunks = rag.search(user_input, top_k=3)
    rag_context = "\n\n---\n".join(rag_chunks) if rag_chunks else ""

    user_message_content = user_input
    if rag_context:
        user_message_content = (
            f"{user_input}\n\n"
            f"[Релевантная документация]\n{rag_context}"
        )

    messages.append({"role": "user", "content": user_message_content})

    # Tool calling loop
    while True:
        response = chat_completion(
            messages=messages,
            model=model,
            api_url=api_url,
            api_key=api_key,
            tools=openai_tools,
        )

        # Добавляем ответ ассистента в историю
        messages.append(response["raw_message"])

        if response["tool_calls"]:
            execute_tool_calls(response["tool_calls"], mcp, messages)
            # Продолжаем цикл — LLM обработает результаты инструментов
            continue

        # Финальный текстовый ответ
        return response["content"] or "(нет ответа)"


# ── идентификация клиента ──────────────────────────────────────────────────────

def identify_customer(mcp: MCPClient) -> tuple[dict | None, list[dict]]:
    """
    Спрашивает email клиента, ищет в CRM, возвращает (user, tickets).
    """
    print("Введите email клиента (или нажмите Enter, чтобы пропустить): ", end="", flush=True)
    email = input().strip()

    if not email:
        return None, []

    result_str = mcp.call_tool("search_user_by_email", {"email": email})
    try:
        user = json.loads(result_str)
    except json.JSONDecodeError:
        print(f"  ⚠️  Ошибка разбора ответа CRM: {result_str}")
        return None, []

    if "error" in user:
        print(f"  ⚠️  {user['error']}")
        return None, []

    # Получаем тикеты
    tickets_str = mcp.call_tool("list_user_tickets", {"user_id": user["id"]})
    try:
        tickets = json.loads(tickets_str)
    except json.JSONDecodeError:
        tickets = []

    return user, tickets


def build_system_prompt(user: dict | None, tickets: list[dict]) -> str:
    if user is None:
        customer_block = CUSTOMER_UNKNOWN
    else:
        open_tickets = [t for t in tickets if t.get("status") != "resolved"]
        open_summary = ", ".join(
            f"[{t['id']}] {t['subject']} ({t['status']})"
            for t in open_tickets
        ) or "нет"

        customer_block = CUSTOMER_KNOWN.format(
            name=user["name"],
            email=user["email"],
            plan=user["plan"],
            registered=user["registered"],
            open_tickets=open_summary,
        )

        if tickets:
            ticket_details = "\n".join(
                f"  • [{t['id']}] {t['subject']} | статус: {t['status']} | приоритет: {t['priority']}\n"
                f"    Описание: {t['description']}"
                for t in tickets
            )
            customer_block += f"\nВсе тикеты клиента:\n{ticket_details}"

    return SYSTEM_PROMPT_TEMPLATE.format(customer_block=customer_block)


# ── главный цикл ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI-ассистент поддержки пользователей")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI модель (default: gpt-4o)")
    args = parser.parse_args()

    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Ошибка: не задана переменная окружения OPENAI_API_KEY")
        sys.exit(1)

    print("=" * 60)
    print("  🎧 AI-ассистент поддержки пользователей")
    print(f"  Модель: {args.model}")
    print("=" * 60)
    print()

    # 1. Индексируем документацию
    rag = RAGIndex(DOCS_DIR, api_url, api_key)

    # 2. Запускаем MCP CRM-сервер
    print("[MCP] Запускаем CRM-сервер...")
    mcp = MCPClient(CRM_SERVER)
    mcp_tools = mcp.list_tools()
    openai_tools = mcp_tools_to_openai(mcp_tools)
    print(f"[MCP] Доступные инструменты: {[t['name'] for t in mcp_tools]}\n")

    # 3. Идентифицируем клиента
    user, tickets = identify_customer(mcp)

    if user:
        print(f"\n✅ Клиент найден: {user['name']} ({user['plan']})")
        print(f"   Тикетов: {len(tickets)} всего, {sum(1 for t in tickets if t['status'] != 'resolved')} открытых\n")
    else:
        print("\n⚠️  Продолжаем без идентификации клиента\n")

    # 4. Строим system prompt с контекстом клиента
    system_prompt = build_system_prompt(user, tickets)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print("─" * 60)
    print("Чат начат. Введите 'выход' или нажмите Ctrl+C для завершения.")
    print("─" * 60)

    # 5. Основной чат-цикл
    try:
        while True:
            print()
            try:
                user_input = input("Вы: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in {"выход", "exit", "quit", "q"}:
                break

            print("\nАссистент: ", end="", flush=True)
            try:
                answer = run_agent_turn(
                    user_input=user_input,
                    messages=messages,
                    mcp=mcp,
                    rag=rag,
                    openai_tools=openai_tools,
                    model=args.model,
                    api_url=api_url,
                    api_key=api_key,
                )
                print(answer)
            except RuntimeError as e:
                print(f"\n❌ Ошибка: {e}")

    except KeyboardInterrupt:
        print("\n\nПрерывание...")
    finally:
        mcp.close()
        print("\nСессия завершена.")


if __name__ == "__main__":
    main()
