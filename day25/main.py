"""
RAG Chat CLI — День 25
Мини-чат с историей диалога, RAG-поиском и памятью задачи.

Запуск:
    python main.py --model gpt-4o-mini

Команды в чате:

    /quit или /exit  — выход

    /reset           — сброс истории и task_state
    /memory          — показать текущее состояние task_state
    /sources         — показать источники последнего ответа
    /help            — список команд
"""

import argparse
import os
import sys

from vector_store import VectorStore
from task_memory import TaskMemory
from chat import ChatEngine, _format_sources

# ── Цвета для терминала ────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"
BOLD = "\033[1m"

RESET = "\033[0m"


def print_banner():
    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════╗")
    print("║       RAG Chat  •  День 25           ║")
    print("║  /help для списка команд             ║")
    print("╚══════════════════════════════════════╝")
    print(RESET)


def print_help():
    print(f"{YELLOW}Команды:{RESET}")
    print("  /quit, /exit  — выход")
    print("  /reset        — новый диалог (сброс истории и памяти задачи)")
    print("  /memory       — показать текущую память задачи")
    print("  /sources      — источники последнего ответа")
    print("  /help         — эта справка")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Chat — мини-чат с памятью задачи"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Модель OpenAI API (например: gpt-4o-mini)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://172.27.112.1:11435",
        help="Хост Ollama для эмбеддингов (по умолчанию: http://172.27.112.1:11435)",
    )
    parser.add_argument(
        "--index",
        default="index/structural.json",
        help="Путь к файлу индекса (по умолчанию: index/structural.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Количество чанков для RAG (по умолчанию: 5)",
    )
    return parser.parse_args()


def get_env() -> tuple[str, str]:

    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            f"{YELLOW}Предупреждение: OPENAI_API_KEY не задан.{RESET}",
            file=sys.stderr,
        )
    return api_url, api_key


def main():
    args = parse_args()
    api_url, api_key = get_env()

    print_banner()
    print(f"{GRAY}Модель: {args.model} | Индекс: {args.index} | top-k: {args.top_k}{RESET}")
    print(f"{GRAY}Ollama: {args.ollama_host}{RESET}\n")

    # Инициализация
    print(f"{GRAY}Загрузка индекса...{RESET}", end=" ", flush=True)
    try:
        store = VectorStore(index_path=args.index, ollama_host=args.ollama_host)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{YELLOW}Ошибка загрузки индекса: {e}{RESET}")
        sys.exit(1)
    print(f"{GREEN}ОК{RESET}")

    memory = TaskMemory(api_url=api_url, api_key=api_key, model=args.model)
    engine = ChatEngine(
        api_url=api_url,
        api_key=api_key,
        model=args.model,
        vector_store=store,
        task_memory=memory,
        top_k=args.top_k,
    )

    last_chunks: list[dict] = []

    print(f"{GREEN}Чат готов. Введите вопрос или /help{RESET}\n")


    while True:
        try:
            user_input = input(f"{BOLD}Вы:{RESET} ").strip()

        except (EOFError, KeyboardInterrupt):
            print(f"\n{GRAY}До свидания!{RESET}")
            break

        if not user_input:
            continue

        # ── Команды ──────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit"):
            print(f"{GRAY}До свидания!{RESET}")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/reset":
            engine.reset()
            last_chunks = []
            print(f"{YELLOW}История и память задачи сброшены.{RESET}\n")
            continue

        if user_input.lower() == "/memory":
            block = memory.format_for_prompt()
            print(f"{CYAN}{block if block else 'Память задачи пуста.'}{RESET}\n")
            continue


        if user_input.lower() == "/sources":
            if last_chunks:
                print(f"{CYAN}📚 Источники последнего ответа: {_format_sources(last_chunks)}{RESET}")
                for i, c in enumerate(last_chunks, 1):
                    label = c["source"]
                    if c.get("section"):
                        label += f" / {c['section'].strip()}"
                    print(f"  [{i}] {label}  (score: {c['score']})")
            else:
                print(f"{GRAY}Источники недоступны — ещё не было ответов.{RESET}")
            print()
            continue

        # ── Основной запрос ───────────────────────────────────────

        print(f"{GRAY}[Поиск в базе знаний...]{RESET}", end=" ", flush=True)
        try:
            response, last_chunks = engine.send(user_input)
        except RuntimeError as e:
            print(f"\n{YELLOW}Ошибка: {e}{RESET}\n")
            continue
        if last_chunks:
            print(f"{GRAY}найдено {len(last_chunks)} релевантных фрагментов{RESET}")
        else:
            print(f"{GRAY}релевантных фрагментов нет (score < 0.3){RESET}")

        # Ответ ассистента
        print(f"\n{BOLD}{GREEN}Ассистент:{RESET}")
        print(response)

        # Источники (всегда, после каждого ответа)
        print(f"\n{CYAN}📚 Источники: {_format_sources(last_chunks)}{RESET}")

        # Состояние памяти задачи (краткое)
        print(f"{GRAY}[{engine.turn_count} сообщений | {memory.summary()}]{RESET}\n")


if __name__ == "__main__":
    main()
