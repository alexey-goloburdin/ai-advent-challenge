# main.py
import argparse
import os
import sys
import time

import threading

from agent import AgentConfig, CompanyRequisitesAgent


def _spinner(stop_event: threading.Event) -> None:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r{frame}")
        sys.stdout.flush()
        i += 1

        time.sleep(0.08)


def _clear_current_line() -> None:
    sys.stdout.write("\r" + (" " * 120) + "\r")
    sys.stdout.flush()


def _get_args():
    parser = argparse.ArgumentParser(
        description="CLI-чат: агент для сбора реквизитов компании через OpenAI Responses API"
    )
    parser.add_argument("--model", type=str, default="gpt-5.2")

    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--json_format", action="store_true", help="Финальный ответ будет в JSON")
    parser.add_argument("--max_history_turns", type=int, default=12, help="Сколько последних ходов хранить в контексте")
    parser.add_argument("--memory_path", type=str, default="memory.json", help="JSON-файл журнала сообщений")
    return parser.parse_args()


def _print_restored_history(agent: CompanyRequisitesAgent) -> None:
    msgs = agent.get_persisted_messages()
    if not msgs:
        return


    print("Восстановленная история из memory файла:")
    print("-" * 60)
    for m in msgs:
        role = m["role"]
        prefix = "Вы" if role == "user" else "Агент"

        print(f"{prefix}: {m['text']}\n")
    print("-" * 60)
    print()


def main() -> None:
    args = _get_args()

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_base_url:
        raise SystemExit("Не задана переменная окружения OPENAI_BASE_URL")
    if not openai_api_key:
        raise SystemExit("Не задана переменная окружения OPENAI_API_KEY")

    config = AgentConfig(
        model=args.model,
        max_output_tokens=args.max_tokens,
        json_format=args.json_format,

        max_history_turns=args.max_history_turns,
        memory_path=args.memory_path,
    )


    agent = CompanyRequisitesAgent(
        base_url=openai_base_url,
        api_key=openai_api_key,
        config=config,
    )

    print(
        "Это агент для получения реквизитов компании.\n"
        "Пиши сообщения. Выход: Ctrl+D (Linux/macOS) или Ctrl+Z+Enter (Windows).\n"
    )

    # Рендерим историю из файла при старте
    _print_restored_history(agent)


    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nВыход.")
            break

        if not user_text:
            continue

        stop_event = threading.Event()
        t = threading.Thread(target=_spinner, args=(stop_event,), daemon=True)
        t.start()
        try:
            assistant_text = agent.reply(user_text)
        finally:
            stop_event.set()
            t.join(timeout=0.2)
            _clear_current_line()

        print(assistant_text)


if __name__ == "__main__":
    main()
