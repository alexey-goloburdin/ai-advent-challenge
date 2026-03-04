"""
CLI-чат: агент для сбора реквизитов компании с трёхслойной моделью памяти.

Команды:
  /memory     — показать состояние всех слоёв памяти
  /short      — показать краткосрочную память (историю диалога)
  /working    — показать рабочую память (данные задачи)
  /long       — показать долговременную память (профиль пользователя)
  /reset      — сбросить рабочую память (начать новую задачу)
  /newsession — очистить краткосрочную память (новая сессия)
  /help       — показать справку по командам
"""

import argparse
import json
import os
import sys
import time
import threading

from agent import AgentConfig, CompanyRequisitesAgent
from memory_manager import MemoryManager, MemoryConfig


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
        description="CLI-чат: агент для сбора реквизитов с трёхслойной памятью"
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--extractor_model", type=str, default="gpt-4o-mini",
                        help="Модель для экстрактора памяти (дешёвая)")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--json_format", action="store_true",
                        help="Финальный ответ будет в JSON")
    parser.add_argument("--context_messages", type=int, default=10,
                        help="Сколько сообщений включать в контекст LLM")

    # Профиль пользователя
    parser.add_argument("--profile", type=str, default=None,
                        help="Путь к .md файлу с предпочтениями пользователя")

    # Пути к файлам памяти
    parser.add_argument("--short_term_path", type=str, default="memory_short_term.json",
                        help="Файл краткосрочной памяти")
    parser.add_argument("--working_path", type=str, default="memory_working.json",
                        help="Файл рабочей памяти")
    parser.add_argument("--long_term_path", type=str, default="memory_long_term.json",
                        help="Файл долговременной памяти")

    return parser.parse_args()


def _print_memory_section(title: str, data: dict) -> None:
    """Красиво вывести секцию памяти."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_history(agent: CompanyRequisitesAgent) -> None:
    """Вывести историю из краткосрочной памяти."""
    history = agent.get_short_term_history()
    if not history:
        print("Краткосрочная память пуста.")
        return

    print(f"\n{'=' * 60}")
    print("  КРАТКОСРОЧНАЯ ПАМЯТЬ (история диалога)")
    print('=' * 60)
    for msg in history:
        role = msg["role"]
        text = msg["text"]
        ts = msg.get("timestamp", "")
        prefix = "Вы" if role == "user" else "Агент"
        print(f"\n[{ts}] {prefix}:")
        print(f"  {text}")
    print()


def _print_working(memory: MemoryManager) -> None:
    """Вывести рабочую память."""
    from dataclasses import asdict
    _print_memory_section(
        "РАБОЧАЯ ПАМЯТЬ (данные текущей задачи)",
        asdict(memory.working)
    )


def _print_long_term(memory: MemoryManager) -> None:
    """Вывести долговременную память."""
    from dataclasses import asdict
    _print_memory_section(
        "ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (профиль пользователя)",
        asdict(memory.long_term)
    )


def _print_help() -> None:
    print("""
Доступные команды:
  /memory     — показать состояние всех слоёв памяти
  /short      — показать краткосрочную память (историю диалога)
  /working    — показать рабочую память (данные задачи)
  /long       — показать долговременную память (профиль пользователя)
  /reset      — сбросить рабочую память (начать новую задачу)
  /newsession — очистить краткосрочную память (новая сессия)
  /help       — показать эту справку
""")


def _handle_command(cmd: str, agent: CompanyRequisitesAgent) -> bool:
    """
    Обработать команду. Возвращает True если команда обработана.
    """
    cmd = cmd.strip().lower()

    if cmd == "/memory":
        print(agent.get_memory_state())
        return True

    if cmd == "/short":
        _print_history(agent)
        return True

    if cmd == "/working":
        _print_working(agent.memory)
        return True

    if cmd == "/long":
        _print_long_term(agent.memory)
        return True

    if cmd == "/reset":
        agent.reset_task()
        print("Рабочая память сброшена. Можно начать сбор реквизитов заново.")
        return True

    if cmd == "/newsession":
        agent.new_session()
        print("Краткосрочная память очищена. Начата новая сессия.")
        return True

    if cmd == "/help":
        _print_help()
        return True

    return False


def main() -> None:
    args = _get_args()

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_base_url:
        raise SystemExit("Не задана переменная окружения OPENAI_BASE_URL")
    if not openai_api_key:
        raise SystemExit("Не задана переменная окружения OPENAI_API_KEY")

    # Загружаем профиль пользователя из файла
    user_profile_text = None
    if args.profile:
        from pathlib import Path
        profile_path = Path(args.profile)
        if not profile_path.exists():
            raise SystemExit(f"Файл профиля не найден: {args.profile}")
        user_profile_text = profile_path.read_text(encoding="utf-8")
        print(f"Загружен профиль из {args.profile}\n")

    config = AgentConfig(
        model=args.model,
        extractor_model=args.extractor_model,
        max_output_tokens=args.max_tokens,
        json_format=args.json_format,
        short_term_path=args.short_term_path,
        working_path=args.working_path,
        long_term_path=args.long_term_path,
        context_messages_count=args.context_messages,
        user_profile_text=user_profile_text,
    )

    agent = CompanyRequisitesAgent(
        base_url=openai_base_url,
        api_key=openai_api_key,
        config=config,
    )

    print(
        "Агент для сбора реквизитов компании с трёхслойной памятью.\n"
        "Команды: /memory, /short, /working, /long, /reset, /newsession, /help\n"
        "Выход: Ctrl+D (Linux/macOS) или Ctrl+Z+Enter (Windows).\n"
    )

    # Показываем текущее состояние памяти при старте
    history = agent.get_short_term_history()
    if history:
        print(f"Восстановлено {len(history)} сообщений из краткосрочной памяти.")
        print("Введите /short чтобы посмотреть историю.\n")

    working_data = agent.memory.working.collected_data
    if working_data:
        print(f"В рабочей памяти уже есть данные: {list(working_data.keys())}")
        print("Введите /working чтобы посмотреть детали.\n")

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

        # Проверяем команды
        if user_text.startswith("/"):
            if _handle_command(user_text, agent):
                continue
            else:
                print(f"Неизвестная команда: {user_text}. Введите /help для справки.")
                continue

        # Обычное сообщение — отправляем агенту
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
