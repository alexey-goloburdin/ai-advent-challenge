# main.py
"""
CLI-чат: агент для сбора реквизитов компании.
Поддерживает три стратегии управления контекстом:
- sliding_window: только последние N сообщений
- sticky_facts: facts (key-value) + последние N сообщений
- branching: checkpoint'ы и независимые ветки диалога

Использование:
    python main.py --strategy sliding_window --window_size 6
    python main.py --strategy sticky_facts --window_size 4
    python main.py --strategy branching --window_size 10

Команды в чате:
    /help           — справка по командам
    /info           — информация о текущей стратегии
    /facts          — показать извлечённые факты (sticky_facts)
    /reset          — сбросить диалог
    
    # Только для branching:
    /checkpoint <name>              — создать checkpoint
    /branch <name> from <checkpoint> — создать ветку от checkpoint
    /switch <branch>                — переключиться на ветку
    /branches                       — список веток
    /checkpoints                    — список checkpoint'ов
"""

import argparse
import os
import sys
import time
import threading
import subprocess
import tempfile
from pathlib import Path

from agent import AgentConfig, CompanyRequisitesAgent
from context_strategies import BranchingStrategy, StickyFactsStrategy


# ============================================================================
# CLI Helpers
# ============================================================================

def spinner(stop_event: threading.Event) -> None:
    """Анимация ожидания."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{frames[i % len(frames)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.08)


def clear_line() -> None:
    """Очистить текущую строку."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def print_colored(text: str, color: str = "default") -> None:
    """Печать с цветом."""
    colors = {
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "gray": "\x1b[90m",
        "red": "\x1b[31m",
        "default": "",
    }
    reset = "\x1b[0m"
    prefix = colors.get(color, "")
    print(f"{prefix}{text}{reset}")


# ============================================================================
# Command Handlers
# ============================================================================

def handle_command(cmd: str, agent: CompanyRequisitesAgent) -> bool:
    """
    Обрабатывает команды /xxx.
    Возвращает True если команда обработана, False если это не команда.
    """
    cmd = cmd.strip()
    if not cmd.startswith("/"):
        return False

    parts = cmd[1:].split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "help":
        print_help()
    elif command == "info":
        print_colored(agent.get_strategy_info(), "cyan")
        p, c = agent.get_total_tokens()
        print_colored(f"Всего токенов: prompt={p}, completion={c}, total={p+c}", "gray")
    elif command == "facts":
        handle_facts(agent)
    elif command == "reset":
        agent.reset()
        print_colored("Диалог сброшен.", "yellow")
    elif command == "checkpoint":
        handle_checkpoint(agent, args)
    elif command == "branch":
        handle_branch(agent, args)
    elif command == "switch":
        handle_switch(agent, args)
    elif command == "branches":
        handle_list_branches(agent)
    elif command == "checkpoints":
        handle_list_checkpoints(agent)
    else:
        print_colored(f"Неизвестная команда: /{command}. Введите /help для справки.", "red")

    return True


def print_help() -> None:
    """Печать справки."""
    help_text = """
Команды:
  /help           — эта справка
  /info           — информация о стратегии и статистика токенов
  /facts          — показать извлечённые факты (только sticky_facts)
  /reset          — сбросить диалог

Команды для branching стратегии:
  /checkpoint <name>              — создать checkpoint
  /branch <name> from <checkpoint> — создать ветку от checkpoint
  /switch <branch>                — переключиться на ветку
  /branches                       — список веток
  /checkpoints                    — список checkpoint'ов

Специальный ввод:
  \\e              — открыть редактор для ввода
  @file.txt       — прочитать сообщение из файла
"""
    print_colored(help_text.strip(), "cyan")


def handle_facts(agent: CompanyRequisitesAgent) -> None:
    """Показать факты (для sticky_facts)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, StickyFactsStrategy):
        print_colored("Команда /facts доступна только для стратегии sticky_facts", "yellow")
        return

    facts = agent.get_facts()
    if not facts:
        print_colored("Факты пока не извлечены.", "gray")
        return

    print_colored("Извлечённые факты:", "green")
    for key, value in facts.items():
        print(f"  {key}: {value}")


def handle_checkpoint(agent: CompanyRequisitesAgent, args: str) -> None:
    """Создать checkpoint (для branching)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, BranchingStrategy):
        print_colored("Команда /checkpoint доступна только для стратегии branching", "yellow")
        return

    name = args.strip()
    if not name:
        print_colored("Укажите имя checkpoint: /checkpoint <name>", "red")
        return

    if agent.create_checkpoint(name):
        print_colored(f"Checkpoint '{name}' создан.", "green")
    else:
        print_colored(f"Не удалось создать checkpoint '{name}' (возможно, уже существует).", "red")


def handle_branch(agent: CompanyRequisitesAgent, args: str) -> None:
    """Создать ветку (для branching)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, BranchingStrategy):
        print_colored("Команда /branch доступна только для стратегии branching", "yellow")
        return

    # Парсим: <name> from <checkpoint>
    if " from " not in args:
        print_colored("Формат: /branch <name> from <checkpoint>", "red")
        return

    parts = args.split(" from ", 1)
    branch_name = parts[0].strip()
    checkpoint_name = parts[1].strip()

    if not branch_name or not checkpoint_name:
        print_colored("Формат: /branch <name> from <checkpoint>", "red")
        return

    if agent.create_branch(branch_name, checkpoint_name):
        print_colored(f"Ветка '{branch_name}' создана от checkpoint '{checkpoint_name}'.", "green")
    else:
        print_colored(
            f"Не удалось создать ветку (ветка уже есть или checkpoint не существует).",
            "red"
        )


def handle_switch(agent: CompanyRequisitesAgent, args: str) -> None:
    """Переключиться на ветку (для branching)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, BranchingStrategy):
        print_colored("Команда /switch доступна только для стратегии branching", "yellow")
        return

    branch_name = args.strip()
    if not branch_name:
        print_colored("Укажите имя ветки: /switch <branch>", "red")
        return

    if agent.switch_branch(branch_name):
        print_colored(f"Переключились на ветку '{branch_name}'.", "green")
        # Показываем последнее сообщение в ветке
        msgs = agent.get_persisted_messages()
        if msgs:
            last = msgs[-1]
            prefix = "Вы" if last["role"] == "user" else "Агент"
            print_colored(f"Последнее сообщение: {prefix}: {last['text'][:100]}...", "gray")
    else:
        print_colored(f"Ветка '{branch_name}' не найдена.", "red")


def handle_list_branches(agent: CompanyRequisitesAgent) -> None:
    """Список веток (для branching)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, BranchingStrategy):
        print_colored("Команда /branches доступна только для стратегии branching", "yellow")
        return

    branches = agent.list_branches()
    current = agent.get_current_branch()

    print_colored("Ветки:", "green")
    for b in branches:
        marker = " *" if b == current else ""
        info = strategy.get_branch_info(b)
        msg_count = info["message_count"] if info else "?"
        print(f"  {b}{marker} ({msg_count} сообщений)")


def handle_list_checkpoints(agent: CompanyRequisitesAgent) -> None:
    """Список checkpoint'ов (для branching)."""
    strategy = agent.get_strategy()
    if not isinstance(strategy, BranchingStrategy):
        print_colored("Команда /checkpoints доступна только для стратегии branching", "yellow")
        return

    checkpoints = agent.list_checkpoints()
    if not checkpoints:
        print_colored("Checkpoint'ов пока нет.", "gray")
        return

    print_colored("Checkpoint'ы:", "green")
    for cp in checkpoints:
        print(f"  {cp}")


# ============================================================================
# Input Helpers
# ============================================================================

def read_from_editor() -> str:
    """Открыть редактор для ввода."""
    editor = (os.getenv("EDITOR") or "").strip()
    if not editor:
        raise SystemExit("Не задана переменная окружения EDITOR (например: export EDITOR=vim)")

    fd, path = tempfile.mkstemp(prefix="cli_input_", suffix=".txt")
    os.close(fd)

    try:
        rc = subprocess.call(f'{editor} "{path}"', shell=True)
        if rc != 0:
            raise SystemExit(f"Редактор завершился с кодом {rc}")
        return Path(path).read_text(encoding="utf-8")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def read_from_file(cmd: str) -> str:
    """Прочитать из файла (@file.txt)."""
    path_str = cmd[1:].strip()

    if (path_str.startswith('"') and path_str.endswith('"')) or \
       (path_str.startswith("'") and path_str.endswith("'")):
        path_str = path_str[1:-1]

    path = Path(path_str).expanduser()
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")

    return path.read_text(encoding="utf-8")


# ============================================================================
# Main
# ============================================================================

def get_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="CLI-чат: агент для сбора реквизитов компании с разными стратегиями контекста"
    )

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--json_format", action="store_true", help="Финальный ответ в JSON")

    parser.add_argument(
        "--strategy",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "sticky_facts", "branching"],
        help="Стратегия управления контекстом"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Размер окна (количество сообщений)"
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default="memory.json",
        help="Путь к файлу состояния"
    )

    return parser.parse_args()


def print_restored_history(agent: CompanyRequisitesAgent) -> None:
    """Печать восстановленной истории."""
    msgs = agent.get_persisted_messages()
    if not msgs:
        return

    print_colored("Восстановленная история:", "cyan")
    print("-" * 60)
    for m in msgs[-10:]:  # последние 10
        prefix = "Вы" if m["role"] == "user" else "Агент"
        text = m["text"][:200] + "..." if len(m["text"]) > 200 else m["text"]
        print(f"{prefix}: {text}\n")
    if len(msgs) > 10:
        print_colored(f"... и ещё {len(msgs) - 10} сообщений ранее", "gray")
    print("-" * 60)
    print()


def main() -> None:
    args = get_args()

    # Проверяем переменные окружения
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_base_url:
        raise SystemExit("Не задана переменная окружения OPENAI_BASE_URL")
    if not openai_api_key:
        raise SystemExit("Не задана переменная окружения OPENAI_API_KEY")

    # Конфигурация
    config = AgentConfig(
        model=args.model,
        max_output_tokens=args.max_tokens,
        json_format=args.json_format,
        strategy=args.strategy,
        window_size=args.window_size,
        memory_path=args.memory_path,
    )

    # Создаём агента
    agent = CompanyRequisitesAgent(
        base_url=openai_base_url,
        api_key=openai_api_key,
        config=config,
    )

    # Приветствие
    strategy_names = {
        "sliding_window": "Sliding Window",
        "sticky_facts": "Sticky Facts",
        "branching": "Branching",
    }
    print_colored(
        f"Агент для сбора реквизитов компании\n"
        f"Стратегия: {strategy_names[args.strategy]} (window={args.window_size})\n"
        f"Введите /help для списка команд\n"
        f"Выход: Ctrl+D или Ctrl+C\n",
        "green"
    )

    # Показываем восстановленную историю
    print_restored_history(agent)

    # Основной цикл
    while True:
        try:
            # Показываем текущую ветку для branching
            branch_indicator = ""
            if args.strategy == "branching":
                branch = agent.get_current_branch()
                if branch and branch != "main":
                    branch_indicator = f"[{branch}] "

            raw = input(f"{branch_indicator}> ")
            raw = (raw or "").strip()

            # Специальный ввод
            if raw == r"\e":
                user_text = read_from_editor().strip()
            elif raw.startswith("@"):
                user_text = read_from_file(raw).strip()
            else:
                user_text = raw

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nВыход.")
            break

        if not user_text:
            continue

        # Проверяем команды
        if handle_command(user_text, agent):
            continue

        # Обычное сообщение — отправляем агенту
        stop_event = threading.Event()
        t = threading.Thread(target=spinner, args=(stop_event,), daemon=True)
        t.start()

        try:
            assistant_text = agent.reply(user_text)
        finally:
            stop_event.set()
            t.join(timeout=0.2)
            clear_line()

        print(assistant_text)

        # Статистика токенов
        stats = agent.get_last_token_stats()
        if stats:
            prompt_tokens, completion_tokens, total = stats
            print_colored(
                f"[tokens: in={prompt_tokens}, out={completion_tokens}, total={total}]",
                "gray"
            )


if __name__ == "__main__":
    main()
