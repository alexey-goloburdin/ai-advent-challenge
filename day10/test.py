# test_strategies.py
"""
Скрипт для автоматического тестирования и сравнения трёх стратегий
на одном и том же сценарии сбора реквизитов.

Запуск:
    python test_strategies.py

Результат:
    - Прогоняет сценарий на каждой стратегии
    - Выводит сравнительную таблицу
    - Сохраняет детальный отчёт в comparison_report.md
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from agent import AgentConfig, CompanyRequisitesAgent
from context_strategies import StickyFactsStrategy, BranchingStrategy


# ============================================================================
# Тестовый сценарий: сбор реквизитов за 12 сообщений
# ============================================================================

TEST_SCENARIO = [
    # 1. Начало
    "Привет, мне нужно заключить договор",
    
    # 2. Название компании
    "Компания называется ООО «Рога и Копыта»",
    
    # 3. ИНН
    "ИНН 7712345678",
    
    # 4. ОГРН  
    "ОГРН 1177746123456",
    
    # 5. Юридический адрес
    "Юридический адрес: 123456, г. Москва, ул. Тверская, д. 1, офис 100",
    
    # 6. Подписант
    "Договор подписывает генеральный директор Иванов Иван Иванович",
    
    # 7. Банк
    "Банк — Сбербанк",
    
    # 8. БИК
    "БИК 044525225",
    
    # 9. Расчётный счёт
    "Расчётный счёт 40702810938000012345",
    
    # 10. Корр. счёт
    "Корреспондентский счёт 30101810400000000225",
    
    # 11. Проверка памяти (важно для теста!)
    "Напомни, какой у нас ИНН и кто подписант?",
    
    # 12. Финал
    "Отлично, выведи все реквизиты",
]

# Ожидаемые значения для проверки
EXPECTED_VALUES = {
    "full_legal_name": "ООО «Рога и Копыта»",
    "inn": "7712345678",
    "ogrn": "1177746123456",
    "legal_address": "123456, г. Москва, ул. Тверская, д. 1, офис 100",
    "signatory_name": "Иванов Иван Иванович",
    "signatory_position": "генеральный директор",
    "bank_name": "Сбербанк",
    "bik": "044525225",
    "account_number": "40702810938000012345",
    "correspondent_account": "30101810400000000225",
}


# ============================================================================
# Result Structures
# ============================================================================

@dataclass
class TestResult:
    """Результат тестирования одной стратегии."""
    strategy: str
    window_size: int
    
    # Статистика токенов
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    facts_extraction_tokens: Tuple[int, int] = (0, 0)  # для sticky_facts
    
    # Качество
    remembered_inn: bool = False      # Помнит ИНН в msg 11?
    remembered_signatory: bool = False # Помнит подписанта в msg 11?
    final_contains_all: bool = False   # Финальный ответ содержит всё?
    
    # История ответов
    responses: List[str] = field(default_factory=list)
    
    # Дополнительно
    execution_time: float = 0.0
    error: str = ""


def check_response_contains(response: str, values: List[str]) -> bool:
    """Проверяет, что ответ содержит все указанные значения."""
    response_lower = response.lower()
    for val in values:
        if val.lower() not in response_lower:
            return False
    return True


# ============================================================================
# Test Runner
# ============================================================================

def run_test(
    strategy: str,
    window_size: int,
    base_url: str,
    api_key: str,
    scenario: List[str],
) -> TestResult:
    """Прогоняет сценарий на указанной стратегии."""
    
    result = TestResult(strategy=strategy, window_size=window_size)
    
    # Уникальный файл памяти для каждого теста
    memory_path = f"test_memory_{strategy}_{window_size}.json"
    
    # Удаляем старый файл если есть
    if Path(memory_path).exists():
        Path(memory_path).unlink()
    
    config = AgentConfig(
        model=os.getenv("TEST_MODEL", "gpt-4o-mini"),
        max_output_tokens=1024,
        json_format=False,
        strategy=strategy,
        window_size=window_size,
        memory_path=memory_path,
    )
    
    try:
        agent = CompanyRequisitesAgent(
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        
        start_time = time.time()
        
        for i, msg in enumerate(scenario, 1):
            print(f"  [{strategy}] Сообщение {i}/{len(scenario)}...", end=" ", flush=True)
            
            response = agent.reply(msg)
            result.responses.append(response)
            
            stats = agent.get_last_token_stats()
            if stats:
                result.total_prompt_tokens += stats[0]
                result.total_completion_tokens += stats[1]
            
            print("OK")
        
        result.execution_time = time.time() - start_time
        
        # Проверка качества
        
        # 1. Проверяем msg 11 (напомни ИНН и подписанта)
        if len(result.responses) >= 11:
            resp_11 = result.responses[10]
            result.remembered_inn = "7712345678" in resp_11
            result.remembered_signatory = "иванов" in resp_11.lower()
        
        # 2. Проверяем финальный ответ
        if result.responses:
            final = result.responses[-1]
            required = [
                "Рога и Копыта",
                "7712345678",
                "1177746123456",
                "Москва",
                "Иванов",
                "Сбербанк",
                "044525225",
                "40702810938000012345",
            ]
            result.final_contains_all = check_response_contains(final, required)
        
        # 3. Для sticky_facts — добавляем токены на извлечение
        if isinstance(agent.get_strategy(), StickyFactsStrategy):
            strat = agent.get_strategy()
            result.facts_extraction_tokens = strat.facts_extraction_tokens
        
    except Exception as e:
        result.error = str(e)
        print(f"ОШИБКА: {e}")
    
    finally:
        # Чистим временный файл
        if Path(memory_path).exists():
            Path(memory_path).unlink()
    
    return result


def run_branching_test(
    window_size: int,
    base_url: str,
    api_key: str,
) -> TestResult:
    """
    Специальный тест для branching:
    - Создаём checkpoint после сбора основных данных
    - Создаём две ветки с разными банками
    - Проверяем независимость веток
    """
    
    result = TestResult(strategy="branching", window_size=window_size)
    memory_path = f"test_memory_branching_{window_size}.json"
    
    if Path(memory_path).exists():
        Path(memory_path).unlink()
    
    config = AgentConfig(
        model=os.getenv("TEST_MODEL", "gpt-4o-mini"),
        max_output_tokens=1024,
        json_format=False,
        strategy="branching",
        window_size=window_size,
        memory_path=memory_path,
    )
    
    # Базовый сценарий до checkpoint
    base_messages = [
        "Привет, нужно заключить договор",
        "Компания ООО «Рога и Копыта», ИНН 7712345678",
        "ОГРН 1177746123456, адрес: Москва, Тверская 1",
        "Подписант — генеральный директор Иванов Иван Иванович",
    ]
    
    try:
        agent = CompanyRequisitesAgent(
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        
        start_time = time.time()
        
        # Базовые сообщения
        print("  [branching] Базовый диалог...", flush=True)
        for msg in base_messages:
            response = agent.reply(msg)
            result.responses.append(f"[main] {response[:100]}...")
            stats = agent.get_last_token_stats()
            if stats:
                result.total_prompt_tokens += stats[0]
                result.total_completion_tokens += stats[1]
        
        # Создаём checkpoint
        print("  [branching] Создаём checkpoint 'before_bank'...", flush=True)
        agent.create_checkpoint("before_bank")
        
        # Ветка A: Сбербанк
        print("  [branching] Ветка A: Сбербанк...", flush=True)
        agent.create_branch("sberbank", "before_bank")
        agent.switch_branch("sberbank")
        
        response_a = agent.reply("Банк Сбербанк, БИК 044525225, счёт 40702810938000012345, корр 30101810400000000225")
        result.responses.append(f"[sberbank] {response_a[:100]}...")
        stats = agent.get_last_token_stats()
        if stats:
            result.total_prompt_tokens += stats[0]
            result.total_completion_tokens += stats[1]
        
        response_a_final = agent.reply("Покажи все реквизиты")
        result.responses.append(f"[sberbank final] {response_a_final[:200]}...")
        stats = agent.get_last_token_stats()
        if stats:
            result.total_prompt_tokens += stats[0]
            result.total_completion_tokens += stats[1]
        
        # Ветка B: Тинькофф
        print("  [branching] Ветка B: Тинькофф...", flush=True)
        agent.create_branch("tinkoff", "before_bank")
        agent.switch_branch("tinkoff")
        
        response_b = agent.reply("Банк Тинькофф, БИК 044525974, счёт 40702810910000012345, корр 30101810145250000974")
        result.responses.append(f"[tinkoff] {response_b[:100]}...")
        stats = agent.get_last_token_stats()
        if stats:
            result.total_prompt_tokens += stats[0]
            result.total_completion_tokens += stats[1]
        
        response_b_final = agent.reply("Покажи все реквизиты")
        result.responses.append(f"[tinkoff final] {response_b_final[:200]}...")
        stats = agent.get_last_token_stats()
        if stats:
            result.total_prompt_tokens += stats[0]
            result.total_completion_tokens += stats[1]
        
        result.execution_time = time.time() - start_time
        
        # Проверки
        # Ветка sberbank должна содержать Сбербанк
        result.remembered_inn = "сбербанк" in response_a_final.lower()
        # Ветка tinkoff должна содержать Тинькофф
        result.remembered_signatory = "тинькофф" in response_b_final.lower()
        # Обе ветки помнят базовые данные
        result.final_contains_all = (
            "7712345678" in response_a_final and
            "7712345678" in response_b_final
        )
        
        print("  [branching] Тест завершён", flush=True)
        
    except Exception as e:
        result.error = str(e)
        print(f"ОШИБКА: {e}")
    
    finally:
        if Path(memory_path).exists():
            Path(memory_path).unlink()
    
    return result


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: List[TestResult]) -> str:
    """Генерирует Markdown-отчёт."""
    
    lines = [
        "# Сравнение стратегий управления контекстом",
        "",
        "## Сводная таблица",
        "",
        "| Стратегия | Window | Prompt Tokens | Completion Tokens | Total | Помнит ИНН? | Помнит подписанта? | Всё в финале? | Время (с) |",
        "|-----------|--------|---------------|-------------------|-------|-------------|-------------------|---------------|-----------|",
    ]
    
    for r in results:
        total = r.total_prompt_tokens + r.total_completion_tokens
        if r.strategy == "sticky_facts":
            total += sum(r.facts_extraction_tokens)
        
        lines.append(
            f"| {r.strategy} | {r.window_size} | {r.total_prompt_tokens} | "
            f"{r.total_completion_tokens} | {total} | "
            f"{'✅' if r.remembered_inn else '❌'} | "
            f"{'✅' if r.remembered_signatory else '❌'} | "
            f"{'✅' if r.final_contains_all else '❌'} | "
            f"{r.execution_time:.1f} |"
        )
    
    lines.extend([
        "",
        "## Детали по стратегиям",
        "",
    ])
    
    for r in results:
        lines.extend([
            f"### {r.strategy} (window={r.window_size})",
            "",
        ])
        
        if r.error:
            lines.append(f"**Ошибка:** {r.error}")
            lines.append("")
            continue
        
        if r.strategy == "sticky_facts":
            p, c = r.facts_extraction_tokens
            lines.append(f"**Дополнительные токены на извлечение фактов:** {p} + {c} = {p+c}")
            lines.append("")
        
        lines.append("**Ответы:**")
        lines.append("")
        for i, resp in enumerate(r.responses, 1):
            # Обрезаем длинные ответы
            short = resp[:300] + "..." if len(resp) > 300 else resp
            short = short.replace("\n", " ")
            lines.append(f"{i}. {short}")
        lines.append("")
    
    lines.extend([
        "## Выводы",
        "",
        "### Sliding Window",
        "- ✅ Простая реализация",
        "- ✅ Предсказуемый расход токенов",
        "- ❌ Теряет информацию из начала диалога",
        "",
        "### Sticky Facts",
        "- ✅ Сохраняет важные данные",
        "- ✅ Оптимальный расход токенов",
        "- ⚠️ Дополнительные вызовы LLM для извлечения фактов",
        "",
        "### Branching",
        "- ✅ Позволяет исследовать альтернативы",
        "- ✅ Независимые ветки диалога",
        "- ⚠️ Сложнее в реализации и использовании",
        "",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not base_url or not api_key:
        print("Установите переменные OPENAI_BASE_URL и OPENAI_API_KEY")
        sys.exit(1)
    
    print("=" * 60)
    print("Тестирование стратегий управления контекстом")
    print("=" * 60)
    print()
    
    results: List[TestResult] = []
    
    # Тест 1: Sliding Window с маленьким окном (6 сообщений)
    print("[1/5] Sliding Window (window=6)")
    result = run_test("sliding_window", 6, base_url, api_key, TEST_SCENARIO)
    results.append(result)
    print()
    
    # Тест 2: Sliding Window с большим окном (20 сообщений)
    print("[2/5] Sliding Window (window=20)")
    result = run_test("sliding_window", 20, base_url, api_key, TEST_SCENARIO)
    results.append(result)
    print()
    
    # Тест 3: Sticky Facts с маленьким окном
    print("[3/5] Sticky Facts (window=4)")
    result = run_test("sticky_facts", 4, base_url, api_key, TEST_SCENARIO)
    results.append(result)
    print()
    
    # Тест 4: Sticky Facts с большим окном
    print("[4/5] Sticky Facts (window=10)")
    result = run_test("sticky_facts", 10, base_url, api_key, TEST_SCENARIO)
    results.append(result)
    print()
    
    # Тест 5: Branching (специальный сценарий)
    print("[5/5] Branching (window=20)")
    result = run_branching_test(20, base_url, api_key)
    results.append(result)
    print()
    
    # Генерируем отчёт
    print("=" * 60)
    print("Генерация отчёта...")
    print("=" * 60)
    
    report = generate_report(results)
    
    # Сохраняем
    report_path = Path("comparison_report.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")
    
    # Выводим краткую таблицу в консоль
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"{'Стратегия':<20} {'Window':<8} {'Tokens':<10} {'ИНН?':<6} {'Подп?':<6} {'Всё?':<6}")
    print("-" * 60)
    
    for r in results:
        total = r.total_prompt_tokens + r.total_completion_tokens
        if r.strategy == "sticky_facts":
            total += sum(r.facts_extraction_tokens)
        
        print(
            f"{r.strategy:<20} {r.window_size:<8} {total:<10} "
            f"{'✅' if r.remembered_inn else '❌':<6} "
            f"{'✅' if r.remembered_signatory else '❌':<6} "
            f"{'✅' if r.final_contains_all else '❌':<6}"
        )


if __name__ == "__main__":
    main()
