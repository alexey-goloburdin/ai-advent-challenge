import argparse
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List


JSON_STOP_SEQUENCE = "<JSON_END>"


@dataclass
class Args:
    model: str
    max_tokens: int
    json_format: bool


def build_system_prompt(json_format: bool) -> str:
    prompt = (
        "Ты — агент для получения реквизитов компании.\n"
        "Веди диалог с пользователем так, чтобы собрать реквизиты и затем вернуть их ОДНИМ финальным сообщением.\n"
        "\n"
        "Нужно собрать и вернуть ТОЛЬКО следующий состав реквизитов:\n"
        "1) полное название юридического лица\n"
        "2) ИНН\n"
        "3) ОГРН или ОГРНИП (в зависимости от типа)\n"
        "4) юридический адрес\n"
        "5) подписант (ФИО + должность; кто подписывает договор)\n"
        "6) банковские реквизиты: наименование банка, БИК, расчётный счёт, корреспондентский счёт\n"
        "\n"
        "Правила:\n"
        "- Если каких-то полей не хватает, задавай уточняющие вопросы, пока не соберёшь всё.\n"
        "- Не выдумывай значения. Если пользователь не знает поле — попроси уточнить/проверить.\n"
        "- Когда все поля собраны, верни их одним финальным сообщением и НЕ задавай больше вопросов.\n"
        "- Не добавляй лишних пояснений, дисклеймеров или общих фраз.\n"
    )

    if json_format:
        prompt += (
            "\n"
            "ФИНАЛЬНЫЙ ОТВЕТ: строго валидный JSON без Markdown и без комментариев.\n"
            f"После JSON добавь стоп-последовательность {JSON_STOP_SEQUENCE}.\n"
            "До стоп-последовательности должен быть только JSON.\n"
            "\n"
            "JSON-ключи используй строго такие:\n"
            "{\n"
            '  "full_legal_name": "...",\n'
            '  "inn": "...",\n'
            '  "ogrn_or_ogrnip": "...",\n'
            '  "legal_address": "...",\n'
            '  "signatory": {"name": "...", "position": "..."},\n'
            '  "bank_details": {\n'
            '    "bank_name": "...",\n'
            '    "bik": "...",\n'
            '    "account_number": "...",\n'
            '    "correspondent_account": "..."\n'
            "  }\n"
            "}\n"
        )

    return prompt


def extract_assistant_text(resp: Dict[str, Any]) -> str:
    out = resp.get("output")
    if not isinstance(out, list) or not out:
        raise ValueError("Нет поля output в ответе или оно пустое")

    texts: List[str] = []
    for item in out:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t)

    if not texts:
        raise ValueError("Не удалось извлечь текст из output/content")

    return "\n".join(texts).strip()


def call_openai_responses(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    json_format: bool,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
    }
    if json_format:
        payload["stop"] = [JSON_STOP_SEQUENCE]

    json_data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/responses",
        data=json_data,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
            return extract_assistant_text(resp_json)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise SystemExit(f"HTTP-ошибка {e.code}: {body or '(без тела ответа)'}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"Ошибка сети: {e.reason}") from None
    except json.JSONDecodeError:
        raise SystemExit("Не удалось распарсить JSON-ответ от LLM") from None
    except Exception as e:
        raise SystemExit(
            f"Ошибка при отправке запроса к LLM: {type(e).__name__}: {e}"
        ) from None


def _spinner(stop_event: threading.Event) -> None:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    # Держим текст всегда на одной строке
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r{frame}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.08)


def _clear_current_line() -> None:
    # Стираем текущую строку
    sys.stdout.write("\r" + (" " * 120) + "\r")
    sys.stdout.flush()


def _get_args():
    parser = argparse.ArgumentParser(
        description="Консольный чат для сбора реквизитов компании через OpenAI Responses API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Название модели (по умолчанию: gpt-5.2)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Максимум токенов в ответе (по умолчанию: 1024)",
    )
    parser.add_argument(
        "--json_format",
        action="store_true",
        help="Финальный ответ будет в JSON",
    )

    args = parser.parse_args()
    return Args(
        model=args.model,
        max_tokens=args.max_tokens,
        json_format=args.json_format,
    )


def main() -> None:
    args = _get_args()

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_base_url:
        raise SystemExit("Не задана переменная окружения OPENAI_BASE_URL")
    if not openai_api_key:
        raise SystemExit("Не задана переменная окружения OPENAI_API_KEY")

    system_prompt = build_system_prompt(args.json_format)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}
    ]

    print("Это ассистент для получения реквизитов компании. Введи сообщение. Выход: Ctrl+D (Linux/macOS) или Ctrl+Z+Enter (Windows).")

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

        messages.append({"role": "user", "content": [{"type": "input_text", "text": user_text}]})

        # Спиннер на время ожидания ответа
        stop_event = threading.Event()
        t = threading.Thread(target=_spinner, args=(stop_event,), daemon=True)
        t.start()
        try:
            assistant_text = call_openai_responses(
                base_url=openai_base_url,
                api_key=openai_api_key,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                json_format=args.json_format,
            )
        finally:
            stop_event.set()
            # дать спиннеру шанс завершиться и очистить строку
            t.join(timeout=0.2)
            _clear_current_line()
        # Конец спиннера

        # Если в json_format модель вернула стоп, уберём его при выводе
        to_print = assistant_text
        if args.json_format and JSON_STOP_SEQUENCE in to_print:
            to_print = to_print.split(JSON_STOP_SEQUENCE, 1)[0].rstrip()

        print(to_print)

        messages.append({"role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})


if __name__ == "__main__":
    main()
