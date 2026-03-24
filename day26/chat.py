import argparse
import json
import sys
import urllib.request
from urllib.error import URLError


def stream_chat(base_url: str, messages: list) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": "",  # LM Studio использует загруженную модель, model можно оставить пустым
        "messages": messages,

        "stream": True,
        "temperature": 0.2,
        "max_tokens": 4096,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


    full_response = ""
    try:
        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:

                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:

                        print(delta, end="", flush=True)
                        full_response += delta
                except json.JSONDecodeError:
                    pass
    except KeyboardInterrupt:
        print("\n[Прервано]")
    except URLError as e:
        print(f"\n[Ошибка подключения]: {e.reason}")

    print()  # перевод строки после ответа
    return full_response


def main():

    parser = argparse.ArgumentParser(description="Простой чат с LM Studio через streaming")
    parser.add_argument(
        "--url",
        default="http://localhost:1234",

        help="Базовый URL LM Studio (по умолчанию: http://localhost:1234)",
    )
    args = parser.parse_args()

    print(f"Чат с моделью по адресу {args.url}")
    print("Введите 'exit' или 'quit' для выхода\n")

    messages = []

    while True:
        try:
            sys.stdout.write("Вы: ")
            sys.stdout.flush()
            raw = sys.stdin.buffer.readline()
            if not raw:  # Ctrl+D
                print("\nВыход.")
                break
            user_input = raw.decode("utf-8").strip()
        except KeyboardInterrupt:
            print("\nПрервано. Выход.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Выход.")
            break

        messages.append({"role": "user", "content": user_input})

        print("Модель: ", end="", flush=True)
        response = stream_chat(args.url, messages)

        if response:
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
