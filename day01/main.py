import argparse
import json
import os
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(
        description="Программа отправляет промпт в OpenAI и выводит ответ"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Промпт"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Название модели (по умолчанию: gpt-5.2)"
    )
    args = parser.parse_args()

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    data = {
        "model": args.model,
        "input": args.prompt
    }
    json_data = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(f"{openai_base_url}/responses",
                                 data=json_data,
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {openai_api_key}")

    try:
        with urllib.request.urlopen(req) as response:
            llm_response = json.loads(
                response.read().decode()
            )["output"][0]["content"][0]["text"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP-ошибка {e.code}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"Ошибка сети: {e.reason}") from None
    except json.JSONDecodeError:
        raise SystemExit("Не удалось распарсить JSON-ответ от LLM")
    except Exception as e:
        raise SystemExit(f"Ошибка при отправке запроса к LLM: "
                         f"{type(e).__name__}: {e}") from None
    print(llm_response)


if __name__ == "__main__":
    main()
