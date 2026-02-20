import argparse
import json
import os
import time
import urllib.request
import urllib.error


MODELS = [
    "gpt-4o-mini",  # слабая
    "gpt-4.1",      # средняя
    "gpt-5",        # сильная
]

PRICES_PER_1K = {
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.00060,
    },
    "gpt-4.1": {
        "input": 0.003,
        "output": 0.012,
    },
    "gpt-5": {
        "input": 0.00125,
        "output": 0.010,
    },
}


def extract_text(resp: dict) -> str:
    out = resp.get("output") or []
    if not isinstance(out, list):
        return ""

    chunks = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for part in content:

            if not isinstance(part, dict):

                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            elif isinstance(part.get("text"), str):
                chunks.append(part["text"])

    return "\n".join(chunks).strip()


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES_PER_1K.get(model)
    if not p:
        return 0.0
    return (input_tokens / 1000.0) * p["input"] + (output_tokens / 1000.0) * p["output"]


def call_model(base_url: str, api_key: str, model: str, prompt: str, debug: bool):
    payload = {
        "model": model,
        "input": prompt,
    }

    req = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = f"HTTP {e.code}"
        if body:
            msg += f": {body}"
        raise RuntimeError(msg) from None
    except urllib.error.URLError as e:

        raise RuntimeError(f"Network error: {e.reason}") from None

    elapsed = time.perf_counter() - start

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Bad JSON response: {raw[:500]}") from None

    # Некоторые прокси любят возвращать {"error": null} — это НЕ ошибка
    err = data.get("error")
    if isinstance(err, dict):
        # нормальная ошибка API в теле 200
        raise RuntimeError(f"API error: {json.dumps(err, ensure_ascii=False)}") from None

    text = extract_text(data)
    usage = data.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cost = compute_cost_usd(model, input_tokens, output_tokens)

    if debug:
        return {
            "model": model,
            "text": text,
            "time": elapsed,
            "input_tokens": input_tokens,

            "output_tokens": output_tokens,
            "cost": cost,
            "raw": data,
        }

    return {
        "model": model,
        "text": text,
        "time": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Шлёт один prompt в 3 модели OpenAI и сравнивает время/токены/стоимость (urllib + Responses API)."
    )
    parser.add_argument("prompt", type=str, help="Промпт")
    parser.add_argument("--debug", action="store_true", help="Печатать сырой JSON при ошибках")
    args = parser.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit("Не задан OPENAI_API_KEY")
    if not base_url:
        raise SystemExit("Не задан OPENAI_BASE_URL и не удалось использовать дефолт")

    results = []

    for model in MODELS:
        print(f"\n=== Запрос к модели {model} ===")
        try:
            r = call_model(base_url, api_key, model, args.prompt, args.debug)
            results.append({"ok": True, **r})

            if r["text"]:
                print(r["text"])
            else:
                print("(пустой текстовый ответ)")

            print(f"\nВремя: {r['time']:.2f}s")
            print(f"Токены: {r['input_tokens']} → {r['output_tokens']}")
            print(f"Стоимость: ${r['cost']:.5f}")

        except Exception as e:
            results.append({"ok": False, "model": model, "error": f"{type(e).__name__}: {e}"})
            print(f"Ошибка {model}: {type(e).__name__}: {e}")
            if args.debug:
                # ничего дополнительно здесь не делаем; тело HTTP уже включено в ошибку
                pass

    print("\n\n===== ИТОГ =====")
    for r in results:
        if r.get("ok"):
            print(
                f"{r['model']:12} | "
                f"{r['time']:6.2f}s | "
                f"in:{r['input_tokens']:5} out:{r['output_tokens']:5} | "
                f"${r['cost']:.5f}"
            )

        else:
            print(f"{r['model']:12} | ERROR | {r['error']}")

if __name__ == "__main__":
    main()
