"""
bench.py — бенчмарк оптимизации локальной LLM для задачи извлечения реквизитов

Использует нативный LM Studio API: POST /api/v1/chat
  Поля запроса (по документации lmstudio.ai/docs/developer/rest/chat):
    - model             : str
    - system_prompt     : str   (системный промпт)
    - input             : str   (сообщение пользователя — строка)
    - reasoning         : "off" (отключает reasoning для всех запросов)
    - store             : false (stateless, не сохранять историю)

    - temperature       : float
    - max_output_tokens : int   (не max_tokens!)
    - top_k             : int
    - repeat_penalty    : float (не presence_penalty!)

  Поля ответа:
    - output[].type == "message" -> output[].content
    - stats.input_tokens, stats.total_output_tokens, stats.tokens_per_second

Использование:
  python bench.py --url http://192.168.56.1:1234
  python bench.py --url http://192.168.56.1:1234 --model qwen/qwen3.5-9b --runs 3
  python bench.py --url http://192.168.56.1:1234 --profile after
  python bench.py --url http://192.168.56.1:1234 --verbose
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class Bank:
    name: str
    BIC: str
    current_account: str
    corporate_account: str



@dataclass
class Customer:
    name: str
    INN: str
    OGRN: str
    address: str
    signatory: str

    bank: Bank


TEST_CASES = [
    {
        "id": "simple",
        "text": (
            "ООО «Ромашка», ИНН 7701234567, ОГРН 1027700132195, "
            "юр. адрес: 125009, г. Москва, ул. Тверская, д. 1, "
            "директор Иванов Иван Иванович, "
            "банк: ПАО Сбербанк, БИК 044525225, р/с 40702810400000012345, к/с 30101810400000000225"
        ),
        "expected_inn": "7701234567",
    },
    {
        "id": "invoice",
        "text": (
            "Исполнитель: АО «Технологии Будущего»\n"
            "ИНН / КПП: 5010034271 / 501001001\n"

            "ОГРН: 1025000345678\n"
            "Адрес: 141700, Московская обл., г. Долгопрудный, пр-т Первый, д. 5\n"
            "Подписант: Генеральный директор Петров П.П.\n"
            "Банк получателя: ПАО «ВТБ», БИК: 044525187\n"
            "Расчётный счёт: 40702810700025023456\n"
            "Корр. счёт: 30101810700000000187"
        ),
        "expected_inn": "5010034271",
    },
    {
        "id": "noisy",
        "text": (
            "Договор оказания услуг №42 от 15 марта 2024 года.\n"
            "Заказчик: ООО «Горизонт», в лице генерального директора Сидорова А.В.\n"
            "Исполнитель: ИП Кузнецова Мария Александровна, ИНН 771234567890, "

            "ОГРНИП 318774600123456, адрес: 115093, г. Москва, ул. Павловская, д.18, кв.5, "
            "р/с 40802810800001234567 в АО «Альфа-Банк», БИК 044525593, "
            "к/с 30101810200000000593"
        ),
        "expected_inn": "771234567890",
    },
    {
        "id": "missing_fields",
        "text": (
            "Поставщик: ЗАО «Металлторг»\n"
            "ИНН: 6670123456\n"
            "Расчётный счёт: 40702810500000098765\n"

            "БИК: 046577674\n"
            "Банк: Уральский банк ПАО Сбербанк"
        ),
        "expected_inn": "6670123456",
    },
]

SYSTEM_BEFORE = "Извлеки реквизиты компании из текста и верни результат в формате JSON. Отвечай только JSON без пояснений."
USER_BEFORE   = "Извлеки реквизиты из следующего текста и верни JSON:\n\n{text}"

SYSTEM_AFTER = """Ты — парсер реквизитов российских компаний и ИП.
Твоя единственная задача: извлечь реквизиты из текста и вернуть ТОЛЬКО валидный JSON.

СТРОГИЕ ПРАВИЛА:
- Первый символ ответа — открывающая фигурная скобка {
- Последний символ ответа — закрывающая фигурная скобка }
- Никакого markdown, никаких ```, никаких пояснений до или после JSON
- Если поле отсутствует в тексте — значение пустая строка ""
- Не выдумывай данные, которых нет в тексте

Верни JSON точно с этими ключами:
{
  "name": "полное название юридического лица или ИП",
  "INN": "ИНН — 10 или 12 цифр",
  "OGRN": "ОГРН 13 цифр или ОГРНИП 15 цифр",
  "address": "юридический адрес",
  "signatory": "ФИО и должность подписанта",
  "bank": {
    "name": "название банка",
    "BIC": "БИК — 9 цифр",
    "current_account": "расчётный счёт — 20 цифр",
    "corporate_account": "корреспондентский счёт — 20 цифр"

  }
}
"""

USER_AFTER = "Извлеки реквизиты из текста и верни JSON:\n\n{text}"


PROFILES = {
    "before": {
        "label": "ДО оптимизации",
        "system": SYSTEM_BEFORE,
        "user_template": USER_BEFORE,
        "params": {
            "temperature": 0.7,
            "max_output_tokens": 2048,
        },

    },
    "after": {
        "label": "ПОСЛЕ оптимизации",
        "system": SYSTEM_AFTER,
        "user_template": USER_AFTER,
        "params": {
            "temperature": 0.1,
            "max_output_tokens": 512,
            "top_k": 20,
            "repeat_penalty": 1.1,
        },
    },
}


def call_lm_studio(base_url, model, system, user_text, params, timeout=120):
    """
    POST /api/v1/chat — нативный LM Studio API.
    Документация: lmstudio.ai/docs/developer/rest/chat
    """
    url = base_url.rstrip("/") + "/api/v1/chat"

    payload = {
        "model": model,
        "system_prompt": system,      # системный промпт
        "input": user_text,           # строка, не массив
        "reasoning": "off",           # отключаем reasoning
        "store": False,               # stateless
        **params,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio",
        },
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = ""
        try:
            err = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            "HTTP {} от LM Studio ({})\n  Payload: {}\n  Ответ:   {}".format(
                e.code, url,
                json.dumps(payload, ensure_ascii=False)[:400],
                err[:400],
            )
        ) from e
    except urllib.error.URLError as e:

        raise RuntimeError("Не удалось подключиться к LM Studio ({}): {}".format(url, e)) from e


    elapsed = time.perf_counter() - t0

    content = ""
    for block in body.get("output", []):
        if block.get("type") == "message":
            content = block.get("content", "")
            break

    s = body.get("stats", {})
    usage = {
        "prompt_tokens":     s.get("input_tokens", "?"),
        "completion_tokens": s.get("total_output_tokens", "?"),
        "tokens_per_second": s.get("tokens_per_second", 0),
    }
    return content, elapsed, usage


def evaluate(content, expected_inn):
    result = {"json_valid": False, "inn_correct": False, "has_bank": False, "score": 0, "parsed": None, "error": None}

    clean = content.strip()

    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(clean)
        result.update({"json_valid": True, "parsed": parsed, "score": 1})
    except json.JSONDecodeError as e:
        result["error"] = str(e)
        return result
    inn = parsed.get("INN") or parsed.get("inn") or ""

    if inn.strip() == expected_inn:
        result["inn_correct"] = True
        result["score"] += 1
    bank = parsed.get("bank") or {}
    if isinstance(bank, dict) and (bank.get("BIC") or bank.get("bic")):
        result["has_bank"] = True
        result["score"] += 1
    return result


RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"
RED = "\033[31m";  YELLOW = "\033[33m"; CYAN = "\033[36m"; DIM = "\033[2m"

def c(text, code): return "{}{}{}".format(code, text, RESET)



def print_header(text):
    print(); print(c("=" * 70, BOLD)); print(c("  " + text, BOLD + CYAN)); print(c("=" * 70, BOLD))



def print_result_row(case_id, ev, elapsed, usage, verbose):
    sc = ev["score"]
    sc_c = GREEN if sc == 3 else (YELLOW if sc >= 1 else RED)
    flags = [
        c("JSON✓", GREEN) if ev["json_valid"]  else c("JSON✗", RED),
        c("INN✓",  GREEN) if ev["inn_correct"] else c("INN✗",  RED),
        c("BANK✓", GREEN) if ev["has_bank"]    else c("BANK✗", RED),
    ]
    tps = usage.get("tokens_per_second", 0)
    tps_s = " ({:.1f} t/s)".format(tps) if tps else ""
    print("  {} {:<16} {}  {:.2f}s  prompt={} compl={}{}".format(

        c("[{}/3]".format(sc), sc_c), case_id, " ".join(flags),

        elapsed, usage.get("prompt_tokens","?"), usage.get("completion_tokens","?"), tps_s,
    ))
    if ev.get("error"):
        print(c("         JSON error: {}".format(ev["error"]), RED))
    if verbose and ev.get("parsed"):

        for line in json.dumps(ev["parsed"], ensure_ascii=False, indent=2).splitlines():
            print(c("    " + line, DIM))


def print_summary(profile_stats):

    print(); print(c("─" * 70, BOLD)); print(c("  ИТОГИ", BOLD)); print(c("─" * 70, BOLD))
    print("  {:<22} {:>6} {:>6} {:>6} {:>8} {:>8} {:>7}".format("Профиль","JSON","INN","BANK","Время","Токены","t/s"))
    print(c("  " + "-" * 66, DIM))
    rows = []
    for pn, st in profile_stats.items():
        t = st["total"] or 1
        rows.append((
            PROFILES[pn]["label"], st["json_ok"], st["inn_ok"], st["bank_ok"], st["total"],
            st["total_elapsed"]/t, st["total_tokens"]/t,
            st["total_tps"]/st["tps_count"] if st["tps_count"] else 0,
        ))
    for label, ok, inn_ok, bank_ok, total, avg_e, avg_t, avg_tps in rows:
        jc = GREEN if ok==total else (YELLOW if ok>0 else RED)

        ic = GREEN if inn_ok==total else (YELLOW if inn_ok>0 else RED)
        print("  {:<22} {}  {}  {}  {:>7.2f}s  {:>7.0f}  {:>6.1f}".format(
            label, c("{}/{}".format(ok,total),jc), c("{}/{}".format(inn_ok,total),ic),
            "{}/{}".format(bank_ok,total), avg_e, avg_t, avg_tps,
        ))
    keys = list(profile_stats.keys())
    if len(keys) == 2:
        eb = profile_stats[keys[0]]["total_elapsed"]
        ea = profile_stats[keys[1]]["total_elapsed"]
        if eb > 0:
            diff = (eb - ea) / eb * 100
            print(); print(c("  Оптимизированный профиль на {:.1f}% {}".format(abs(diff), "быстрее" if diff>0 else "медленнее"), BOLD))



def run_profile(profile_name, profile, args):
    stats = {"total":0,"json_ok":0,"inn_ok":0,"bank_ok":0,"total_elapsed":0.0,"total_tokens":0,"total_tps":0.0,"tps_count":0}
    print_header(profile["label"])
    print(c("  Модель: {}  |  Прогонов: {}".format(args.model, args.runs), DIM))
    print(c("  Параметры: {}".format(profile["params"]), DIM))

    for case in TEST_CASES:
        elapsed_runs = []; usage_last = {}; eval_last = {}; content_last = ""
        for run_i in range(args.runs):
            try:
                content, elapsed, usage = call_lm_studio(
                    args.url, args.model,
                    profile["system"],
                    profile["user_template"].format(text=case["text"]),
                    profile["params"], timeout=args.timeout,
                )
            except RuntimeError as e:
                print(c("  ОШИБКА [{}] run {}: {}".format(case["id"], run_i+1, e), RED))
                continue
            elapsed_runs.append(elapsed); usage_last = usage; content_last = content
            eval_last = evaluate(content, case["expected_inn"])

        if not elapsed_runs:
            continue

        avg_elapsed = sum(elapsed_runs)/len(elapsed_runs)
        comp = usage_last.get("completion_tokens", 0)
        tps  = usage_last.get("tokens_per_second", 0)
        stats["total"]         += 1
        stats["total_elapsed"] += avg_elapsed
        if isinstance(comp, (int, float)): stats["total_tokens"] += comp
        if tps: stats["total_tps"] += tps; stats["tps_count"] += 1
        if eval_last.get("json_valid"):  stats["json_ok"]  += 1
        if eval_last.get("inn_correct"): stats["inn_ok"]   += 1
        if eval_last.get("has_bank"):    stats["bank_ok"]  += 1


        print_result_row(case["id"], eval_last, avg_elapsed, usage_last, args.verbose)
        if args.verbose:
            print(c("    RAW: {}".format(content_last[:400]), DIM))

    return stats



def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк оптимизации LLM для извлечения реквизитов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",     required=True, help="Базовый URL LM Studio, например http://192.168.56.1:1234")
    parser.add_argument("--model",   default="qwen/qwen3.5-9b", help="Идентификатор модели")
    parser.add_argument("--profile", choices=["before","after","both"], default="both")
    parser.add_argument("--runs",    type=int, default=1, help="Прогонов на кейс")
    parser.add_argument("--timeout", type=int, default=120, help="Таймаут запроса (сек)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Печатать ответы модели")
    args = parser.parse_args()

    if not args.url.startswith("http"):
        parser.error("URL должен начинаться с http://")

    profiles_to_run = ["before","after"] if args.profile == "both" else [args.profile]
    all_stats = {}
    for pn in profiles_to_run:
        all_stats[pn] = run_profile(pn, PROFILES[pn], args)


    if len(all_stats) > 1:
        print_summary(all_stats)
    print()


if __name__ == "__main__":
    main()
