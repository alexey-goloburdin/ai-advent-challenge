import argparse
import json
import os
import sys

DEFAULT_INDEX = "../day21/index/structural.json"
DEFAULT_QUESTIONS = "questions.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="День 24 — Цитаты, источники и анти-галлюцинации")
    p.add_argument("--index",               default=DEFAULT_INDEX,            help="Путь к structural.json")
    p.add_argument("--questions",           default=DEFAULT_QUESTIONS,        help="Путь к questions.json")
    p.add_argument("--model",               default="gpt-4o-mini",            help="OpenAI модель")
    p.add_argument("--top-k",               type=int,   default=20,           help="Кол-во кандидатов для поиска")
    p.add_argument("--rerank-top-k",        type=int,   default=5,            help="Кол-во чанков после реранкинга")
    p.add_argument("--window",              type=int,   default=3,            help="Соседних чанков с каждой стороны")
    p.add_argument("--dont-know-threshold", type=float, default=None,         help="Порог rerank score для режима 'не знаю' (например 5.0)")
    p.add_argument("--ollama-host",         default="http://localhost:11434", help="Ollama host")
    p.add_argument("--embed-model",         default="nomic-embed-text",       help="Ollama модель для эмбеддингов")
    p.add_argument("--output",              default="results.json",           help="Куда сохранить результаты")
    p.add_argument("--no-plain",            action="store_true",              help="Не запускать режим без RAG")

    return p


def get_embedding(text: str, host: str, model: str) -> list[float]:
    import urllib.request

    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["embedding"]


def print_structured(result: dict) -> None:
    """Красивый вывод структурированного ответа."""
    if result.get("dont_know"):
        print(f"  ⚠️  НЕ ЗНАЮ: {result['answer']}")
        return

    print(f"  Ответ: {result['answer']}")

    if result.get("sources"):
        print("\n  Источники:")
        for s in result["sources"]:
            print(f"    • {s}")
    else:
        print("\n  Источники: ❌ отсутствуют")

    if result.get("quotes"):
        print("\n  Цитаты:")
        for q in result["quotes"]:
            print(f"    « {q[:200]}{'...' if len(q) > 200 else ''} »")
    else:
        print("\n  Цитаты: ❌ отсутствуют")


def check_result(result: dict) -> dict:
    """Автоматическая проверка качества ответа."""
    return {
        "has_answer":  bool(result.get("answer")) and not result.get("dont_know"),
        "has_sources": bool(result.get("sources")),
        "has_quotes":  bool(result.get("quotes")),
        "dont_know":   result.get("dont_know", False),
    }



def main() -> None:
    args = build_parser().parse_args()


    sys.path.insert(0, os.path.dirname(__file__))
    from src.retriever import load_index
    from src.rag import answer_without_rag, answer_with_rag


    print("=== Загрузка индекса ===")
    index = load_index(args.index)
    print(f"  Чанков в индексе: {len(index)}")

    print("\n=== Загрузка вопросов ===")
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Вопросов: {len(questions)}")

    print(f"\n=== Режим ===")
    print(f"  top-k (поиск):       {args.top_k}")
    print(f"  rerank top-k:        {args.rerank_top_k}")

    print(f"  window:              {args.window}")
    print(f"  dont-know threshold: {args.dont_know_threshold}")
    print(f"  без RAG:             {'нет (--no-plain)' if args.no_plain else 'да'}")

    results = []

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        expected = q.get("expected", "")
        source   = q.get("source", "")

        print(f"\n{'=' * 60}")
        print(f"Вопрос #{qid}: {question}")
        print(f"Ожидание: {expected}")
        print(f"Источник: {source}")

        # ── Без RAG ───────────────────────────────────────────────
        if not args.no_plain:
            print("\n▶ Без RAG...")
            answer_plain = answer_without_rag(question, model=args.model)
            print(f"  {answer_plain}")
        else:
            answer_plain = None

        # ── Эмбеддинг вопроса ─────────────────────────────────────
        query_embedding = get_embedding(

            question,

            host=args.ollama_host,
            model=args.embed_model,
        )

        # ── С RAG ─────────────────────────────────────────────────
        print("\n▶ С RAG (LLM rerank + структурированный ответ)...")
        rag_result = answer_with_rag(
            question=question,
            query_embedding=query_embedding,
            index=index,
            model=args.model,
            top_k=args.top_k,
            window=args.window,
            rerank_top_k=args.rerank_top_k,
            dont_know_threshold=args.dont_know_threshold,
        )
        print_structured(rag_result)

        # Чанки
        print(f"\n  Чанки после реранкинга ({len(rag_result['chunks'])}):")
        for c in rag_result["chunks"]:

            print(f"    {c['source']} / {c['section']} "
                  f"(cosine={c['score']:.3f} rerank={c.get('rerank_score', 0):.1f})")


        # Проверка качества
        checks = check_result(rag_result)
        print(f"\n  Проверка:")
        print(f"    ответ есть:    {'✓' if checks['has_answer']  else '✗'}")
        print(f"    источники:     {'✓' if checks['has_sources'] else '✗'}")
        print(f"    цитаты:        {'✓' if checks['has_quotes']  else '✗'}")
        if checks["dont_know"]:
            print(f"    режим НЕ ЗНАЮ: ✓")

        results.append({
            "id":            qid,
            "question":      question,
            "expected":      expected,
            "expected_source": source,
            "answer_plain":  answer_plain,
            "answer_rag":    rag_result["answer"],
            "sources":       rag_result["sources"],
            "quotes":        rag_result["quotes"],
            "dont_know":     rag_result["dont_know"],
            "chunks": [

                {
                    "source":       c["source"],
                    "section":      c["section"],
                    "cosine_score": c["score"],
                    "rerank_score": c.get("rerank_score", 0),
                }
                for c in rag_result["chunks"]
            ],

            "checks": checks,
        })

    # ── Сохранение ────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Результаты сохранены → {args.output}")

    # ── Итоговая таблица ──────────────────────────────────────────
    print(f"\n{'=' * 60}")

    print("ИТОГ")
    print(f"{'=' * 60}")
    print(f"{'#':<4} {'Вопрос':<35} {'Ответ':^6} {'Источн':^7} {'Цитаты':^7} {'!Знаю':^6}")
    print("-" * 60)
    for r in results:
        c = r["checks"]
        print(
            f"{r['id']:<4} {r['question'][:34]:<35} "
            f"{'✓' if c['has_answer']  else '✗':^6} "
            f"{'✓' if c['has_sources'] else '✗':^7} "
            f"{'✓' if c['has_quotes']  else '✗':^7} "
            f"{'✓' if c['dont_know']   else '·':^6}"
        )


    # Сводная статистика
    total = len(results)
    n_answer  = sum(1 for r in results if r["checks"]["has_answer"])
    n_sources = sum(1 for r in results if r["checks"]["has_sources"])
    n_quotes  = sum(1 for r in results if r["checks"]["has_quotes"])
    n_dk      = sum(1 for r in results if r["checks"]["dont_know"])

    print(f"\nСтатистика ({total} вопросов):")
    print(f"  Ответ есть:    {n_answer}/{total}")

    print(f"  Источники:     {n_sources}/{total}")
    print(f"  Цитаты:        {n_quotes}/{total}")
    print(f"  Режим НЕ ЗНАЮ: {n_dk}/{total}")


if __name__ == "__main__":
    main()
