import argparse
import json
import os
import sys

DEFAULT_INDEX = "../day21/index/structural.json"
DEFAULT_QUESTIONS = "questions.json"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="День 23 — Реранкинг и фильтрация")
    p.add_argument("--index",            default=DEFAULT_INDEX,            help="Путь к structural.json")
    p.add_argument("--questions",        default=DEFAULT_QUESTIONS,        help="Путь к questions.json")
    p.add_argument("--model",            default="gpt-4o-mini",            help="OpenAI модель")
    p.add_argument("--top-k",            type=int,   default=20,           help="Кол-во кандидатов для первичного поиска")

    p.add_argument("--rerank-top-k",     type=int,   default=5,            help="Кол-во чанков после реранкинга")
    p.add_argument("--window",           type=int,   default=3,            help="Соседних чанков с каждой стороны")

    p.add_argument("--min-rerank-score", type=float, default=None,         help="Минимальный rerank score (например -5.0)")

    p.add_argument("--reranker-model",   default=DEFAULT_RERANKER,         help="Cross-encoder модель для реранкинга")
    p.add_argument("--no-rerank",        action="store_true",              help="Отключить реранкинг (только cosine)")
    p.add_argument("--rewrite",          action="store_true",              help="Включить query rewrite")
    p.add_argument("--ollama-host",      default="http://localhost:11434", help="Ollama host")
    p.add_argument("--embed-model",      default="nomic-embed-text",       help="Ollama модель для эмбеддингов")
    p.add_argument("--output",           default="results.json",           help="Куда сохранить результаты")
    p.add_argument("--llm-rerank", action="store_true", help="Использовать LLM вместо cross-encoder для реранкинга")
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


def format_chunks(chunks: list[dict], with_rerank: bool) -> list[str]:
    result = []

    for c in chunks:

        score_str = f"cosine={c['score']:.3f}"
        if with_rerank and "rerank_score" in c:
            score_str += f" rerank={c['rerank_score']:.2f}"
        result.append(f"{c['source']} / {c['section']} ({score_str})")
    return result


def main() -> None:
    args = build_parser().parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from src.retriever import load_index
    from src.rag import answer_without_rag, answer_with_rag

    reranker_model = None if args.no_rerank else args.reranker_model

    print("=== Загрузка индекса ===")

    index = load_index(args.index)

    print(f"  Чанков в индексе: {len(index)}")

    print("\n=== Загрузка вопросов ===")
    with open(args.questions, "r", encoding="utf-8") as f:

        questions = json.load(f)
    print(f"  Вопросов: {len(questions)}")

    print(f"\n=== Режим ===")
    print(f"  top-k (поиск):    {args.top_k}")
    print(f"  rerank top-k:     {args.rerank_top_k}")
    print(f"  window:           {args.window}")
    print(f"  реранкинг:        {'выключен (--no-rerank)' if args.no_rerank else reranker_model}")
    print(f"  query rewrite:    {'да' if args.rewrite else 'нет'}")
    print(f"  min rerank score: {args.min_rerank_score}")

    results = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q.get("expected", "")
        source = q.get("source", "")

        print(f"\n{'=' * 60}")
        print(f"Вопрос #{qid}: {question}")
        print(f"Ожидание: {expected}")
        print(f"Источник: {source}")

        result = {
            "id": qid,
            "question": question,
            "expected": expected,
            "expected_source": source,
        }


        # ── Без RAG ───────────────────────────────────────────────
        print("\n▶ Без RAG...")
        answer_plain = answer_without_rag(question, model=args.model)
        print(f"  {answer_plain}")

        result["answer_plain"] = answer_plain

        # ── Эмбеддинг вопроса ─────────────────────────────────────
        query_embedding = get_embedding(
            question,
            host=args.ollama_host,
            model=args.embed_model,
        )

        # ── С RAG без реранкинга ──────────────────────────────────
        print("\n▶ С RAG (только cosine)...")
        rag_plain = answer_with_rag(
            question=question,
            query_embedding=query_embedding,
            index=index,
            model=args.model,
            top_k=args.top_k,
            window=args.window,
            reranker_model=None,
            rerank_top_k=args.rerank_top_k,
            rewrite=False,
        )
        print(f"  {rag_plain['answer']}")
        plain_sources = format_chunks(rag_plain["chunks"][:args.rerank_top_k], with_rerank=False)
        print(f"\n  Чанки (cosine, top-{args.rerank_top_k}):")
        for s in plain_sources:
            print(f"    {s}")
        result["answer_rag_plain"] = rag_plain["answer"]
        result["sources_rag_plain"] = plain_sources

        # ── С RAG + реранкинг + query rewrite ────────────────────
        label = "С RAG + реранкинг"
        if args.rewrite:
            label += " + query rewrite"
        print(f"\n▶ {label}...")

        rag_full = answer_with_rag(
            question=question,
            query_embedding=query_embedding,
            index=index,
            model=args.model,
            top_k=args.top_k,
            window=args.window,
            reranker_model=reranker_model,
            rerank_top_k=args.rerank_top_k,
            min_rerank_score=args.min_rerank_score,
            rewrite=args.rewrite,
        )

        if rag_full["rewritten_query"]:
            print(f"  Query rewrite: {rag_full['rewritten_query']}")


        print(f"  {rag_full['answer']}")
        full_sources = format_chunks(rag_full["chunks"], with_rerank=not args.no_rerank)
        print(f"\n  Чанки после реранкинга ({len(full_sources)}):")

        for s in full_sources:
            print(f"    {s}")

        result["answer_rag_reranked"] = rag_full["answer"]

        result["sources_rag_reranked"] = full_sources
        result["rewritten_query"] = rag_full["rewritten_query"]

        results.append(result)

    # ── Сохранение ────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результаты сохранены → {args.output}")

    # ── Итоговая таблица ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ИТОГ")
    print(f"{'=' * 60}")
    print(f"{'#':<4} {'Вопрос':<40} {'Cosine':^8} {'Rerank':^8}")
    print("-" * 60)
    for r in results:
        exp_src = r["expected_source"].lower().split("/")[0].strip()

        found_plain = any(
            exp_src in s.lower()
            for s in r.get("sources_rag_plain", [])
        )
        found_reranked = any(
            exp_src in s.lower()
            for s in r.get("sources_rag_reranked", [])
        )
        q_short = r["question"][:39]
        print(

            f"{r['id']:<4} {q_short:<40} "

            f"{'✓' if found_plain else '✗':^8} "
            f"{'✓' if found_reranked else '✗':^8}"

        )


if __name__ == "__main__":
    main()
