import argparse
import json
import os
import sys

# day22 лежит рядом с day21, индекс берём оттуда
DEFAULT_INDEX = "../day21/index/structural.json"
DEFAULT_QUESTIONS = "questions.json"



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="День 22 — RAG evaluation")
    p.add_argument("--index",      default=DEFAULT_INDEX,    help="Путь к structural.json")
    p.add_argument("--questions",  default=DEFAULT_QUESTIONS, help="Путь к questions.json")
    p.add_argument("--model",      default="gpt-4o-mini",    help="OpenAI модель")
    p.add_argument("--top-k",      type=int, default=5,      help="Кол-во чанков для RAG")
    p.add_argument("--ollama-host",default="http://localhost:11434", help="Ollama host")
    p.add_argument("--embed-model",default="nomic-embed-text", help="Ollama модель для эмбеддингов")
    p.add_argument("--output",     default="results.json",   help="Куда сохранить результаты")
    p.add_argument("--window", type=int, default=3, help="Соседних чанков с каждой стороны (sentence-window)")
    return p


def get_embedding(text: str, host: str, model: str) -> list[float]:
    import json as _json
    import urllib.request

    payload = _json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data["embedding"]



def main() -> None:
    args = build_parser().parse_args()

    # Импорты из src/
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

        # Ответ без RAG
        print("\n▶ Без RAG...")
        answer_plain = answer_without_rag(question, model=args.model)
        print(f"  {answer_plain[:1000]}{'...' if len(answer_plain) > 1000 else ''}")

        # Эмбеддинг вопроса

        query_embedding = get_embedding(
            question,
            host=args.ollama_host,
            model=args.embed_model,
        )

        # Ответ с RAG
        print("\n▶ С RAG...")
        answer_rag, chunks = answer_with_rag(
            question,
            query_embedding=query_embedding,
            index=index,
            model=args.model,
            top_k=args.top_k,
            window=args.window,
        )
        print(f"  {answer_rag[:1000]}{'...' if len(answer_rag) > 1000 else ''}")

        # Найденные источники
        found_sources = [
            f"{c['source']} / {c['section']} (score={c['score']:.3f})"
            for c in chunks
        ]
        print("\n  Найденные чанки:")

        for s in found_sources:
            print(f"    {s}")

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "expected_source": source,
            "answer_plain": answer_plain,
            "answer_rag": answer_rag,
            "retrieved_sources": found_sources,
        })

    # Сохраняем результаты
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результаты сохранены → {args.output}")

    # Итоговая таблица
    print(f"\n{'=' * 60}")
    print("ИТОГ")
    print(f"{'=' * 60}")
    print(f"{'#':<4} {'Вопрос':<45} {'Источник найден?'}")
    print("-" * 60)
    for r in results:
        exp_src = r["expected_source"].lower()
        found = any(
            exp_src.split("/")[0].strip() in s.lower()
            for s in r["retrieved_sources"]

        )
        mark = "✓" if found else "✗"
        q_short = r["question"][:44]
        print(f"{r['id']:<4} {q_short:<45} {mark}")


if __name__ == "__main__":
    main()
