import argparse

from src.loader import load_documents
from src.chunkers import fixed_size_chunks, structural_chunks
from src.embedder import embed_chunks
from src.indexer import save_index
from src.compare import compare_strategies


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="День 21 — Индексация документов")
    p.add_argument("--docs",        default="documents",    help="Папка с документами")

    p.add_argument("--index-dir",   default="index",        help="Папка для индексов")
    p.add_argument("--chunk-size",  type=int, default=500,  help="Размер чанка (fixed)")
    p.add_argument("--overlap",     type=int, default=50,   help="Overlap (fixed)")
    p.add_argument("--model",       default="nomic-embed-text")
    p.add_argument("--strategy",    choices=["fixed", "structural", "both"], default="both")
    p.add_argument("--no-embed",    action="store_true",    help="Только chunking, без эмбеддингов")
    return p


def main():
    args = build_parser().parse_args()

    print("\n=== Загрузка документов ===")
    documents = load_documents(args.docs)
    if not documents:
        print("Документы не найдены. Положи файлы в папку documents/")
        return
    print(f"Загружено документов: {len(documents)}")

    all_fixed = []
    all_structural = []

    for doc in documents:
        if args.strategy in ("fixed", "both"):
            all_fixed.extend(fixed_size_chunks(doc, args.chunk_size, args.overlap))
        if args.strategy in ("structural", "both"):
            all_structural.extend(structural_chunks(doc))

    if args.strategy == "both":
        compare_strategies(all_fixed, all_structural)

    if args.no_embed:
        print("\n--no-embed: пропускаем генерацию эмбеддингов")
        return

    if all_fixed:
        print(f"\n=== Эмбеддинги: fixed ({len(all_fixed)} чанков) ===")
        records = embed_chunks(all_fixed, model=args.model)
        save_index(records, f"{args.index_dir}/fixed.json")

    if all_structural:
        print(f"\n=== Эмбеддинги: structural ({len(all_structural)} чанков) ===")
        records = embed_chunks(all_structural, model=args.model)
        save_index(records, f"{args.index_dir}/structural.json")

    print("\n✓ Готово")


if __name__ == "__main__":
    main()
