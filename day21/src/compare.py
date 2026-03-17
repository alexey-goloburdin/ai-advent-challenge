from .chunkers import Chunk


def compare_strategies(
    fixed_chunks: list[Chunk],
    structural_chunks: list[Chunk],
) -> None:

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING")
    print("=" * 60)

    def stats(chunks: list[Chunk], name: str):
        sizes = [len(c.text) for c in chunks]
        sections = {c.section for c in chunks if c.section}
        sources = {c.source for c in chunks}
        print(f"\n▶ {name}")
        print(f"  Чанков всего:      {len(chunks)}")
        print(f"  Средний размер:    {sum(sizes) // len(sizes)} символов")
        print(f"  Мин / Макс:        {min(sizes)} / {max(sizes)} символов")
        print(f"  Источников:        {len(sources)}")
        print(f"  Уникальных section:{len(sections)}")

        if sections:

            examples = list(sections)[:3]
            print(f"  Примеры разделов:  {examples}")

    stats(fixed_chunks, "Fixed-size")
    stats(structural_chunks, "Structural")

    print("\n▶ Примеры чанков (первые 200 символов)")
    print("\n  [Fixed #0]")
    print(" ", fixed_chunks[0].text[:200].replace("\n", " "))
    print("\n  [Structural #0]")
    print(" ", structural_chunks[0].text[:200].replace("\n", " "))
    print("=" * 60)
