import re
from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    strategy: str
    chunk_index: int
    total_chunks: int


# ─── Стратегия 1: Fixed-size ──────────────────────────────────────────────────


def fixed_size_chunks(
    doc: Document,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    text = doc.text
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append(Chunk(
                chunk_id=f"{doc.source}::fixed::{index}",
                text=chunk_text,
                source=doc.source,
                title=doc.title,
                section="",
                strategy="fixed",
                chunk_index=index,
                total_chunks=0,
            ))
            index += 1


        start += chunk_size - overlap

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


# ─── Стратегия 2: Structural (итеративная, без рекурсии) ─────────────────────

def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """
    Делит текст по паттерну.
    Возвращает список (section_name, section_text).
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]

    result = []
    boundaries = [m.start() for m in matches] + [len(text)]
    names = [m.group(0).strip() for m in matches]

    # текст до первого заголовка
    before = text[:boundaries[0]].strip()
    if before:
        result.append(("", before))

    for i, start in enumerate(boundaries[:-1]):
        section_text = text[start:boundaries[i + 1]]
        result.append((names[i], section_text))


    return result


def _fixed_split(text: str, max_size: int) -> list[str]:
    """Финальный уровень — просто режем по размеру."""
    pieces = []

    start = 0

    while start < len(text):
        piece = text[start:start + max_size].strip()
        if piece:
            pieces.append(piece)
        start += max_size
    return pieces


def structural_chunks(
    doc: Document,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
) -> list[Chunk]:
    chunks = []
    index = 0

    h1 = re.compile(r"^#\s+.+", re.MULTILINE)
    h2 = re.compile(r"^#{2,3}\s+.+", re.MULTILINE)

    # Уровень 1: делим по H1
    for h1_name, h1_text in _split_by_pattern(doc.text, h1):
        h1_text = h1_text.strip()
        if not h1_text:
            continue

        if len(h1_text) <= max_chunk_size:

            if len(h1_text) >= min_chunk_size:

                chunks.append(Chunk(
                    chunk_id=f"{doc.source}::structural::{index}",
                    text=h1_text,
                    source=doc.source,

                    title=doc.title,
                    section=h1_name,
                    strategy="structural",
                    chunk_index=index,
                    total_chunks=0,
                ))
                index += 1
            continue

        # Уровень 2: делим по H2
        for h2_name, h2_text in _split_by_pattern(h1_text, h2):
            h2_text = h2_text.strip()
            if not h2_text:
                continue

            section = f"{h1_name} / {h2_name}".strip(" /")

            if len(h2_text) <= max_chunk_size:
                if len(h2_text) >= min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"{doc.source}::structural::{index}",
                        text=h2_text,
                        source=doc.source,
                        title=doc.title,
                        section=section,
                        strategy="structural",
                        chunk_index=index,

                        total_chunks=0,
                    ))
                    index += 1

                continue

            # Уровень 3: делим по абзацам с буфером
            paragraphs = re.split(r"\n\n+", h2_text)
            buffer = ""
            part = 0

            for para in paragraphs:

                para = para.strip()

                if not para:
                    continue

                # Абзац сам по себе больше max — режем fixed
                if len(para) > max_chunk_size:
                    if buffer:
                        chunks.append(Chunk(
                            chunk_id=f"{doc.source}::structural::{index}",
                            text=buffer,
                            source=doc.source,
                            title=doc.title,
                            section=f"{section} (part {part + 1})",

                            strategy="structural",
                            chunk_index=index,
                            total_chunks=0,
                        ))
                        index += 1
                        part += 1
                        buffer = ""

                    for piece in _fixed_split(para, max_chunk_size):
                        chunks.append(Chunk(
                            chunk_id=f"{doc.source}::structural::{index}",
                            text=piece,
                            source=doc.source,

                            title=doc.title,
                            section=f"{section} (part {part + 1})",
                            strategy="structural",
                            chunk_index=index,
                            total_chunks=0,
                        ))
                        index += 1
                        part += 1

                    continue

                if len(buffer) + len(para) + 2 > max_chunk_size and buffer:
                    chunks.append(Chunk(

                        chunk_id=f"{doc.source}::structural::{index}",
                        text=buffer,
                        source=doc.source,

                        title=doc.title,
                        section=f"{section} (part {part + 1})",
                        strategy="structural",
                        chunk_index=index,
                        total_chunks=0,
                    ))
                    index += 1
                    part += 1

                    buffer = para
                else:
                    buffer = (buffer + "\n\n" + para).strip() if buffer else para

            if buffer and len(buffer) >= min_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{doc.source}::structural::{index}",
                    text=buffer,
                    source=doc.source,
                    title=doc.title,
                    section=f"{section} (part {part + 1})" if part > 0 else section,
                    strategy="structural",
                    chunk_index=index,
                    total_chunks=0,
                ))
                index += 1

    for chunk in chunks:
        chunk.total_chunks = len(chunks)


    return chunks
