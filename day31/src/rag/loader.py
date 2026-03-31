import json
import pathlib


def load_documents(project_path: str) -> list[dict]:
    """Читает README.md и docs/*.md из проекта."""
    root = pathlib.Path(project_path)
    docs = []

    candidates = [root / "README.md"]
    docs_dir = root / "docs"
    if docs_dir.exists():
        candidates += sorted(docs_dir.glob("*.md"))

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        chunks = _split_by_sections(text, source=str(path))
        docs.extend(chunks)

    return docs


def _split_by_sections(text: str, source: str) -> list[dict]:
    """Разбивает markdown на чанки по заголовкам H1/H2."""
    lines = text.splitlines()
    chunks = []
    current_title = source
    current_lines = []

    for line in lines:
        if line.startswith("## ") or line.startswith("# "):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({"source": source, "title": current_title, "content": content})
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({"source": source, "title": current_title, "content": content})

    return chunks if chunks else [{"source": source, "title": source, "content": text.strip()}]
