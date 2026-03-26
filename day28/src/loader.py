import os
from pathlib import Path
from dataclasses import dataclass



@dataclass
class Document:
    text: str
    source: str      # путь к файлу
    title: str       # имя файла без расширения



def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        raise ImportError("Установи pypdf: pip install pypdf")


def load_documents(docs_dir: str) -> list[Document]:
    docs_path = Path(docs_dir)
    documents = []

    for path in sorted(docs_path.rglob("*")):
        if not path.is_file():

            continue

        ext = path.suffix.lower()

        try:

            if ext in (".md", ".txt", ".py", ".rst"):
                text = load_text(path)
            elif ext == ".pdf":
                text = load_pdf(path)
            else:
                continue

            if text.strip():
                documents.append(Document(
                    text=text,
                    source=str(path.relative_to(docs_path)),
                    title=path.stem,
                ))
                print(f"  Загружен: {path.name} ({len(text)} символов)")

        except Exception as e:
            print(f"  Ошибка при загрузке {path.name}: {e}")

    return documents
