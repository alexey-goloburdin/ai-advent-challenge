"""RAG indexer: AST-based chunking of Python code + Markdown docs."""

import ast
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_python_file(source: str, filepath: str) -> list[dict]:
    """Extract top-level functions/classes (+ methods) as individual chunks."""
    try:

        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to whole-file chunk if unparseable
        return [{"content": source[:4000], "id": filepath, "metadata": {"file": filepath, "type": "file"}}]

    chunks = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Only top-level or class-level (skip nested functions inside functions)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            continue

        try:
            chunk_lines = source.splitlines()[node.lineno - 1: node.end_lineno]
            chunk_text = "\n".join(chunk_lines)
        except Exception:
            continue

        if len(chunk_text.strip()) < 30:
            continue

        name = node.name

        node_type = (
            "class" if isinstance(node, ast.ClassDef) else "function"
        )

        # Docstring
        docstring = ast.get_docstring(node) or ""

        chunk_id = f"{filepath}::{name}::{node.lineno}"
        chunks.append({
            "id": chunk_id,
            "content": chunk_text[:3000],  # guard against huge functions
            "metadata": {
                "file": filepath,
                "type": node_type,
                "name": name,
                "docstring": docstring[:300],
                "start_line": node.lineno,
            },
        })

    if not chunks:
        # Whole file as single chunk (e.g. only imports/constants)
        chunks.append({
            "id": filepath,
            "content": source[:3000],
            "metadata": {"file": filepath, "type": "file", "name": "", "docstring": ""},
        })

    return chunks


def _chunk_markdown_file(source: str, filepath: str) -> list[dict]:
    """Split Markdown by ## headings; each section = one chunk."""
    chunks = []
    current_heading = "intro"
    current_lines: list[str] = []

    counter = 0

    for line in source.splitlines():
        if line.startswith("## "):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if len(text) > 50:
                    chunk_id = f"{filepath}::{counter}::{current_heading}"
                    counter += 1
                    chunks.append({
                        "id": chunk_id,
                        "content": text[:3000],
                        "metadata": {
                            "file": filepath,
                            "type": "markdown",

                            "name": current_heading,
                            "docstring": "",
                        },
                    })

            current_heading = line.lstrip("# ").strip()
            current_lines = [line]

        else:
            current_lines.append(line)

    # Last section
    if current_lines:
        text = "\n".join(current_lines).strip()
        if len(text) > 50:
            chunk_id = f"{filepath}::{counter}::end"
            chunks.append({
                "id": chunk_id,
                "content": text[:3000],
                "metadata": {
                    "file": filepath,
                    "type": "markdown",
                    "name": current_heading,
                    "docstring": "",
                },
            })

    if not chunks:

        chunks.append({
            "id": filepath,
            "content": source[:3000],
            "metadata": {"file": filepath, "type": "markdown", "name": "", "docstring": ""},
        })

    return chunks


# ── File collection ───────────────────────────────────────────────────────────


_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build", "*.egg-info",
}


def collect_chunks(repo_dir: str) -> list[dict]:
    """Walk repo and collect all chunks from Python + selected Markdown files."""
    repo_path = Path(repo_dir)

    # git clone sometimes creates a subdirectory (e.g. target_dir/repo_name/)

    # detect this by checking if repo_path itself has no .py files but a subdir does
    subdirs = [d for d in repo_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not any(repo_path.rglob("*.py")) and len(subdirs) == 1:
        repo_path = subdirs[0]
        print(f"  → Detected repo root: {repo_path}")

    chunks: list[dict] = []
    py_count = md_count = 0

    # 1. All Python files in the repo
    for py_file in repo_path.rglob("*.py"):
        # Skip hidden/cache dirs — check only relative path parts
        rel_parts = py_file.relative_to(repo_path).parts
        if any(part in _SKIP_DIRS or part.endswith(".egg-info")

               for part in rel_parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")

        except Exception:
            continue


        rel = str(py_file.relative_to(repo_path))
        file_chunks = _chunk_python_file(source, rel)
        chunks.extend(file_chunks)
        py_count += 1

    # 2. README.md (top-level)

    for readme in repo_path.glob("README*.md"):
        try:
            source = readme.read_text(encoding="utf-8", errors="replace")
            rel = str(readme.relative_to(repo_path))
            chunks.extend(_chunk_markdown_file(source, rel))
            md_count += 1
        except Exception:
            pass

    # 3. docs/ru/docs/**/*.md
    docs_dir = repo_path / "docs" / "ru" / "docs"
    if docs_dir.exists():
        for md_file in docs_dir.rglob("*.md"):
            try:
                source = md_file.read_text(encoding="utf-8", errors="replace")
                rel = str(md_file.relative_to(repo_path))
                chunks.extend(_chunk_markdown_file(source, rel))
                md_count += 1
            except Exception:
                pass

    print(f"  → Collected chunks from {py_count} .py files and {md_count} .md files")
    print(f"  → Total chunks: {len(chunks)}")
    return chunks


# ── ChromaDB index ────────────────────────────────────────────────────────────

def build_index(chunks: list[dict]) -> chromadb.Collection:
    """Embed all chunks and store in an in-memory ChromaDB collection."""
    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")

    ef = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=api_base,
        model_name="text-embedding-3-small",
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="codebase",

        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


    # ChromaDB accepts max ~166 documents per batch to stay under token limits
    BATCH = 100
    total = len(chunks)

    for i in range(0, total, BATCH):
        batch = chunks[i: i + BATCH]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[c["metadata"] for c in batch],

        )
        print(f"  → Indexed {min(i + BATCH, total)}/{total} chunks...", end="\r")

    print(f"  → Indexed {total}/{total} chunks. Done.          ")
    return collection


def search(collection: chromadb.Collection, query: str, n: int = 5) -> list[dict]:
    """Return top-n most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=n)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return chunks
