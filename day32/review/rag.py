"""RAG indexer: AST-based chunking of Python code + Markdown docs."""

import ast
import json
import os
import urllib.request
from pathlib import Path

import chromadb


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize(text: str, max_chars: int = 2000) -> str:
    """Strip null bytes and control chars, truncate safely on char boundary."""
    KEEP = {chr(9), chr(10), chr(13)}  # tab, newline, carriage return
    text = "".join(c for c in text if c >= " " or c in KEEP)
    return text[:max_chars].strip()


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
            "content": _sanitize(chunk_text),  # guard against huge functions
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
            "content": _sanitize(source),
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
                        "content": _sanitize(text),
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
                "content": _sanitize(text),
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
            "content": _sanitize(source),
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

    # Final filter: remove empty or whitespace-only content
    chunks = [c for c in chunks if len(c["content"].strip()) >= 20]

    print(f"  → Collected chunks from {py_count} .py files and {md_count} .md files")
    print(f"  → Total chunks: {len(chunks)}")
    return chunks


# ── ChromaDB index ────────────────────────────────────────────────────────────

EMBED_BATCH = 32        # texts per API request (smaller = more reliable, more progress)
PARALLEL_REQUESTS = 8  # concurrent embedding requests


def _call_embed_api(texts: list[str], api_key: str, api_base: str) -> list[list[float]]:
    """Single embedding API call, returns vectors in input order."""
    payload = json.dumps({
        "model": "text-embedding-3-small",
        "input": [t.encode("utf-8", errors="replace").decode("utf-8") for t in texts],
    }).encode()
    url = f"{api_base.rstrip('/')}/embeddings"
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept-Encoding", "identity")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    data = json.loads(raw.decode())
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def _embed_batch(args: tuple) -> tuple[int, list]:
    """Embed one batch; returns (offset, embeddings).
    On HTTP 400 splits batch in half and retries each part separately.
    """
    import time
    import urllib.error
    offset, texts, api_key, api_base = args
    batch_num = offset // EMBED_BATCH + 1
    print(f"  → Batch {batch_num}: sending {len(texts)} chunks...", flush=True)

    for attempt in range(3):
        try:
            vectors = _call_embed_api(texts, api_key, api_base)
            return offset, vectors
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 400 and len(texts) > 1:
                # Split in half and recurse — isolates the bad chunk
                mid = len(texts) // 2
                print(f"  ⚠ Batch {batch_num} got 400, splitting {len(texts)}→{mid}+{len(texts)-mid}...", flush=True)
                _, left  = _embed_batch((offset,        texts[:mid],  api_key, api_base))
                _, right = _embed_batch((offset + mid,  texts[mid:],  api_key, api_base))
                return offset, left + right
            elif e.code == 400 and len(texts) == 1:
                # Single bad chunk — replace with zero vector and warn
                print(f"  ⚠ Skipping 1 bad chunk at offset {offset}: {body[:120]}", flush=True)
                # Return zero vector of standard dimension for text-embedding-3-small
                return offset, [[0.0] * 1536]
            if attempt == 2:
                raise
            print(f"  ⚠ Batch {batch_num} attempt {attempt+1} failed: {e}, retrying...", flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  ⚠ Batch {batch_num} attempt {attempt+1} failed: {e}, retrying...", flush=True)
            time.sleep(2 ** attempt)


def build_index(chunks: list[dict]) -> chromadb.Collection:
    """Embed all chunks in parallel and store in an in-memory ChromaDB collection."""
    from concurrent.futures import ThreadPoolExecutor, as_completed


    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")

    total = len(chunks)
    texts = [c["content"] for c in chunks]

    batch_args = [
        (offset, texts[offset:offset + EMBED_BATCH], api_key, api_base)

        for offset in range(0, total, EMBED_BATCH)
    ]

    print(f"  → Embedding {total} chunks in {len(batch_args)} batches "
          f"({PARALLEL_REQUESTS} parallel)...")


    embeddings: list = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as pool:

        futures = [pool.submit(_embed_batch, args) for args in batch_args]
        for future in as_completed(futures):
            offset, vectors = future.result()
            for j, v in enumerate(vectors):
                embeddings[offset + j] = v
            completed += len(vectors)
            batch_num = offset // EMBED_BATCH + 1
            print(f"  ✓ Batch {batch_num} done — {completed}/{total} chunks embedded", flush=True)

    print(f"  → Embedded {total}/{total}. Adding to index...    ")

    client = chromadb.Client()
    collection = client.create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"},
    )

    INSERT_BATCH = 500
    for i in range(0, total, INSERT_BATCH):
        b = chunks[i:i + INSERT_BATCH]
        collection.add(
            ids=[c["id"] for c in b],
            documents=[c["content"] for c in b],

            metadatas=[c["metadata"] for c in b],
            embeddings=embeddings[i:i + INSERT_BATCH],
        )

    print(f"  → Index built: {total} chunks.")
    return collection


def search(collection: chromadb.Collection, query: str, n: int = 5) -> list[dict]:
    """Return top-n most relevant chunks for a query."""
    # Embed query manually (collection has no embedding_function)
    _, query_embedding = _embed_batch((0, [query],
                                       os.environ["OPENAI_API_KEY"],
                                       os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")))
    results = collection.query(query_embeddings=[query_embedding[0]], n_results=n)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        chunks.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return chunks
