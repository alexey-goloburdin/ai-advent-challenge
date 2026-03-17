import os
import json
import urllib.request
from .chunkers import Chunk


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/embeddings"
MODEL = "nomic-embed-text"



def get_embedding(text: str, model: str = MODEL) -> list[float]:
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["embedding"]


def embed_chunks(
    chunks: list[Chunk],
    model: str = MODEL,
    verbose: bool = True,

) -> list[dict]:
    result = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"  Эмбеддинг {i + 1}/{total}: {chunk.chunk_id[:60]}...")

        embedding = get_embedding(chunk.text, model=model)

        result.append({

            "chunk_id":    chunk.chunk_id,
            "text":        chunk.text,
            "source":      chunk.source,
            "title":       chunk.title,
            "section":     chunk.section,

            "strategy":    chunk.strategy,
            "chunk_index": chunk.chunk_index,
            "total_chunks":chunk.total_chunks,
            "embedding":   embedding,
        })

    return result
