# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
# Basic run
python main.py --lm-studio-url http://localhost:1234

# Skip non-RAG baseline comparison
python main.py --lm-studio-url http://localhost:1234 --no-plain

# Enable "don't know" mode with confidence threshold
python main.py --lm-studio-url http://localhost:1234 --dont-know-threshold 5.0
```

**Key CLI flags:** `--lm-studio-url` (LM Studio API base URL, default `http://localhost:1234`), `--model` (chat/rerank model, default `qwen/qwen3-5b`), `--embed-model` (embedding model, default `nomic-ai/nomic-embed-text-v1.5-GGUF`), `--index` (path to JSON index), `--top-k` (retrieval candidates, default 20), `--rerank-top-k` (final chunks, default 5), `--window` (sentence window neighbors, default 3), `--output` (results file, default `results.json`).

Requires LM Studio running with two models loaded: chat model (`qwen/qwen3-5b`) and embedding model (`nomic-ai/nomic-embed-text-v1.5-GGUF`). No API key needed, no Ollama.

## Architecture

RAG pipeline with LLM reranking:

```
Question → [Ollama] embed → [Retriever] cosine similarity (top-K)
         → [Reranker] LLM scores 0-10 each chunk → top rerank-K
         → [RAG] LLM generates {answer, sources, quotes} JSON
```

**`src/retriever.py`** — Cosine similarity search over pre-built index. Supports sentence-window expansion (fetches neighboring chunks to widen context).

**`src/reranker.py`** — LLM scores each candidate 0–10 for relevance. Expensive (one API call per chunk). Optional threshold filters low-confidence results to trigger "don't know" response.

**`src/rag.py`** — Orchestrates full pipeline: builds context string, calls LLM with structured JSON prompt, parses response. Answer includes `sources`, `quotes`, and `dont_know` flag.

**`src/llm.py`** — LM Studio API calls using `urllib` (no external HTTP library, no auth token).

**`src/embedder.py`** — Fetches embeddings from local Ollama (`nomic-embed-text`, 768-dim).

**`src/chunkers.py`** — Two strategies: fixed-size (500 chars, 50 overlap) and structural (hierarchical H1→H2→paragraph). Pre-built indexes live in `index/structural.json` and `index/fixed.json`.

**`src/compare.py`** — Runs chunking strategy comparison (structural vs fixed-size).

## Index Format

Each entry in the JSON index: `{chunk_id, text, source, section, embedding: [768 floats]}`. Indexes are gitignored and must exist before running (default path: `../day21/index/structural.json` or local `index/structural.json`).

## Output

Results written to `results.json` (gitignored). Each record includes `answer_rag`, `answer_plain`, `sources`, `quotes`, `dont_know`, per-chunk scores (`cosine_score`, `rerank_score`), and `checks` (boolean pass/fail for each output field).

All prompts and output are in Russian.
