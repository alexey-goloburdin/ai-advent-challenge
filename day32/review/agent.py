"""Review agent: assembles RAG context + diff, calls LLM, formats output."""

import json

import os
import urllib.request
from datetime import datetime

import chromadb

from .github import PRInfo
from .rag import search


# ── Diff parsing ──────────────────────────────────────────────────────────────

def _extract_diff_hunks(diff: str, max_chars: int = 12000) -> str:
    """Truncate diff to max_chars while keeping whole hunk headers."""
    if len(diff) <= max_chars:
        return diff

    return diff[:max_chars] + "\n\n... [diff truncated] ..."


def _extract_queries_from_diff(pr_info: PRInfo) -> list[str]:
    """
    Build RAG search queries from PR metadata.
    We query for:
      - each changed file path (to find related code)
      - PR title (semantic match)
      - changed module names
    """
    queries = []

    # PR title as semantic query
    queries.append(pr_info.title)

    # Each changed file → derive module name
    for f in pr_info.changed_files[:10]:
        queries.append(f)
        # e.g. "fastapi/routing.py" → "routing"
        stem = f.replace("/", " ").replace("_", " ").replace(".py", "").replace(".md", "")
        queries.append(stem)

    return list(dict.fromkeys(queries))  # deduplicate, preserve order


# ── RAG context assembly ──────────────────────────────────────────────────────

def _build_rag_context(collection, pr_info: PRInfo) -> str:
    """Run multiple queries and assemble deduplicated context block."""
    if collection is None:
        return "(RAG index is empty — no codebase context available)"
    queries = _extract_queries_from_diff(pr_info)
    seen_ids: set[str] = set()
    all_chunks: list[dict] = []

    for query in queries[:6]:  # limit total queries
        results = search(collection, query, n=4)
        for chunk in results:
            chunk_id = chunk["metadata"].get("file", "") + chunk["metadata"].get("name", "")
            if chunk_id not in seen_ids and chunk["score"] > 0.25:
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)

    if not all_chunks:
        return "(no relevant context found in codebase)"

    # Sort by relevance score descending
    all_chunks.sort(key=lambda c: c["score"], reverse=True)
    top_chunks = all_chunks[:8]

    lines = []
    for i, chunk in enumerate(top_chunks, 1):
        meta = chunk["metadata"]

        file_info = meta.get("file", "?")
        name_info = meta.get("name", "")
        type_info = meta.get("type", "")

        score = chunk["score"]
        header = f"[{i}] {file_info}"
        if name_info:
            header += f" :: {name_info}"
        header += f" ({type_info}, relevance: {score:.2f})"
        lines.append(header)
        lines.append(chunk["content"])
        lines.append("")

    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI-compatible API via urllib."""
    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    api_key = os.environ["OPENAI_API_KEY"]
    url = f"{api_url.rstrip('/')}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,

        "max_tokens": 4096,
    }).encode()


    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req) as resp:

        data = json.loads(resp.read().decode())

    return data["choices"][0]["message"]["content"]


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be given:
1. A GitHub Pull Request diff
2. Relevant context from the codebase (retrieved via RAG)

Your task is to perform a thorough code review and return a structured Markdown report.

Structure your review EXACTLY as follows:

## Summary
Brief 2-3 sentence overview of what this PR does.

## Potential Bugs

List specific bugs or logic errors you see. For each:
- Describe the problem clearly
- Reference the file and line from the diff
- Suggest a fix

If none found, write: *No potential bugs identified.*

## Architectural Issues

Concerns about design, patterns, coupling, or consistency with the existing codebase.
Reference the RAG context when relevant (e.g. "this differs from how X is done in Y").

If none found, write: *No architectural issues identified.*

## Recommendations
Improvements that are not bugs: performance, readability, testing, documentation.

## Conclusion
One-paragraph overall assessment with a recommendation: Approve / Request Changes / Needs Discussion.

---

Rules:
- Be specific and actionable. No vague comments.
- Reference file paths and line numbers from the diff when possible.
- Use the codebase context to check consistency with existing patterns.
- Write in English.
"""



def _build_user_prompt(pr_info: PRInfo, rag_context: str) -> str:

    diff_excerpt = _extract_diff_hunks(pr_info.diff)

    return f"""# PR #{pr_info.number}: {pr_info.title}

**Repository:** {pr_info.owner}/{pr_info.repo}
**Base branch:** {pr_info.base_branch} ← {pr_info.head_branch}

**Description:**
{pr_info.description or "(no description provided)"}

**Changed files ({len(pr_info.changed_files)}):**
{chr(10).join("- " + f for f in pr_info.changed_files)}

---

## Diff

```diff
{diff_excerpt}
```

---


## Relevant Codebase Context (RAG)

{rag_context}


---

Please provide your code review now.
"""



# ── Main entry ────────────────────────────────────────────────────────────────

def run_review(
    pr_info: PRInfo,

    collection: chromadb.Collection,
    model: str,
    output_path: str,
) -> str:
    """Run the full review pipeline, print to stdout, save to file."""

    print("  → Building RAG context...")
    rag_context = _build_rag_context(collection, pr_info)
    rag_chunks_count = rag_context.count("\n[")
    print(f"  → RAG context assembled ({rag_chunks_count} chunks retrieved)")

    print(f"  → Calling LLM ({model})...")
    user_prompt = _build_user_prompt(pr_info, rag_context)
    review_text = _call_llm(model, SYSTEM_PROMPT, user_prompt)

    # Build final markdown document
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""# Code Review: PR #{pr_info.number}

**{pr_info.title}**  

**Repository:** [{pr_info.owner}/{pr_info.repo}](https://github.com/{pr_info.owner}/{pr_info.repo})  
**PR URL:** https://github.com/{pr_info.owner}/{pr_info.repo}/pull/{pr_info.number}  
**Model:** {model}  
**Reviewed at:** {timestamp}  

---

"""
    full_output = header + review_text

    # Print to stdout
    print("\n" + "=" * 70)
    print(full_output)
    print("=" * 70 + "\n")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"  → Review saved to: {output_path}")

    return full_output
