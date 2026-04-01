#!/usr/bin/env python3
"""
PR Review Agent — Day 32 assignment.

Usage:

    python main.py https://github.com/fastapi/fastapi/pull/15269 gpt-4o

Environment variables required:
    OPENAI_API_KEY   — API key
    OPENAI_API_URL   — API base URL (default: https://api.openai.com/v1)

Optional:
    GITHUB_TOKEN     — GitHub token to avoid rate limiting on large repos
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from review.github import fetch_pr_info, clone_repo
from review.rag import collect_chunks, build_index
from review.agent import run_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI code review agent for GitHub PRs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pr_url",
        help="GitHub PR URL, e.g. https://github.com/fastapi/fastapi/pull/15269",

    )
    parser.add_argument(
        "model",
        help="OpenAI model name, e.g. gpt-4o or gpt-4o-mini",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output markdown file path (default: review_<owner>_<repo>_<pr>.md)",
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Directory to clone/use repo (default: temp dir)",
    )
    return parser.parse_args()


def check_env() -> None:

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    check_env()

    print("\n🔍 PR Review Agent")
    print("=" * 50)

    # ── Step 1: Fetch PR metadata + diff ─────────────────────────────────────
    print("\n[1/4] Fetching PR information...")
    pr_info = fetch_pr_info(args.pr_url)
    print(f"  ✓ PR #{pr_info.number}: {pr_info.title}")
    print(f"  ✓ Changed files: {len(pr_info.changed_files)}")
    print(f"  ✓ Diff size: {len(pr_info.diff):,} chars")

    # ── Step 2: Clone repository ──────────────────────────────────────────────
    print("\n[2/4] Cloning repository...")

    use_temp = args.repo_dir is None
    if use_temp:

        tmp_dir = tempfile.mkdtemp(prefix=f"pr_review_{pr_info.repo}_")
        repo_dir = os.path.join(tmp_dir, pr_info.repo)
    else:

        repo_dir = args.repo_dir
        Path(repo_dir).mkdir(parents=True, exist_ok=True)

    try:
        clone_repo(pr_info.owner, pr_info.repo, repo_dir, pr_info.base_branch)
        print(f"  ✓ Repository available at: {repo_dir}")


        # ── Step 3: Build RAG index ───────────────────────────────────────────
        print("\n[3/4] Building RAG index...")
        chunks = collect_chunks(repo_dir)
        if not chunks:

            print("  ⚠ No chunks found — review will proceed without RAG context.")
            collection = None
        else:
            collection = build_index(chunks)
            print(f"  ✓ Index built: {len(chunks)} chunks")

        # ── Step 4: Run review ────────────────────────────────────────────────
        print("\n[4/4] Running AI review...")

        output_path = args.output or f"review_{pr_info.owner}_{pr_info.repo}_{pr_info.number}.md"

        run_review(
            pr_info=pr_info,
            collection=collection,
            model=args.model,
            output_path=output_path,
        )

        print("\n✅ Done!")


    finally:
        if use_temp:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
