"""
Day 34 — File Assistant Agent

Chat mode (default):
  python main.py --project /path/to/project

Single task mode:
  python main.py --project /path/to/project "find all usages of MyClass"
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AI assistant for working with project files")
    parser.add_argument("task", nargs="?", default=None, help="Task in plain language (omit for chat mode)")
    parser.add_argument("--project", default=".", help="Path to the Python project root (default: current directory)")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_URL"):
        print("Error: OPENAI_API_URL is not set", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    project_root = str(Path(args.project).resolve())
    if not Path(project_root).is_dir():
        print(f"Error: project directory not found: {project_root}", file=sys.stderr)
        sys.exit(1)

    if args.task:
        from src.agent import run_agent
        run_agent(task=args.task, model=args.model, project_root=project_root)
    else:
        from src.agent import run_chat
        run_chat(model=args.model, project_root=project_root)


if __name__ == "__main__":
    main()
