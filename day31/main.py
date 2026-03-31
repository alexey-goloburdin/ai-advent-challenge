import argparse
import sys

from src.rag.loader import load_documents
from src.rag.retriever import build_index, load_index, search
from src.mcp.client import GitMCPClient
from src.agent.assistant import ask

GIT_SERVER = "src/mcp/git_server.py"


def main():
    parser = argparse.ArgumentParser(description="Dev Assistant — Day 31")
    parser.add_argument("--project", required=True, help="Путь к проекту для индексирования")
    args = parser.parse_args()

    # 1. RAG: загрузить и переиндексировать документацию
    print(f"[RAG] Загружаем документацию из {args.project}...")
    docs = load_documents(args.project)
    if not docs:
        print("[RAG] Документы не найдены. Проверьте путь.")
        sys.exit(1)
    build_index(docs)
    index = load_index()
    print(f"[RAG] Готово: {len(index)} чанков в индексе.\n")

    # 2. MCP: запустить git-сервер
    print("[MCP] Запускаем git-сервер...")
    git_client = GitMCPClient(GIT_SERVER)
    print("[MCP] Готово.\n")

    print("Dev Assistant готов. Введите /help <вопрос> или 'exit' для выхода.\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break

            if not line:
                continue
            if line.lower() == "exit":
                break

            if line.startswith("/help"):
                question = line[5:].strip()
                if not question:
                    print("Использование: /help <вопрос о проекте>\n")
                    continue

                print()
                answer = ask(question, index, git_client, search)
                print(f"{answer}\n")

            else:
                print("Неизвестная команда. Используйте /help <вопрос>\n")

    finally:
        git_client.close()


if __name__ == "__main__":
    main()
