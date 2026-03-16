"""
MCP Server: Search
Инструменты для поиска информации в интернете через DuckDuckGo.
Транспорт: stdio
"""

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("search-server")



@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Поиск в интернете через DuckDuckGo.

    Args:
        query: поисковый запрос
        max_results: максимальное количество результатов (по умолчанию 5)


    Returns:
        Список результатов: заголовок, URL, краткое описание
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(

                f"### {r['title']}\n"
                f"URL: {r['href']}\n"
                f"{r['body']}\n"
            )

    if not results:
        return "Ничего не найдено по запросу."

    return "\n---\n".join(results)



@mcp.tool()
def search_news(query: str, max_results: int = 5) -> str:
    """

    Поиск свежих новостей через DuckDuckGo News.

    Args:

        query: поисковый запрос
        max_results: максимальное количество результатов (по умолчанию 5)

    Returns:
        Список новостей: заголовок, источник, дата, описание
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            results.append(
                f"### {r['title']}\n"
                f"Источник: {r['source']} | Дата: {r.get('date', 'н/д')}\n"
                f"URL: {r['url']}\n"
                f"{r['body']}\n"
            )

    if not results:
        return "Новостей не найдено."

    return "\n---\n".join(results)



if __name__ == "__main__":
    mcp.run(transport="stdio")
