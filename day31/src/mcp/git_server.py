import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("git-assistant")


@mcp.tool()
def git_branch() -> str:
    """Возвращает название текущей git-ветки."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or "detached HEAD"
    except subprocess.CalledProcessError as e:
        return f"Ошибка: {e.stderr.strip()}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
