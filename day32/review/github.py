"""GitHub utilities: parse PR URL, fetch diff, clone repo."""

import json

import os
import subprocess
import urllib.request

import urllib.error
from dataclasses import dataclass



@dataclass
class PRInfo:
    owner: str
    repo: str
    number: int
    title: str
    description: str
    diff: str
    base_branch: str
    head_branch: str
    changed_files: list[str]


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse https://github.com/owner/repo/pull/123 → (owner, repo, number)."""
    parts = url.rstrip("/").split("/")
    if len(parts) < 7 or parts[5] != "pull":
        raise ValueError(f"Invalid PR URL: {url}")
    owner = parts[3]
    repo = parts[4]
    number = int(parts[6])
    return owner, repo, number


def _github_api_request(path: str, token: str | None = None) -> dict:

    """Make a GitHub API request, return parsed JSON."""
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _fetch_diff(owner: str, repo: str, number: int, token: str | None = None) -> str:
    """Fetch raw unified diff for a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3.diff")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_pr_info(pr_url: str) -> PRInfo:
    """Fetch all PR metadata + diff from GitHub API."""
    owner, repo, number = parse_pr_url(pr_url)
    token = os.environ.get("GITHUB_TOKEN")

    print(f"  → Fetching PR #{number} from {owner}/{repo}...")

    pr_data = _github_api_request(f"/repos/{owner}/{repo}/pulls/{number}", token)
    diff = _fetch_diff(owner, repo, number, token)

    # Fetch list of changed files
    files_data = _github_api_request(
        f"/repos/{owner}/{repo}/pulls/{number}/files", token
    )
    changed_files = [f["filename"] for f in files_data]

    return PRInfo(
        owner=owner,
        repo=repo,
        number=number,

        title=pr_data["title"],
        description=pr_data.get("body") or "",
        diff=diff,
        base_branch=pr_data["base"]["ref"],
        head_branch=pr_data["head"]["ref"],
        changed_files=changed_files,
    )


def clone_repo(owner: str, repo: str, target_dir: str, base_branch: str) -> None:
    """Shallow-clone the repo at base branch into target_dir."""
    clone_url = f"https://github.com/{owner}/{repo}.git"
    print(f"  → Cloning {clone_url} (branch: {base_branch})...")

    if os.path.exists(os.path.join(target_dir, ".git")):
        print(f"  → Directory {target_dir} already exists, skipping clone.")
        return

    result = subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", base_branch,
            "--single-branch",
            clone_url,
            target_dir,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Retry without branch specification (some repos use different default names)
        result2 = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, target_dir],
            capture_output=True,
            text=True,
        )
        if result2.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{result2.stderr}")
