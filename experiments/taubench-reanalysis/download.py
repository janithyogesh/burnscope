"""Download tau-bench's historical trajectories at a pinned commit.

Fetches only the pinned commit (no full history) into ./data/, which is
git-ignored, and records where the data came from in data/SOURCE.json.
This script never runs any tau-bench code.

Usage:
    uv run python experiments/taubench-reanalysis/download.py
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_URL = "https://github.com/sierra-research/tau-bench.git"
COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
DATA_DIR = Path(__file__).parent / "data"
CHECKOUT_DIR = DATA_DIR / "tau-bench"
EXPECTED_FILES = [
    "gpt-4o-airline.json",
    "gpt-4o-retail.json",
    "sonnet-35-new-airline.json",
    "sonnet-35-new-retail.json",
]


def git(*args: str) -> str:
    """Run a git command inside the checkout folder and return its output."""
    result = subprocess.run(
        ["git", "-C", str(CHECKOUT_DIR), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def current_commit() -> str | None:
    """Return the commit already checked out, or None if there is no checkout."""
    if not (CHECKOUT_DIR / ".git").exists():
        return None
    return git("rev-parse", "HEAD")


def download() -> None:
    """Fetch the pinned commit, unless it is already present."""
    if current_commit() == COMMIT:
        print(f"Already at {COMMIT[:7]}, nothing to download.")
        return
    if CHECKOUT_DIR.exists():
        sys.exit(f"{CHECKOUT_DIR} exists but is not at {COMMIT[:7]}. Delete it and rerun.")
    CHECKOUT_DIR.mkdir(parents=True)
    git("init", "-q")
    git("remote", "add", "origin", REPO_URL)
    print(f"Fetching {REPO_URL} at {COMMIT[:7]} ...")
    git("fetch", "-q", "--depth", "1", "origin", COMMIT)
    git("checkout", "-q", "FETCH_HEAD")


def verify_and_record() -> None:
    """Check the commit and expected files, then write data/SOURCE.json."""
    commit = current_commit()
    if commit != COMMIT:
        sys.exit(f"Expected commit {COMMIT}, found {commit}.")
    trajectories = CHECKOUT_DIR / "historical_trajectories"
    missing = [name for name in EXPECTED_FILES if not (trajectories / name).exists()]
    if missing:
        sys.exit(f"Missing trajectory files: {', '.join(missing)}")
    source = {
        "repo": REPO_URL,
        "commit": commit,
        "license": "MIT",
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": EXPECTED_FILES,
    }
    (DATA_DIR / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"OK: {len(EXPECTED_FILES)} trajectory files at commit {commit}.")


if __name__ == "__main__":
    download()
    verify_and_record()
