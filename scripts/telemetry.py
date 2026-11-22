#!/usr/bin/env python3
"""Read-only organic adoption telemetry tracker for `cloture`.

Queries public read-only APIs (GitHub repo/releases, PyPI pypistats, crates.io)
to log real third-party traction snapshots for evidence-driven iteration.
Zero credentials required (optional GITHUB_TOKEN increases GitHub rate limit).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO = "pgg898200-ship-it/cloture"
PYPI_PKG = "cloture"
CRATES = ("cloture-core", "cloture-cli")
USER_AGENT = "cloture-telemetry/0.1 (+https://github.com/pgg898200-ship-it/cloture)"


def _get_json(url: str, extra_headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def collect_snapshot(repo: str = REPO) -> Dict[str, Any]:
    gh_headers: Dict[str, str] = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        gh_headers["Authorization"] = f"Bearer {token}"

    gh_repo = _get_json(f"https://api.github.com/repos/{repo}", gh_headers)
    gh_releases = _get_json(f"https://api.github.com/repos/{repo}/releases", gh_headers)

    release_downloads = 0
    release_count = 0
    if isinstance(gh_releases, list):
        release_count = len(gh_releases)
        for rel in gh_releases:
            for asset in rel.get("assets", []):
                release_downloads += int(asset.get("download_count", 0))

    pypi_recent = _get_json(f"https://pypistats.org/api/packages/{PYPI_PKG}/recent")
    pypi_meta = _get_json(f"https://pypi.org/pypi/{PYPI_PKG}/json")

    crates_stats: Dict[str, Any] = {}
    for crate in CRATES:
        data = _get_json(f"https://crates.io/api/v1/crates/{crate}")
        if data and "crate" in data:
            c = data["crate"]
            crates_stats[crate] = {
                "downloads": c.get("downloads", 0),
                "recent_downloads": c.get("recent_downloads", 0),
                "max_version": c.get("max_version"),
            }
        else:
            crates_stats[crate] = None

    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": repo,
        "github": (
            {
                "stars": gh_repo.get("stargazers_count", 0),
                "forks": gh_repo.get("forks_count", 0),
                "watchers": gh_repo.get("subscribers_count", 0),
                "open_issues": gh_repo.get("open_issues_count", 0),
                "releases": release_count,
                "release_asset_downloads": release_downloads,
            }
            if isinstance(gh_repo, dict) and "stargazers_count" in gh_repo
            else None
        ),
        "pypi": {
            "published": bool(pypi_meta and "info" in pypi_meta),
            "version": pypi_meta["info"].get("version") if pypi_meta and "info" in pypi_meta else None,
            "recent": pypi_recent.get("data") if isinstance(pypi_recent, dict) else None,
        },
        "crates_io": crates_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public organic traction telemetry for cloture.")
    parser.add_argument("--repo", default=REPO, help="GitHub owner/repo")
    parser.add_argument("--append", type=Path, default=None, help="Append JSONL snapshot to path")
    args = parser.parse_args()

    snap = collect_snapshot(args.repo)
    print(json.dumps(snap, indent=2))
    if args.append is not None:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snap) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
