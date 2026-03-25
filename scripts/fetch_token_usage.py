#!/usr/bin/env python3
"""
Fetch daily token usage from the Anthropic Admin API and write/merge
snapshots into src/data/token-usage.json.

API docs: https://platform.claude.com/docs/en/build-with-claude/usage-cost-api

Requires:
  ANTHROPIC_ADMIN_API_KEY  — Admin API key (starts with sk-ant-admin...)

Usage:
  python scripts/fetch_token_usage.py                    # yesterday
  python scripts/fetch_token_usage.py 2026-03-01 2026-03-24  # date range
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.parse
    requests = None  # fall back to stdlib below

API_URL = "https://api.anthropic.com/v1/organizations/usage_report/messages"
DATA_FILE = Path(__file__).resolve().parent.parent / "src" / "data" / "token-usage.json"


# ---------------------------------------------------------------------------
# HTTP helpers (requests or stdlib fallback)
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict, params: dict) -> dict:
    if requests is not None:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    # stdlib fallback
    qs = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{qs}"
    req = urllib.request.Request(full_url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def fetch_usage(start: datetime, end: datetime, api_key: str) -> list[dict]:
    """Fetch token usage grouped by model for the given UTC time range.

    The API uses cursor-based pagination; this function fetches all pages.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    params: dict = {
        "starting_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ending_at": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bucket_width": "1d",
        "group_by[]": "model",
    }

    records: list[dict] = []
    page = 1
    while True:
        payload = _get(API_URL, headers=headers, params=params)
        batch = payload.get("data", [])
        records.extend(batch)
        print(f"  Page {page}: {len(batch)} record(s)")
        if not payload.get("has_more"):
            break
        params["page"] = payload["next_page"]
        page += 1

    return records


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def normalise(records: list[dict]) -> list[dict]:
    """Convert raw API records to our canonical storage format."""
    out = []
    for r in records:
        # bucket_start_time: "2026-03-24T00:00:00Z"
        date = r.get("bucket_start_time", "")[:10]
        if not date:
            continue
        out.append({
            "date": date,
            "model": r.get("model") or "unknown",
            "input_tokens": int(r.get("input_tokens") or 0),
            "output_tokens": int(r.get("output_tokens") or 0),
            "cached_input_tokens": int(r.get("cached_input_tokens") or 0),
            "cache_creation_input_tokens": int(r.get("cache_creation_input_tokens") or 0),
        })
    return out


def merge(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge new rows into existing list, deduplicating by (date, model)."""
    index = {(r["date"], r["model"]): r for r in existing}
    for r in new_rows:
        index[(r["date"], r["model"])] = r
    return sorted(index.values(), key=lambda r: (r["date"], r["model"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_ADMIN_API_KEY", "").strip()
    if not api_key:
        print("Error: ANTHROPIC_ADMIN_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if len(sys.argv) == 3:
        try:
            start = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc) + timedelta(days=1)
        except ValueError as exc:
            print(f"Error: {exc}\nUsage: fetch_token_usage.py [YYYY-MM-DD [YYYY-MM-DD]]", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: fetch yesterday's full day
        end = now
        start = now - timedelta(days=1)

    print(f"Fetching usage {start.date()} → {(end - timedelta(seconds=1)).date()} …")
    raw = fetch_usage(start, end, api_key)
    new_rows = normalise(raw)
    print(f"  {len(new_rows)} model-day record(s) from API.")

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []

    merged = merge(existing, new_rows)
    DATA_FILE.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"Wrote {len(merged)} total record(s) to {DATA_FILE}.")


if __name__ == "__main__":
    main()
