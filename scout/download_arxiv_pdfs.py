#!/usr/bin/env python3
"""Download arXiv PDFs for papers listed in arxiv_listings.

Reads the ID list from <data-dir>/arxiv_listings/<date>/<category>.json
and downloads the corresponding PDFs to <data-dir>/arxiv_pdfs/. Skips
already downloaded PDFs.

Needs network access to arxiv.org.

Usage:
    python download_arxiv_pdfs.py                              # last weekday, cs.LG
    python download_arxiv_pdfs.py --date 2025-06-09            # specific date
    python download_arxiv_pdfs.py --category cs.AI             # different category
    python download_arxiv_pdfs.py --date 2025-06-09 --limit 10 # at most 10
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


def default_date() -> str:
    """Most recent weekday before today (matches fetch_daily_arxiv.py)."""
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def download_pdf(arxiv_id: str, dest: Path, max_retries: int = 5) -> str:
    """Download a PDF from arXiv. Returns 'ok', 'withdrawn', or 'failed'.

    The PDF is written to a temporary file next to ``dest`` and renamed into
    place only once the download is complete, so an interrupted download never
    leaves a truncated file that a later run would treat as already downloaded.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    try:
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, dest)
                return "ok"
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"    Withdrawn or unavailable (404)")
                    return "withdrawn"
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Download failed: {e}")
                    return "failed"
        return "failed"
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download arXiv PDFs from a listing file.")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of PDFs to download")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    args = parser.parse_args()

    date_str = args.date or default_date()

    base = resolve_data_dir(args.data_dir)
    pdf_dir = base / "arxiv_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Collect IDs from all category listings, deduplicate
    all_ids = []
    for category in args.category:
        listing_file = base / "arxiv_listings" / date_str / f"{category}.json"
        if not listing_file.exists():
            print(f"Listing not found: {date_str}/{category} — skipping.")
            continue
        ids = json.loads(listing_file.read_text())
        print(f"Listing: {date_str}/{category}  ({len(ids)} papers)")
        all_ids.extend(ids)
    all_ids = list(dict.fromkeys(all_ids))  # deduplicate, preserve order

    if not all_ids:
        print("No listings found.")
        return

    # Skip already downloaded
    todo = [aid for aid in all_ids if not (pdf_dir / f"{aid}.pdf").exists()]
    already = len(all_ids) - len(todo)

    if args.limit:
        todo = todo[:args.limit]

    print(f"Total: {len(all_ids)}  |  To download: {len(todo)}  |  Already downloaded: {already}")

    downloaded = 0
    withdrawn = 0
    failed = 0
    for i, aid in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {aid}")
        status = download_pdf(aid, pdf_dir / f"{aid}.pdf")
        if status == "ok":
            downloaded += 1
        elif status == "withdrawn":
            withdrawn += 1
        else:
            failed += 1
        time.sleep(3)  # arXiv rate limit

    print(f"\nDone. Downloaded: {downloaded}  |  Withdrawn: {withdrawn}  |  Failed: {failed}")
    print(f"PDFs saved → {pdf_dir}")


if __name__ == "__main__":
    main()
