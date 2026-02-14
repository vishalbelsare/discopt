#!/usr/bin/env python3
"""Search OpenAlex API and output structured JSON results.

Usage:
    python scripts/search_openalex.py "query" --from-date 2026-01-01 --to-date 2026-02-14

API key: Set OPENALEX_API_KEY env var or pass --api-key. Free keys at openalex.org/settings/api.
Required since 2026-02-13 for more than 100 credits/day.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request


def search_openalex(
    query: str,
    from_date: str | None = None,
    to_date: str | None = None,
    per_page: int = 20,
    api_key: str | None = None,
) -> list[dict]:
    """Query OpenAlex API and return parsed results."""
    filters = [f"default.search:{query}"]
    if from_date:
        filters.append(f"from_created_date:{from_date}")
    if to_date:
        filters.append(f"to_created_date:{to_date}")

    params: dict = {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per_page": per_page,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "discopt-discoptbot/1.0 (mailto:jkitchin@andrew.cmu.edu)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for work in data.get("results", []):
        title = work.get("title", "") or ""
        doi = work.get("doi", "") or ""
        pub_date = work.get("publication_date", "") or ""
        oa_id = work.get("id", "") or ""

        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        # Extract abstract from inverted index if available
        abstract = ""
        inv_index = work.get("abstract_inverted_index")
        if inv_index:
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in inv_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)[:500]

        # Source/journal info
        source = ""
        primary_loc = work.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        source = source_obj.get("display_name", "") or ""

        # Open access link
        link = ""
        best_oa = work.get("best_oa_location") or {}
        link = best_oa.get("pdf_url", "") or best_oa.get("landing_page_url", "") or ""
        if not link and doi:
            link = doi if doi.startswith("http") else f"https://doi.org/{doi}"

        results.append(
            {
                "title": title,
                "authors": authors[:5],  # first 5 authors
                "doi": doi.replace("https://doi.org/", "")
                if doi.startswith("https://doi.org/")
                else doi,
                "published": pub_date,
                "abstract": abstract,
                "source": source,
                "openalex_id": oa_id,
                "link": link,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Search OpenAlex API")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--from-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--per-page", type=int, default=20)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="OpenAlex API key (or set OPENALEX_API_KEY env var)",
    )
    args = parser.parse_args()

    try:
        results = search_openalex(
            args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            per_page=args.per_page,
            api_key=args.api_key,
        )
    except Exception as e:
        print(json.dumps({"error": str(e), "results": []}))
        sys.exit(1)

    print(
        json.dumps(
            {
                "query": args.query,
                "count": len(results),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
