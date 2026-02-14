#!/usr/bin/env python3
"""Search arXiv API and output structured JSON results.

Usage:
    python scripts/search_arxiv.py "query string" [--max-results 20] [--start-date 2026-01-01]
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Query arXiv API and return parsed results."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "discopt-discoptbot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_data)

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        updated_el = entry.find("atom:updated", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        published = published_el.text.strip() if published_el is not None else ""
        updated = updated_el.text.strip() if updated_el is not None else ""

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None:
                authors.append(name_el.text.strip())

        arxiv_id = ""
        pdf_link = ""
        for link_el in entry.findall("atom:link", ns):
            href = link_el.get("href", "")
            if link_el.get("title") == "pdf":
                pdf_link = href
            elif "/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]

        # Extract primary category
        category_el = entry.find("arxiv:primary_category", ns)
        category = category_el.get("term", "") if category_el is not None else ""

        # Extract DOI if present
        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None else ""

        results.append(
            {
                "title": title,
                "authors": authors,
                "arxiv_id": arxiv_id,
                "published": published[:10],  # YYYY-MM-DD
                "updated": updated[:10],
                "abstract": abstract[:500],
                "category": category,
                "doi": doi,
                "pdf_link": pdf_link,
                "link": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Search arXiv API")
    parser.add_argument("query", help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--start-date",
        help="Filter results after this date (YYYY-MM-DD). Client-side filter.",
    )
    args = parser.parse_args()

    try:
        results = search_arxiv(args.query, args.max_results)
    except Exception as e:
        print(json.dumps({"error": str(e), "results": []}))
        sys.exit(1)

    # Client-side date filter (arXiv API doesn't support date ranges natively)
    if args.start_date:
        start = args.start_date
        results = [r for r in results if r["published"] >= start]

    print(json.dumps({"query": args.query, "count": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
