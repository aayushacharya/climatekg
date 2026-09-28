#!/usr/bin/env python3
"""Extract Publication Hub "More Info" entities for local CMIP publications.

The CMIP Publication Hub renders the More Info table via the AJAX endpoint
`ajax/moreinfo/<publication_id>`. This script maps local OpenAlex records from
`paper_metadata_v2.json` to Publication Hub IDs, fetches More Info for the
matched records, and writes a comparison-ready JSON artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from urllib3.exceptions import InsecureRequestWarning
except ImportError:  # pragma: no cover
    InsecureRequestWarning = None


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXTRACTION_DIR.parent

DEFAULT_METADATA_PATH = EXTRACTION_DIR / "papers" / "metadata" / "paper_metadata_v2.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "cmip_publication_moreinfo_entities"
DEFAULT_VIEW_URL = "https://cmip-publications.llnl.gov/view/all/"
DEFAULT_SUBSET_KEY = "cmip_publications"

SPACE_RE = re.compile(r"\s+")
DOI_PREFIX_RE = re.compile(r"^(?:doi\s*:\s*)?(?:https?://(?:dx\.)?doi\.org/)?", re.I)
SITE_ID_RE = re.compile(r"show_citation\((\d+)\)")
OUTER_PUNCT_RE = re.compile(r"^[\s\"'`.,;:()\[\]{}<>]+|[\s\"'`.,;:()\[\]{}<>]+$")

MOREINFO_KEY_MAP = {
    "activities": "activities",
    "experiments": "experiments",
    "models": "sources",
    "frequency": "frequencies",
}
ENTITY_TYPES = ["activities", "experiments", "sources", "frequencies"]


def now_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def display_text(value: Any) -> str:
    text = "" if value is None else str(value)
    previous = None
    while previous != text:
        previous = text
        text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_title(value: Any) -> str:
    text = display_text(value)
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = text.replace("&", " and ")
    text = re.sub(r"co\s*2\b", "co2", text, flags=re.I)
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_doi(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = html.unescape(text).casefold()
    text = DOI_PREFIX_RE.sub("", text).strip().strip(".")
    return text


def normalize_entity_name(value: Any) -> str:
    text = display_text(value)
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    text = OUTER_PUNCT_RE.sub("", text)
    return SPACE_RE.sub(" ", text).casefold().strip()


def normalize_moreinfo_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
    for source_key, target_key in MOREINFO_KEY_MAP.items():
        raw_values = payload.get(source_key) or []
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        seen: set[str] = set()
        values: list[str] = []
        for raw_value in raw_values:
            value = display_text(raw_value)
            normalized_value = normalize_entity_name(value)
            if not value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            values.append(value)
        normalized[target_key] = sorted(values, key=lambda item: item.casefold())
    return normalized


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    verify_ssl: bool,
    timeout: float,
    retries: int,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    verify_ssl: bool,
    timeout: float,
    retries: int,
) -> str:
    response = request_with_retries(
        session,
        url,
        verify_ssl=verify_ssl,
        timeout=timeout,
        retries=retries,
    )
    return response.text


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    verify_ssl: bool,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    response = request_with_retries(
        session,
        url,
        verify_ssl=verify_ssl,
        timeout=timeout,
        retries=retries,
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(payload).__name__}")
    return payload


def title_text_from_div(title_div: Any) -> str:
    parts: list[str] = []
    for child in title_div.contents:
        child_name = getattr(child, "name", None)
        if child_name == "br":
            break
        if hasattr(child, "get_text"):
            parts.append(child.get_text(" ", strip=False))
        else:
            parts.append(str(child))
    return display_text(" ".join(parts))


def parse_publication_rows(html_text: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: list[dict[str, Any]] = []
    for publication in soup.select("div.publication"):
        citation_link = publication.find("a", onclick=SITE_ID_RE)
        if citation_link is None:
            continue
        onclick = str(citation_link.get("onclick") or "")
        site_id_match = SITE_ID_RE.search(onclick)
        if not site_id_match:
            continue
        title_div = publication.select_one("div.titles")
        if title_div is None:
            continue
        year_div = publication.select_one("div.dates")
        authors_div = publication.select_one("div.authors")
        title = title_text_from_div(title_div)
        rows.append(
            {
                "site_id": site_id_match.group(1),
                "site_year": display_text(year_div.get_text(" ", strip=True) if year_div else ""),
                "site_authors_display": display_text(
                    authors_div.get_text(" ", strip=True) if authors_div else ""
                ),
                "site_title": title,
                "site_title_normalized": normalize_title(title),
            }
        )
    return rows


def page_url(view_url: str, scroll_count: int | None) -> str:
    if scroll_count is None:
        return view_url
    separator = "&" if "?" in view_url else "?"
    return f"{view_url}{separator}type=all&scroll_count={scroll_count}"


def fetch_site_rows(
    session: requests.Session,
    view_url: str,
    *,
    verify_ssl: bool,
    timeout: float,
    retries: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []
    for page_index in range(max_pages):
        scroll_count = None if page_index == 0 else page_index
        url = page_url(view_url, scroll_count)
        html_text = fetch_text(
            session,
            url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            retries=retries,
        )
        rows = parse_publication_rows(html_text)
        page_stats.append(
            {
                "page_index": page_index,
                "scroll_count": scroll_count,
                "url": url,
                "row_count": len(rows),
                "byte_count": len(html_text.encode("utf-8")),
            }
        )
        if not rows and page_index > 0:
            break
        all_rows.extend(rows)
    return all_rows, page_stats


def unique_rows_by_site_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        site_id = str(row["site_id"])
        if site_id in seen:
            continue
        seen.add(site_id)
        unique_rows.append(row)
    return unique_rows


def fetch_many_json(
    session: requests.Session,
    urls_by_key: dict[str, str],
    *,
    verify_ssl: bool,
    timeout: float,
    retries: int,
    max_workers: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    def worker(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        key, url = item
        return key, fetch_json(
            session,
            url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            retries=retries,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in urls_by_key.items()}
        for future in concurrent.futures.as_completed(futures):
            key, url = futures[future]
            try:
                result_key, payload = future.result()
                results[result_key] = payload
            except Exception as exc:  # noqa: BLE001 - preserve all fetch failures.
                errors.append({"key": key, "url": url, "error": repr(exc)})
    errors.sort(key=lambda item: item["key"])
    return results, errors


def citation_url(view_url: str, site_id: str) -> str:
    return urljoin(view_url.rstrip("/") + "/", f"ajax/citation/{site_id}")


def moreinfo_url(view_url: str, site_id: str) -> str:
    return urljoin(view_url.rstrip("/") + "/", f"ajax/moreinfo/{site_id}")


def local_author_names(record: dict[str, Any]) -> list[str]:
    authors = record.get("authors") or []
    names: list[str] = []
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict):
                name = display_text(author.get("display_name"))
            else:
                name = display_text(author)
            if name:
                names.append(name)
    return names


def load_local_subset(metadata_path: Path, subset_key: str, limit: int | None) -> list[dict[str, Any]]:
    data = read_json(metadata_path)
    papers = data.get("papers")
    if not isinstance(papers, dict):
        raise ValueError(f"{metadata_path} does not contain a 'papers' object")
    subset = data.get(subset_key)
    if not isinstance(subset, dict) or not isinstance(subset.get("ids"), list):
        raise ValueError(f"{metadata_path} does not contain '{subset_key}.ids'")

    records: list[dict[str, Any]] = []
    for openalex_id in subset["ids"]:
        paper = papers.get(openalex_id)
        if not isinstance(paper, dict):
            continue
        title = display_text(paper.get("title"))
        record = {
            "openalex_id": openalex_id,
            "title": title,
            "title_normalized": normalize_title(title),
            "doi": paper.get("doi") or "",
            "doi_normalized": normalize_doi(paper.get("doi")),
            "publication_year": paper.get("publication_year"),
            "authors": local_author_names(paper),
        }
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def build_site_indices(
    site_rows: list[dict[str, Any]],
    citations_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows_by_id = {str(row["site_id"]): row for row in site_rows}
    doi_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    title_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    site_records: list[dict[str, Any]] = []
    for site_id, citation in citations_by_id.items():
        row = rows_by_id.get(str(site_id), {})
        site_title = display_text(citation.get("title") or row.get("site_title"))
        site_doi = citation.get("doi") or ""
        site_record = {
            "site_id": str(site_id),
            "site_title": site_title,
            "site_title_normalized": normalize_title(site_title),
            "site_doi": site_doi,
            "site_doi_normalized": normalize_doi(site_doi),
            "site_year": citation.get("year") or row.get("site_year"),
            "site_authors": citation.get("authors") or [],
            "site_authors_display": row.get("site_authors_display", ""),
            "site_type": citation.get("type"),
            "site_url": citation.get("url"),
            "raw_citation": citation,
        }
        site_records.append(site_record)
        if site_record["site_doi_normalized"]:
            doi_index[site_record["site_doi_normalized"]].append(site_record)
        if site_record["site_title_normalized"]:
            title_index[site_record["site_title_normalized"]].append(site_record)
    site_records.sort(key=lambda item: int(item["site_id"]))
    return doi_index, title_index, site_records


def choose_site_match(
    local_record: dict[str, Any],
    doi_index: dict[str, list[dict[str, Any]]],
    title_index: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, dict[str, Any] | None]:
    doi = local_record["doi_normalized"]
    title = local_record["title_normalized"]
    if doi and doi in doi_index:
        candidates = doi_index[doi]
        for candidate in candidates:
            if title and candidate["site_title_normalized"] == title:
                return "doi_and_title", candidate
        return "doi", candidates[0]
    if title and title in title_index:
        return "title", title_index[title][0]
    return None, None


def aggregate_entities(records: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    entity_occurrences: dict[str, dict[str, dict[str, Any]]] = {
        entity_type: {} for entity_type in ENTITY_TYPES
    }
    for record in records:
        more_info = record.get("more_info") or {}
        for entity_type in ENTITY_TYPES:
            for entity_name in more_info.get(entity_type, []):
                normalized_name = normalize_entity_name(entity_name)
                if not normalized_name:
                    continue
                bucket = entity_occurrences[entity_type].setdefault(
                    normalized_name,
                    {
                        "name": entity_name,
                        "normalized_name": normalized_name,
                        "paper_ids": [],
                        "site_ids": [],
                        "paper_count": 0,
                    },
                )
                bucket["paper_ids"].append(record["openalex_id"])
                bucket["site_ids"].append(record["site_id"])

    unique_entities: dict[str, list[str]] = {}
    entity_records: dict[str, list[dict[str, Any]]] = {}
    for entity_type, buckets in entity_occurrences.items():
        typed_records: list[dict[str, Any]] = []
        for bucket in buckets.values():
            bucket["paper_ids"] = sorted(set(bucket["paper_ids"]))
            bucket["site_ids"] = sorted(set(bucket["site_ids"]), key=int)
            bucket["paper_count"] = len(bucket["paper_ids"])
            typed_records.append(bucket)
        typed_records.sort(key=lambda item: item["name"].casefold())
        entity_records[entity_type] = typed_records
        unique_entities[entity_type] = [item["name"] for item in typed_records]
    return unique_entities, entity_records


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    if not args.verify_ssl and InsecureRequestWarning is not None:
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

    session = requests.Session()
    session.headers.update({"User-Agent": "esgf-kg-cmip-moreinfo-extractor/1.0"})

    local_records = load_local_subset(args.metadata_path, args.subset_key, args.limit)
    site_rows, page_stats = fetch_site_rows(
        session,
        args.view_url,
        verify_ssl=args.verify_ssl,
        timeout=args.timeout,
        retries=args.retries,
        max_pages=args.max_pages,
    )
    unique_site_rows = unique_rows_by_site_id(site_rows)

    citation_urls = {
        str(row["site_id"]): citation_url(args.view_url, str(row["site_id"]))
        for row in unique_site_rows
    }
    citations_by_id, citation_errors = fetch_many_json(
        session,
        citation_urls,
        verify_ssl=args.verify_ssl,
        timeout=args.timeout,
        retries=args.retries,
        max_workers=args.max_workers,
    )
    doi_index, title_index, site_records = build_site_indices(unique_site_rows, citations_by_id)

    paper_records: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    site_ids_to_fetch: set[str] = set()
    for local_record in local_records:
        match_method, site_record = choose_site_match(local_record, doi_index, title_index)
        if site_record is None or match_method is None:
            unmatched_records.append(local_record)
            match_counts["unmatched"] += 1
            continue
        match_counts[match_method] += 1
        site_id = str(site_record["site_id"])
        site_ids_to_fetch.add(site_id)
        paper_records.append(
            {
                **local_record,
                "match_method": match_method,
                "site_id": site_id,
                "site_title": site_record["site_title"],
                "site_doi": site_record["site_doi"],
                "site_year": site_record["site_year"],
                "site_url": site_record["site_url"],
                "more_info": {entity_type: [] for entity_type in ENTITY_TYPES},
            }
        )

    moreinfo_urls = {
        site_id: moreinfo_url(args.view_url, site_id) for site_id in sorted(site_ids_to_fetch, key=int)
    }
    moreinfo_by_id, moreinfo_errors = fetch_many_json(
        session,
        moreinfo_urls,
        verify_ssl=args.verify_ssl,
        timeout=args.timeout,
        retries=args.retries,
        max_workers=args.max_workers,
    )

    for paper_record in paper_records:
        payload = moreinfo_by_id.get(str(paper_record["site_id"])) or {}
        paper_record["more_info"] = normalize_moreinfo_payload(payload)

    paper_records.sort(key=lambda item: item["openalex_id"])
    unique_entities, entity_records = aggregate_entities(paper_records)
    unique_counts = {entity_type: len(values) for entity_type, values in unique_entities.items()}
    per_type_paper_counts = {
        entity_type: sum(1 for record in paper_records if record["more_info"].get(entity_type))
        for entity_type in ENTITY_TYPES
    }
    records_with_any_moreinfo = sum(
        1
        for record in paper_records
        if any(record["more_info"].get(entity_type) for entity_type in ENTITY_TYPES)
    )

    return {
        "config": {
            "metadata_path": str(args.metadata_path),
            "subset_key": args.subset_key,
            "view_url": args.view_url,
            "verify_ssl": args.verify_ssl,
            "limit": args.limit,
            "max_workers": args.max_workers,
            "timeout": args.timeout,
            "retries": args.retries,
        },
        "summary": {
            "local_record_count": len(local_records),
            "matched_record_count": len(paper_records),
            "unmatched_record_count": len(unmatched_records),
            "match_counts": dict(sorted(match_counts.items())),
            "site_displayed_row_count": len(site_rows),
            "site_unique_publication_id_count": len(unique_site_rows),
            "site_citation_record_count": len(citations_by_id),
            "site_citation_error_count": len(citation_errors),
            "moreinfo_site_id_count": len(moreinfo_by_id),
            "moreinfo_error_count": len(moreinfo_errors),
            "records_with_any_moreinfo": records_with_any_moreinfo,
            "records_without_moreinfo": len(paper_records) - records_with_any_moreinfo,
            "records_with_entities_by_type": per_type_paper_counts,
            "unique_entity_counts": unique_counts,
        },
        "page_stats": page_stats,
        "unique_entities": unique_entities,
        "entity_records": entity_records,
        "paper_records": paper_records,
        "unmatched_records": unmatched_records,
        "site_records": site_records,
        "fetch_errors": {
            "citation": citation_errors,
            "moreinfo": moreinfo_errors,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Publication Hub More Info entities for paper_metadata_v2 "
            "cmip_publications records."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--subset-key", default=DEFAULT_SUBSET_KEY)
    parser.add_argument("--view-url", default=DEFAULT_VIEW_URL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test record limit.")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Verify SSL certificates. Disabled by default because the site currently has SSL issues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    output_dir = args.output_root / run_id
    result = build_result(args)

    write_json(output_dir / "config.json", result["config"])
    write_json(output_dir / "summary.json", result["summary"])
    write_json(output_dir / "unique_entities.json", result["unique_entities"])
    write_json(output_dir / "entity_records.json", result["entity_records"])
    write_json(output_dir / "paper_moreinfo_records.json", result["paper_records"])
    write_jsonl(output_dir / "paper_moreinfo_records.jsonl", result["paper_records"])
    write_json(output_dir / "unmatched_records.json", result["unmatched_records"])
    write_json(output_dir / "site_records.json", result["site_records"])
    write_json(output_dir / "fetch_errors.json", result["fetch_errors"])
    write_json(output_dir / "page_stats.json", result["page_stats"])
    write_json(output_dir / "result.json", result)

    print(f"Output directory: {output_dir}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
