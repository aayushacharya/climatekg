"""
Find candidate papers for the knowledge graph (and download their PDFs).

1. Query Neo4j for top 10 keywords by paper count.
2. Search OpenAlex for most recent papers using the union of keyword URIs,
   filtering out papers already in the KG.
3. Check relevance via LLM (climate model simulation data).
4. Download the PDF to papers/pdfs/{date}/<OpenAlexID>.pdf. A paper is only
   counted as a candidate if its PDF actually downloads and validates.
5. Save metadata to papers/metadata/candidate_papers_{date}.json.

Caps: MAX_PAPERS (1000) per run, MAX_ITERATIONS (200) fallback to avoid infinite loops.
"""

import os
import re
import sys
import json
import datetime
import requests
from pathlib import Path
from time import sleep
from typing import List, Dict, Set, Optional

from neo4j import GraphDatabase
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from tqdm import tqdm


# Load env (GOOGLE_API_KEY, etc.)
load_dotenv()  # also check cwd / parent dirs

# ── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:8687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# ── Limits ─────────────────────────────────────────────────────────────────
MAX_PAPERS = 1000       # max candidate papers per day
MAX_ITERATIONS = 200    # fallback cap on OpenAlex pagination loops
PER_PAGE = 50           # results per OpenAlex request
LLM_MODEL = "llama3.3"
LLM_TIMEOUT = 120
MAX_PDF_DOWNLOAD_BYTES = 100 * 1024 * 1024
PDF_CHUNK_SIZE = 8192
PDF_TIMEOUT = 30

RELEVANCE_PROMPT = (
    "Based on the title and abstract of the paper, analyze whether the paper "
    "likely contains climate model simulation data or results. "
    'Answer with a simple "yes" or "no".\n'
    "Title: {title}\n"
    "Abstract: {abstract}"
)


# ── Neo4j helpers ──────────────────────────────────────────────────────────

def get_top_keywords(n: int = 10) -> List[Dict]:
    """Return top-N keywords ordered by distinct paper count."""
    query = (
        "MATCH (p:Paper)-[:PAPER_HAS_KEYWORD]->(k:Keyword) "
        "WITH k, count(DISTINCT p) AS paper_count "
        "RETURN k.uri AS uri, paper_count "
        "ORDER BY paper_count DESC LIMIT $n"
    )
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(database=NEO4J_DB) as session:
            return [dict(r) for r in session.run(query, n=n)]


def get_existing_paper_ids() -> Set[str]:
    """Collect all paper DOIs and URIs already in the KG (lowercased)."""
    query = "MATCH (p:Paper) RETURN p.doi AS doi, p.uri AS uri"
    ids: Set[str] = set()
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session(database=NEO4J_DB) as session:
            for r in session.run(query):
                for val in (r["doi"], r["uri"]):
                    if val:
                        ids.add(val.strip().lower())
    return ids


# ── OpenAlex helpers ───────────────────────────────────────────────────────

def paper_info_from_api_response(work: Dict) -> Dict:
    """Extract relevant paper information from OpenAlex API response."""
    paper_info = {
                "id": work.get("id"),
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                "cited_by_count": work.get("cited_by_count"),
                "pdf_url": work.get("primary_location", {}).get("pdf_url"),
                "authors": [author.get("author", {}) 
                           for author in work.get("authorships", [])],
                "concepts": work.get("concepts", []),
                "venue_name": ((work.get("primary_location", {}) or {}).get("source", {}) or {}).get("display_name"),
                "venue_id":   ((work.get("primary_location", {}) or {}).get("source", {}) or {}).get("id"),
                "venue_type": ((work.get("primary_location", {}) or {}).get("source", {}) or {}).get("type"),
                "keywords": work.get("keywords", []),
                "relevance_score": work.get("relevance_score"),
                "referenced_works": work.get("referenced_works", []),
            }
    return paper_info

def reconstruct_abstract(inverted_index: Dict) -> Optional[str]:
    """
    Reconstruct abstract from OpenAlex's inverted index format.
    
    Args:
        inverted_index: Dictionary mapping words to positions
    
    Returns:
        Reconstructed abstract text
    """
    if not inverted_index:
        return None
    
    # Create list of (position, word) tuples
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    # Sort by position and join words
    word_positions.sort()
    abstract = " ".join([word for _, word in word_positions])
    
    return abstract

def _build_keyword_filter(keyword_uris: List[str]) -> str:
    """
    Build an OpenAlex filter for the union (OR) of keyword URIs.

    Handles both legacy concept URIs (https://openalex.org/C…)
    and new-style keyword URIs (https://openalex.org/keywords/…).
    Falls back to text search if URIs are plain names.
    """
    if not keyword_uris:
        return ""

    sample = keyword_uris[0]
    if "/keywords/" in sample:
        prefix = "keywords.id"
    elif "/concepts/" in sample or sample.startswith("C"):
        prefix = "concepts.id"
    else:
        # URIs appear to be display names → use text search instead
        search_terms = " OR ".join(keyword_uris)
        return f"default.search:{search_terms}"

    return f"{prefix}:{('|').join(keyword_uris)}"


def search_openalex_papers(kw_filter: str, cursor: str = "*") -> Dict:
    """Fetch one page of recent OA papers matching the keyword filter."""
    params = {
        "filter": f"{kw_filter},has_pdf_url:true,is_oa:true",
        "sort": "publication_date:desc",
        "per_page": PER_PAGE,
        "cursor": cursor,
    }
    resp = requests.get(
        "https://api.openalex.org/works", params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


# ── Deduplication ──────────────────────────────────────────────────────────

def _paper_key(paper: Dict) -> str:
    """Normalised identifier for dedup."""
    doi = (paper.get("doi") or "").strip().lower()
    oa_id = (paper.get("id") or "").strip().lower()
    return doi or oa_id


def _is_duplicate(paper: Dict, existing: Set[str], seen: Set[str]) -> bool:
    doi = (paper.get("doi") or "").strip().lower()
    oa_id = (paper.get("id") or "").strip().lower()
    for v in (doi, oa_id):
        if v and (v in existing or v in seen):
            return True
    return False


def _mark_seen(paper: Dict, seen: Set[str]):
    for v in ((paper.get("doi") or ""), (paper.get("id") or "")):
        v = v.strip().lower()
        if v:
            seen.add(v)


# ── PDF check ──────────────────────────────────────────────────────────────

_PDF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/143.0.0.0 Safari/537.36"
}


def safe_pdf_name(key: str, paper: Dict) -> str:
    """Build the on-disk PDF filename (<OpenAlexID>.pdf) expected downstream."""
    paper_id = paper.get("id") or key
    name = str(paper_id).rstrip("/").split("/")[-1] or str(key)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        name = f"paper_{abs(hash(key))}"
    return f"{name}.pdf"


def download_and_validate_pdf(pdf_url: str, output_path: Path, timeout: int = PDF_TIMEOUT) -> bool:
    """
    Download a PDF to output_path and validate it. Returns True if a valid PDF
    ends up at output_path.

    Resumable: if a valid PDF already exists at output_path, it is reused
    (no re-download). On any failure the partial download is cleaned up and the
    function returns False.
    """
    if not pdf_url:
        return False

    if output_path.exists():
        status, _ = check_pdf(str(output_path))
        if status == "OK":
            return True
        output_path.unlink()  # invalid leftover, re-download

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with requests.get(
            pdf_url,
            headers=_PDF_HEADERS,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as resp:
            if resp.status_code >= 400:
                return False

            total_bytes = 0
            header = bytearray()
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=PDF_CHUNK_SIZE):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > MAX_PDF_DOWNLOAD_BYTES:
                        return False
                    if len(header) < 1024:
                        header.extend(chunk[:1024 - len(header)])
                    f.write(chunk)

        if total_bytes == 0 or b"%PDF-" not in bytes(header):
            return False

        status, _ = check_pdf(str(tmp_path))
        if status != "OK":
            return False

        os.replace(tmp_path, output_path)
        return True
    except Exception:
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def check_pdf(file_path):
    """
    Check if a PDF file is corrupted or has no pages.

    Borrowed from info_extraction/check_pdf_corrupted.py.
    Returns a tuple: (status, message), where status is one of
    'OK', 'CORRUPTED', 'NO_PAGES', or 'ERROR'.
    """
    try:
        import PyPDF2 as pdf_lib
    except ImportError:
        try:
            import pypdf as pdf_lib
        except ImportError as exc:
            return "ERROR", f"PDF parser is not installed: {str(exc)}"

    try:
        with open(file_path, "rb") as f:
            reader = pdf_lib.PdfReader(f)
            num_pages = len(reader.pages)

            if num_pages == 0:
                return "NO_PAGES", "PDF has 0 pages"

            try:
                _ = reader.pages[0]
                return "OK", f"Valid PDF with {num_pages} page(s)"
            except Exception as exc:
                return "CORRUPTED", f"Cannot read pages: {str(exc)}"

    except pdf_lib.errors.PdfReadError as exc:
        return "CORRUPTED", f"PDF read error: {str(exc)}"
    except Exception as exc:
        return "ERROR", f"Unexpected error: {str(exc)}"


# ── Relevance check ───────────────────────────────────────────────────────

def check_relevance(llm: Ollama, title: str, abstract: str) -> bool:
    """Ask LLM whether the paper contains climate-model simulation data."""
    if not title or not abstract:
        return False
    prompt = RELEVANCE_PROMPT.format(title=title, abstract=abstract)
    try:
        resp = llm.complete(prompt)
        answer = str(resp).strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        print(f"  [LLM error] {exc}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────

def find_candidate_papers():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir = Path(__file__).resolve().parent
    meta_dir = script_dir / "papers" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f"candidate_papers_{today}.json"

    pdf_dir = script_dir / "papers" / "pdfs" / today
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: top keywords ───────────────────────────────────────────
    print("Step 1: Fetching top 10 keywords from KG …")
    keywords = get_top_keywords(10)
    if not keywords:
        print("No keywords found in KG. Exiting.")
        return
    for kw in keywords:
        print(f"  {kw['uri']}  (papers: {kw['paper_count']})")

    keyword_uris = [kw["uri"] for kw in keywords if kw.get("uri")]
    kw_filter = _build_keyword_filter(keyword_uris)
    print(f"\nOpenAlex filter: {kw_filter}")

    # ── Step 2: existing papers ────────────────────────────────────────
    print("\nStep 2: Loading existing paper identifiers from KG …")
    existing_ids = get_existing_paper_ids()
    print(f"  {len(existing_ids)} identifiers already in KG.")

    # ── Step 3+4: search → relevance → collect ────────────────────────
    print("\nStep 3: Searching OpenAlex, checking relevance & downloading PDFs …")
    llm = Ollama(model=LLM_MODEL, request_timeout=LLM_TIMEOUT)

    candidates: Dict[str, Dict] = {}
    seen: Set[str] = set()
    cursor = "*"
    iterations = 0

    pbar = tqdm(total=MAX_PAPERS, desc="Candidate papers")
    while len(candidates) < MAX_PAPERS and iterations < MAX_ITERATIONS:
        iterations += 1
        try:
            data = search_openalex_papers(kw_filter, cursor=cursor)
        except Exception as exc:
            print(f"\n  [OpenAlex error] {exc}")
            break

        works = data.get("results", [])
        if not works:
            break

        for work in works:
            if len(candidates) >= MAX_PAPERS:
                break

            paper = paper_info_from_api_response(work)

            if _is_duplicate(paper, existing_ids, seen):
                continue
            _mark_seen(paper, seen)

            if not paper.get("pdf_url"):
                continue

            # Relevance first (cheap) so we don't download irrelevant PDFs.
            if not check_relevance(llm, paper.get("title", ""), paper.get("abstract", "")):
                sleep(0.1)  # light rate-limit for LLM calls
                continue

            # Download + validate; only count the paper if the PDF lands on disk.
            output_path = pdf_dir / safe_pdf_name(paper["id"], paper)
            if download_and_validate_pdf(paper["pdf_url"], output_path):
                candidates[paper["id"]] = paper
                pbar.update(1)

            sleep(0.1)  # light rate-limit for LLM calls

        # next page
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        sleep(0.5)  # respect OpenAlex rate limit

    pbar.close()
    print(f"\nFinished: {len(candidates)} candidates in {iterations} iterations.")

    # ── Step 5: save ───────────────────────────────────────────────────
    output = {
        "date": today,
        "top_keywords": keywords,
        "keyword_uris": keyword_uris,
        "total_candidates": len(candidates),
        "max_iterations": MAX_ITERATIONS,
        "iterations_used": iterations,
        "pdf_dir": str(pdf_dir),
        "papers": candidates,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metadata to {out_path}")
    print(f"PDFs in {pdf_dir}")


if __name__ == "__main__":
    find_candidate_papers()
