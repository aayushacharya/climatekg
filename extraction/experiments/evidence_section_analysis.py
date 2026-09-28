"""Locate every extracted triplet's evidence paragraph inside the parsed paper
markdown, label which section of the paper it came from, and model how much of
the paper we would actually have to send to the LLM to keep those triplets.

Inputs  : v2/extraction/output/evaluated_papers/<date>/<paper>.json
          v2/extraction/papers/parsed/<date>/<paper>.md
Outputs : experiments/output/evidence_sections/*.json|csv

Run from v2/extraction/:
    python experiments/evidence_section_analysis.py --workers 32
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTION_DIR = os.path.dirname(HERE)
EVAL_DIR = os.path.join(EXTRACTION_DIR, "output", "evaluated_papers")
PARSED_DIR = os.path.join(EXTRACTION_DIR, "papers", "parsed")
OUT_DIR = os.path.join(HERE, "output", "evidence_sections")

# --------------------------------------------------------------------------
# Section vocabulary. olmOCR output carries almost no markdown "#" headings, so
# sections are recognised from a curated vocabulary of header lines instead of
# from generic "short line" heuristics (which fragment badly on two-column OCR).
# --------------------------------------------------------------------------

SECTION_PATTERNS = [
    # (canonical label, regex over the lowercased header text)
    ("abstract", r"^(abstract|summary of the paper|synopsis)$"),
    ("introduction", r"^(introduction|background|motivation|1\s*introduction)$"),
    ("related_work", r"^(related works?|literature review|previous (work|studies|research)|prior work)$"),
    ("data", r"^(data|datasets?|data sources?|study (area|region|site)|observations?|"
             r"observational data|model (output )?data|data description)$"),
    ("methods", r"^(methods?|methodology|methodologies|materials and methods|"
                r"(data|models?|observations?) and methods?|methods? and data|"
                r"model description|model(s)? and (data|methods?|experiments?)|"
                r"experimental (setup|design|methods?)|simulations?|model setup|"
                r"model evaluation|approach|theory|theoretical (framework|background)|"
                r"model and methods|numerical (model|methods?|experiments?)|"
                r"the model|analysis methods?|statistical (methods?|analysis))$"),
    ("results", r"^(results?|results? and analysis|analysis|findings|"
                r"results? and discussions?|main results?|evaluation|"
                r"results? and interpretations?)$"),
    ("discussion", r"^(discussions?|discussions? and implications?|"
                   r"implications?|interpretation|limitations)$"),
    ("conclusion", r"^(conclusions?|concluding remarks|"
                   r"summary|summary and conclusions?|conclusions? and summary|"
                   r"summary and discussions?|discussions? and conclusions?|"
                   r"conclusions? and discussions?|conclusions? and (future work|outlook|"
                   r"perspectives?|recommendations?|implications?)|final remarks|"
                   r"closing remarks)$"),
    ("acknowledgments", r"^(acknowledge?ments?|acknowledgement)$"),
    ("references", r"^(references?|bibliography|references? and notes|literature cited|"
                   r"works cited|reference list)$"),
    ("appendix", r"^(appendix( [a-z0-9]{1,3})?|appendices|supplementary (material|information|"
                 r"data|figures?|tables?)s?|supporting information|online resource|"
                 r"electronic supplementary material)$"),
    ("back_matter", r"^(author contributions?|contributions?|competing interests?|"
                    r"conflicts? of interests?|data availability( statement)?|"
                    r"code availability( statement)?|code and data availability|"
                    r"funding( information| statement)?|additional information|"
                    r"declarations?|disclosure statement|ethics? (statement|approval)|"
                    r"orcid( ids?)?|author information)$"),
]
# Deliberately NOT in the vocabulary: "notes", "abbreviations", "nomenclature",
# "open access", "permissions", "contributions", "about the authors". They occur
# as table footnotes mid-body and in front-matter licence blocks, where treating
# them as headers opens a false back-matter region that swallows real body text
# (and, via body_end, truncates the body early).
SECTION_RE = [(label, re.compile(pat)) for label, pat in SECTION_PATTERNS]

# Coarse grouping used for the cost model.
COARSE = {
    "front_matter": "front",
    "abstract": "front",
    "introduction": "front",
    "related_work": "front",
    "data": "core",
    "methods": "core",
    "results": "core",
    "discussion": "core",
    "conclusion": "conclusion",
    "acknowledgments": "back",
    "references": "back",
    "appendix": "back",
    "back_matter": "back",
    "unlabeled": "unlabeled",
}

# Leading enumeration to strip: "3.", "3.1", "IV.", "(2)", "A."
NUM_PREFIX = re.compile(
    r"^(?:[#*\s]*)(?:\(?(?:\d+(?:\.\d+)*|[ivxlIVXL]{1,5}|[A-Za-z])[\.\)]\s*)?\s*"
)
TRAILING_JUNK = re.compile(r"[\s:.\-–—*_#]+$")


def _canonical(text):
    """Return canonical section label for a candidate header string, or None."""
    t = TRAILING_JUNK.sub("", NUM_PREFIX.sub("", text.strip()))
    t = re.sub(r"\s+", " ", t).strip().lower()
    if not t or len(t) > 60:
        return None
    for label, rx in SECTION_RE:
        if rx.match(t):
            return label
    return None


# Labels allowed to be recognised in the inline "HEADER. body text..." form.
# Restricted to body sections: an inline match is weak evidence, and for tail
# sections it fires on table footnotes ("Notes: ...") far more often than on
# genuine headers.
INLINE_OK = {"abstract", "introduction", "related_work", "data", "methods",
             "results", "discussion", "conclusion"}


def find_sections(doc):
    """Return sorted [(char_offset, label)] of recognised section headers.

    A header is either a standalone short line, or -- for body sections only --
    a line that *starts* with a section name followed by a period/colon (common
    in OCR'd older papers, e.g. "ABSTRACT. A numerical model has been ...").
    """
    hits = []
    offset = 0
    for line in doc.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            label = None
            if len(stripped) <= 70:
                label = _canonical(stripped)
            if label is None:
                # inline header: "ABSTRACT. text..." / "1. Introduction The ..."
                m = re.match(r"^([^.:]{3,40})[.:]\s+\S", stripped)
                if m:
                    label = _canonical(m.group(1))
                    if label not in INLINE_OK:
                        label = None
            if label:
                hits.append((offset, label))
        offset += len(line)
    # collapse consecutive duplicates (OCR repeats headers in page furniture)
    out = []
    for off, lab in hits:
        if out and out[-1][1] == lab and off - out[-1][0] < 200:
            continue
        out.append((off, lab))
    return out


def label_at(sections, pos):
    """Section label covering character offset `pos`."""
    lab = "front_matter"
    for off, l in sections:
        if off <= pos:
            lab = l
        else:
            break
    return lab


def body_end(doc, sections):
    """Offset where the citable body ends (start of references/back matter).

    Uses the *last* references-like header so that in-text mentions of a
    "References" word early in the doc do not truncate the body.
    """
    tail_labels = {"references", "appendix", "back_matter", "acknowledgments"}
    cut = len(doc)
    for off, lab in sections:
        if lab in tail_labels and off > 0.4 * len(doc):
            cut = off
            break
    return cut


# --------------------------------------------------------------------------
# Evidence localisation
# --------------------------------------------------------------------------

def normalize(s):
    """Alphanumeric-only lowercase projection + map back to original offsets."""
    chars = []
    idx = []
    for i, ch in enumerate(s):
        if ch.isalnum():
            chars.append(ch.lower())
            idx.append(i)
    return "".join(chars), idx


def locate(doc_norm, ev_norm):
    """Return normalized offset of evidence in doc, or -1. Degrades to prefix
    and mid-slice matching when the LLM lightly reflowed/truncated the quote."""
    if not ev_norm:
        return -1, "empty"
    p = doc_norm.find(ev_norm)
    if p >= 0:
        return p, "exact"
    for L in (200, 120, 80, 50, 35):
        if len(ev_norm) > L:
            p = doc_norm.find(ev_norm[:L])
            if p >= 0:
                return p, f"prefix{L}"
    if len(ev_norm) > 80:
        m = len(ev_norm) // 2
        p = doc_norm.find(ev_norm[m - 30:m + 30])
        if p >= 0:
            return max(0, p - (m - 30)), "mid"
    return -1, "unmatched"


# --------------------------------------------------------------------------

def process_paper(args):
    json_path, quality = args
    pid = os.path.basename(json_path)[:-5]
    date = os.path.basename(os.path.dirname(json_path))
    md_path = os.path.join(PARSED_DIR, date, pid + ".md")
    rec = {
        "paper_id": pid, "date": date, "n_triplets": 0, "n_kept": 0,
        "n_matched": 0, "rows": [], "doc_chars": 0, "body_chars": 0,
        "section_chars": {}, "n_sections": 0, "no_md": False, "well_formed": False,
    }
    if not os.path.exists(md_path):
        rec["no_md"] = True
        return rec
    doc = open(md_path, errors="ignore").read()
    if not doc.strip():
        rec["no_md"] = True
        return rec

    sections = find_sections(doc)
    doc_norm, nmap = normalize(doc)
    if not doc_norm:
        rec["no_md"] = True
        return rec

    rec["doc_chars"] = len(doc)
    labels_found = {l for _, l in sections}
    rec["n_sections"] = len(labels_found)
    # A paper is "well formed" when OCR recovered enough headers that section
    # attribution is trustworthy: several distinct sections plus an explicit
    # end-of-body marker. Papers below this bar dump later sections into
    # whichever earlier header was detected, biasing the distribution.
    rec["well_formed"] = (
        len(labels_found) >= 4
        and bool(labels_found & {"references", "acknowledgments", "back_matter"})
        and bool(labels_found & {"methods", "data", "results", "discussion"})
    )
    cut = body_end(doc, sections)
    rec["body_chars"] = cut

    # characters attributable to each section (for the cost model)
    bounds = [(0, "front_matter")] + sections + [(len(doc), None)]
    sec_chars = collections.Counter()
    for i in range(len(bounds) - 1):
        lab = bounds[i][1]
        if lab is None:
            break
        sec_chars[lab] += bounds[i + 1][0] - bounds[i][0]
    rec["section_chars"] = dict(sec_chars)

    data = json.load(open(json_path))
    triplets = data.get("kg_extraction_triplets") or []
    rec["n_triplets"] = len(triplets)

    for t in triplets:
        if quality and t.get("eval") is not True:
            continue
        rec["n_kept"] += 1
        ev_norm, _ = normalize(t.get("paragraph") or "")
        pos, how = locate(doc_norm, ev_norm)
        if pos < 0:
            rec["rows"].append({
                "section": "UNMATCHED", "match": how, "rel_body": None,
                "rel_doc": None, "relation": t.get("relation"),
                "source_type": t.get("source_entity_type"),
                "target_type": t.get("target_entity_type"),
            })
            continue
        rec["n_matched"] += 1
        orig = nmap[min(pos, len(nmap) - 1)]
        rec["rows"].append({
            "section": label_at(sections, orig),
            "match": how,
            "rel_doc": orig / len(doc),
            "rel_body": min(1.0, orig / cut) if cut > 0 else None,
            "abs_char": orig,
            "relation": t.get("relation"),
            "source_type": t.get("source_entity_type"),
            "target_type": t.get("target_entity_type"),
        })
    return rec


FRONT = ["front_matter", "abstract"]
INTRO = ["introduction", "related_work"]
CORE = ["methods", "data", "results", "discussion"]
TAIL = ["references", "acknowledgments", "back_matter", "appendix"]


def aggregate(recs, quality):
    """Roll a list of per-paper records into distribution + cost/coverage stats."""
    sec_counts = collections.Counter()
    coarse_counts = collections.Counter()
    match_kinds = collections.Counter()
    decile_body = collections.Counter()
    sec_by_relation = collections.defaultdict(collections.Counter)
    sec_by_srctype = collections.defaultdict(collections.Counter)
    sec_chars = collections.Counter()
    total_doc = total_body = 0
    n_trip = n_kept = n_matched = 0
    rel_body_all = []

    for r in recs:
        n_trip += r["n_triplets"]
        n_kept += r["n_kept"]
        n_matched += r["n_matched"]
        total_doc += r["doc_chars"]
        total_body += r["body_chars"]
        for k, v in r["section_chars"].items():
            sec_chars[k] += v
        for row in r["rows"]:
            match_kinds[row["match"]] += 1
            sec_counts[row["section"]] += 1
            coarse_counts[COARSE.get(row["section"], "unlabeled")] += 1
            if row["section"] != "UNMATCHED":
                sec_by_relation[row["relation"]][row["section"]] += 1
                sec_by_srctype[row["source_type"]][row["section"]] += 1
                rb = row["rel_body"]
                if rb is not None:
                    rel_body_all.append(rb)
                    decile_body[min(9, int(rb * 10))] += 1

    matched_total = sum(v for k, v in sec_counts.items() if k != "UNMATCHED")

    # Evidence yield per section: share of evidence vs share of characters.
    # A yield ratio << 1 means the section costs far more tokens than the
    # facts it contributes -- the sections worth cutting.
    per_section = {}
    for s in set(list(sec_counts) + list(sec_chars)):
        if s == "UNMATCHED":
            continue
        ev = sec_counts[s]
        ch = sec_chars[s]
        per_section[s] = {
            "evidence_count": ev,
            "evidence_share": ev / matched_total if matched_total else 0,
            "chars": ch,
            "char_share": ch / total_doc if total_doc else 0,
            "evidence_per_10k_chars": ev / (ch / 10000) if ch else 0,
            "yield_ratio": ((ev / matched_total) / (ch / total_doc))
                           if matched_total and ch and total_doc else 0,
        }
    per_section = dict(sorted(per_section.items(),
                              key=lambda kv: -kv[1]["evidence_share"]))

    def policy(keep):
        got = sum(sec_counts[s] for s in keep)
        cost = sum(sec_chars[s] for s in keep)
        return {
            "sections": sorted(keep),
            "triplet_recall": got / matched_total if matched_total else 0,
            "char_fraction_of_full_doc": cost / total_doc if total_doc else 0,
            "chars": cost,
        }

    section_policy = {
        "abstract_only": policy(FRONT),
        "abstract+intro": policy(FRONT + INTRO),
        "abstract+intro+conclusion": policy(FRONT + INTRO + ["conclusion"]),
        "front+methods+data": policy(FRONT + INTRO + ["methods", "data"]),
        "front+methods+data+results": policy(FRONT + INTRO + ["methods", "data", "results"]),
        "front+core (no conclusion)": policy(FRONT + INTRO + CORE),
        "body_only (drop refs/ack/backmatter/appendix)": policy(
            FRONT + INTRO + CORE + ["conclusion"]),
        "body_only+appendix": policy(FRONT + INTRO + CORE + ["conclusion", "appendix"]),
        "drop_references_only": policy(
            [s for s in sec_chars if s != "references"]),
        "full_paper": policy(list(sec_chars)),
    }

    rel_body_all.sort()
    n = len(rel_body_all)
    body_share = total_body / total_doc if total_doc else 1.0
    prefix_policy = {}
    for i in range(1, 21):
        frac = i / 20
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if rel_body_all[mid] <= frac:
                lo = mid + 1
            else:
                hi = mid
        prefix_policy[f"{frac:.2f}"] = {
            "triplet_recall": lo / n if n else 0,
            "char_fraction_of_full_doc": frac * body_share,
        }

    summary = {
        "papers": len(recs),
        "triplets_total": n_trip,
        "triplets_considered": n_kept,
        "triplets_localized": n_matched,
        "localization_rate": n_matched / n_kept if n_kept else 0,
        "match_kind": dict(match_kinds.most_common()),
        "per_section": per_section,
        "coarse_counts": dict(coarse_counts.most_common()),
        "body_decile_share": {str(k): decile_body[k] / matched_total if matched_total else 0
                              for k in range(10)},
        "body_decile_counts": {str(k): decile_body[k] for k in range(10)},
        "total_doc_chars": total_doc,
        "total_body_chars": total_body,
        "body_share_of_doc": body_share,
        "policy_by_section": section_policy,
        "policy_by_body_prefix": prefix_policy,
    }
    return (summary,
            {k: dict(v.most_common()) for k, v in sec_by_relation.items()},
            {k: dict(v.most_common()) for k, v in sec_by_srctype.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N papers (0=all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample of N papers across all dates (0=all)")
    ap.add_argument("--all-triplets", action="store_true",
                    help="include triplets that failed evaluation (default: eval==True only)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(EVAL_DIR, "*", "*.json")))
    # A paper id can appear in several date partitions (re-runs). entity_linking.py
    # keeps the latest partition per id, so the KG holds one copy; mirror that rule
    # here or the duplicated papers get double-counted.
    by_id = {}
    for f in files:
        by_id[os.path.basename(f)[:-5]] = f  # sorted order -> latest date wins
    n_dupe_files = len(files) - len(by_id)
    files = sorted(by_id.values())
    print(f"deduplicated {n_dupe_files} repeat partition copies -> {len(files)} unique papers")

    if args.limit:
        files = files[:args.limit]
    if args.sample and args.sample < len(files):
        import random
        random.seed(0)
        files = sorted(random.sample(files, args.sample))
    print(f"papers: {len(files)}")
    quality = not args.all_triplets

    with Pool(args.workers) as pool:
        recs = pool.map(process_paper, [(f, quality) for f in files], chunksize=20)

    os.makedirs(OUT_DIR, exist_ok=True)

    usable = [r for r in recs if not r["no_md"]]
    n_no_md = sum(1 for r in recs if r["no_md"])
    well = [r for r in usable if r["well_formed"]]

    all_summary, rel_all, src_all = aggregate(usable, quality)
    well_summary, rel_well, src_well = aggregate(well, quality)

    summary = {
        "papers_total": len(files),
        "duplicate_partition_copies_dropped": n_dupe_files,
        "papers_without_markdown": n_no_md,
        "papers_usable": len(usable),
        "papers_well_formed": len(well),
        "well_formed_share": len(well) / len(usable) if usable else 0,
        "quality_filter": "eval==True" if quality else "none",
        "papers_by_n_sections_detected": dict(sorted(
            collections.Counter(r["n_sections"] for r in usable).items())),
        "all_papers": all_summary,
        "well_formed_papers": well_summary,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(OUT_DIR, "section_by_relation.json"), "w") as fh:
        json.dump({"all_papers": rel_all, "well_formed_papers": rel_well}, fh, indent=2)
    with open(os.path.join(OUT_DIR, "section_by_source_type.json"), "w") as fh:
        json.dump({"all_papers": src_all, "well_formed_papers": src_well}, fh, indent=2)

    per_paper = [{
        "paper_id": r["paper_id"], "date": r["date"], "n_triplets": r["n_triplets"],
        "n_kept": r["n_kept"], "n_matched": r["n_matched"], "doc_chars": r["doc_chars"],
        "body_chars": r["body_chars"], "n_sections": r["n_sections"],
        "well_formed": int(r["well_formed"]),
    } for r in usable]

    with open(os.path.join(OUT_DIR, "per_paper.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_paper[0].keys()))
        w.writeheader()
        w.writerows(per_paper)

    for k in ("papers_total", "papers_without_markdown", "papers_usable",
              "papers_well_formed", "well_formed_share"):
        print(f"{k:28s} {summary[k]}")

    for cohort in ("all_papers", "well_formed_papers"):
        s = summary[cohort]
        print("\n" + "=" * 78)
        print(f"COHORT: {cohort}  ({s['papers']} papers, "
              f"{s['triplets_localized']}/{s['triplets_considered']} triplets localized "
              f"= {s['localization_rate']*100:.1f}%)")
        print("=" * 78)
        print(f"{'section':18s} {'evidence%':>10s} {'chars%':>8s} {'ev/10k ch':>10s} {'yield':>7s}")
        for k, v in s["per_section"].items():
            print(f"  {k:16s} {v['evidence_share']*100:9.2f}% {v['char_share']*100:7.2f}%"
                  f" {v['evidence_per_10k_chars']:10.2f} {v['yield_ratio']:7.2f}")
        print("\n  body-prefix truncation:")
        for k, v in s["policy_by_body_prefix"].items():
            if int(round(float(k) * 20)) % 2:
                continue
            print(f"    first {float(k)*100:5.1f}% of body -> recall {v['triplet_recall']*100:6.2f}%"
                  f"  cost {v['char_fraction_of_full_doc']*100:5.1f}% of doc")
        print("\n  section-selection policies:")
        for k, v in s["policy_by_section"].items():
            print(f"    {k:46s} recall {v['triplet_recall']*100:6.2f}%"
                  f"  cost {v['char_fraction_of_full_doc']*100:5.1f}%")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
