#!/usr/bin/env python3
"""Check whether missed CMIP More Info entities occur in parsed paper text.

This complements `evaluate_cmip_moreinfo_entity_extraction.py`: for every
false-negative ground-truth entity, it searches the corresponding parsed
Markdown paper and records whether the entity's surface form appears there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from evaluate_cmip_moreinfo_entity_extraction import (
    ENTITY_TYPES,
    normalize_entity,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
DEFAULT_EVAL_DIR = (
    SCRIPT_DIR
    / "output"
    / "cmip_moreinfo_entity_evaluation"
    / "20260612_101052_cc460532_eval"
)
DEFAULT_COMPARISON = DEFAULT_EVAL_DIR / "paper_entity_comparison.json"
DEFAULT_PARSED_DIR = EXTRACTION_DIR / "papers" / "parsed" / "2026-02-01"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "parsed_ground_truth_presence"
DEFAULT_PIPELINE_DIRS = [
    EXTRACTION_DIR / "output" / "extracted_papers" / "2026-02-01",
    EXTRACTION_DIR / "output" / "validated_papers" / "2026-02-01",
    EXTRACTION_DIR / "output" / "ontology_conformed_papers" / "2026-02-01",
    EXTRACTION_DIR / "output" / "evidence_validated_papers" / "2026-02-01",
    EXTRACTION_DIR / "output" / "evaluated_papers" / "2026-02-01",
]

GT_TO_PRED_TYPE = {
    "activities": "Activity",
    "experiments": "Experiment",
    "sources": "Source",
    "frequencies": "Frequency",
}
PRED_TO_GT_TYPE = {value: key for key, value in GT_TO_PRED_TYPE.items()}

SPACE_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
DIGIT_ALPHA_RE = re.compile(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)")
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


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


def normalize_search_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).translate(DASH_TRANSLATION)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return SPACE_RE.sub(" ", text)


def query_tokens(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", value)
    text = CAMEL_RE.sub(" ", text)
    text = DIGIT_ALPHA_RE.sub(" ", text)
    return re.findall(r"[a-z0-9]+", text.casefold())


def compact_with_map(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    char_to_original: list[int] = []
    for index, char in enumerate(value):
        if char.isalnum():
            chars.append(char.casefold())
            char_to_original.append(index)
    return "".join(chars), char_to_original


def snippet(text: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    value = text[left:right].replace("\n", " ")
    return SPACE_RE.sub(" ", value).strip()


def find_surface_form(text: str, query: str) -> dict[str, Any]:
    """Find a query in text with exact, flexible-token, and compact matching."""

    normalized_text = normalize_search_text(text)
    normalized_query = normalize_search_text(query).strip()
    if not normalized_query:
        return {"matched": False, "match_type": "none", "count": 0, "snippet": None}

    folded_text = normalized_text.casefold()
    folded_query = normalized_query.casefold()
    exact_index = folded_text.find(folded_query)
    if exact_index >= 0:
        count = folded_text.count(folded_query)
        return {
            "matched": True,
            "match_type": "casefold_substring",
            "count": count,
            "snippet": snippet(
                normalized_text,
                exact_index,
                exact_index + len(folded_query),
            ),
        }

    tokens = query_tokens(query)
    if tokens:
        pattern = r"(?<![a-z0-9])" + r"[^a-z0-9]*".join(
            re.escape(token) for token in tokens
        ) + r"(?![a-z0-9])"
        matches = list(re.finditer(pattern, folded_text))
        if matches:
            first = matches[0]
            return {
                "matched": True,
                "match_type": "flexible_token_sequence",
                "count": len(matches),
                "snippet": snippet(normalized_text, first.start(), first.end()),
            }

    compact_query = "".join(tokens)
    if len(compact_query) >= 3:
        compact_text, char_to_original = compact_with_map(normalized_text)
        compact_index = compact_text.find(compact_query)
        if compact_index >= 0:
            compact_end = compact_index + len(compact_query) - 1
            start = char_to_original[compact_index]
            end = char_to_original[compact_end] + 1
            return {
                "matched": True,
                "match_type": "compact_alnum",
                "count": compact_text.count(compact_query),
                "snippet": snippet(normalized_text, start, end),
            }

    return {"matched": False, "match_type": "none", "count": 0, "snippet": None}


def norm(value: str, entity_type: str) -> str:
    return normalize_entity(value, entity_type, use_frequency_aliases=True)


def similar_enough(left: str, right: str, threshold: float) -> bool:
    if not left or not right or left == right:
        return False
    left_compact = re.sub(r"[^a-z0-9]+", "", left)
    right_compact = re.sub(r"[^a-z0-9]+", "", right)
    if left_compact == right_compact and left_compact:
        return True
    if SequenceMatcher(None, left, right).ratio() >= threshold:
        return True
    if SequenceMatcher(None, left_compact, right_compact).ratio() >= threshold:
        return True
    shorter, longer = sorted([left_compact, right_compact], key=len)
    return len(shorter) >= 5 and shorter in longer and len(shorter) / len(longer) >= 0.55


def collect_typed_labels(value: Any) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {entity_type: set() for entity_type in ENTITY_TYPES}

    def add(label: Any, pred_type: Any) -> None:
        gt_type = PRED_TO_GT_TYPE.get(str(pred_type))
        if gt_type and label:
            labels[gt_type].add(str(label))

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            add(obj.get("entity_instance"), obj.get("entity_type"))
            add(obj.get("source_entity_instance"), obj.get("source_entity_type"))
            add(obj.get("target_entity_instance"), obj.get("target_entity_type"))
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(value)
    if isinstance(value, dict):
        for model_name in value.get("model") or []:
            labels["sources"].add(str(model_name))
    return labels


def merge_label_maps(left: dict[str, set[str]], right: dict[str, set[str]]) -> None:
    for entity_type, values in right.items():
        left[entity_type].update(values)


def load_pipeline_context(
    paper_ids: set[str],
    pipeline_dirs: list[Path],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    typed_labels = {
        paper_id: {entity_type: set() for entity_type in ENTITY_TYPES}
        for paper_id in paper_ids
    }
    raw_text = {paper_id: "" for paper_id in paper_ids}

    for paper_id in sorted(paper_ids):
        raw_chunks: list[str] = []
        for pipeline_dir in pipeline_dirs:
            path = pipeline_dir / f"{paper_id}.json"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            raw_chunks.append(text)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            merge_label_maps(typed_labels[paper_id], collect_typed_labels(data))
        raw_text[paper_id] = "\n".join(raw_chunks)

    return typed_labels, raw_text


def best_near_match(
    query: str,
    candidates: set[str],
    entity_type: str,
    threshold: float,
) -> dict[str, Any] | None:
    query_norm = norm(query, entity_type)
    best: dict[str, Any] | None = None
    for candidate in candidates:
        candidate_norm = norm(candidate, entity_type)
        if not similar_enough(query_norm, candidate_norm, threshold):
            continue
        score = max(
            SequenceMatcher(None, query_norm, candidate_norm).ratio(),
            SequenceMatcher(
                None,
                re.sub(r"[^a-z0-9]+", "", query_norm),
                re.sub(r"[^a-z0-9]+", "", candidate_norm),
            ).ratio(),
        )
        if best is None or score > best["score"]:
            best = {
                "label": candidate,
                "normalized_label": candidate_norm,
                "score": score,
            }
    return best


def classify_downstream_status(
    paper_record: dict[str, Any],
    entity_type: str,
    missed_label: str,
    pipeline_typed_labels: dict[str, set[str]],
    pipeline_raw_text: str,
    near_threshold: float,
) -> dict[str, Any]:
    final_same_type = set(paper_record.get("extracted", {}).get(entity_type) or [])
    near_final = best_near_match(
        missed_label,
        final_same_type,
        entity_type,
        near_threshold,
    )
    if near_final:
        return {"status": "final_same_type_near_label", "match": near_final}

    missed_norm = norm(missed_label, entity_type)
    for other_type, labels in (paper_record.get("extracted", {}) or {}).items():
        if other_type == entity_type:
            continue
        for label in labels or []:
            if norm(label, other_type) == missed_norm:
                return {
                    "status": "final_wrong_type_exact",
                    "match": {"label": label, "entity_type": other_type},
                }

    raw_mention = find_surface_form(pipeline_raw_text, missed_label)
    if raw_mention["matched"]:
        return {"status": "mentioned_in_pipeline_record_not_entity", "match": raw_mention}

    near_raw = best_near_match(
        missed_label,
        pipeline_typed_labels.get(entity_type, set()),
        entity_type,
        near_threshold,
    )
    if near_raw:
        return {"status": "raw_same_type_near_label", "match": near_raw}

    return {"status": "not_found_exact_or_near_in_pipeline_outputs", "match": None}


def missed_ground_truth_records(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for paper_id, paper_record in sorted(comparison.items()):
        comparison_by_type = paper_record.get("comparison_by_type") or {}
        for entity_type in ENTITY_TYPES:
            type_record = comparison_by_type.get(entity_type) or {}
            for label in type_record.get("missed_ground_truth") or []:
                records.append(
                    {
                        "paper_id": paper_id,
                        "title": paper_record.get("title"),
                        "doi": paper_record.get("doi"),
                        "site_id": paper_record.get("site_id"),
                        "entity_type": entity_type,
                        "ground_truth_entity": label,
                        "normalized_entity": norm(label, entity_type),
                    }
                )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_status: Counter[str] = Counter()
    by_status_and_parsed: dict[str, Counter[str]] = defaultdict(Counter)
    match_type_counts: Counter[str] = Counter()

    for record in records:
        entity_type = record["entity_type"]
        parsed_key = "parsed_present" if record["parsed_presence"]["matched"] else "parsed_absent"
        by_type[entity_type][parsed_key] += 1
        by_type[entity_type]["total"] += 1
        if record["parsed_presence"]["matched"]:
            match_type_counts[record["parsed_presence"]["match_type"]] += 1
        status = record["downstream_status"]
        by_status[status] += 1
        by_status_and_parsed[status][parsed_key] += 1
        by_status_and_parsed[status]["total"] += 1

    total = len(records)
    parsed_present = sum(
        1 for record in records if record["parsed_presence"]["matched"]
    )
    return {
        "missed_ground_truth_total": total,
        "parsed_present_total": parsed_present,
        "parsed_absent_total": total - parsed_present,
        "parsed_present_rate": parsed_present / total if total else 0.0,
        "parsed_presence_by_type": {
            entity_type: {
                **counts,
                "parsed_present_rate": (
                    counts["parsed_present"] / counts["total"]
                    if counts["total"]
                    else 0.0
                ),
            }
            for entity_type, counts in sorted(by_type.items())
        },
        "parsed_match_type_counts": dict(sorted(match_type_counts.items())),
        "downstream_status_counts": dict(sorted(by_status.items())),
        "downstream_status_by_parsed_presence": {
            status: {
                **counts,
                "parsed_present_rate": (
                    counts["parsed_present"] / counts["total"]
                    if counts["total"]
                    else 0.0
                ),
            }
            for status, counts in sorted(by_status_and_parsed.items())
        },
    }


def group_by_paper(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for record in records:
        paper_id = record["paper_id"]
        paper = grouped.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "title": record["title"],
                "doi": record["doi"],
                "site_id": record["site_id"],
                "missed_ground_truth": {entity_type: [] for entity_type in ENTITY_TYPES},
            },
        )
        paper["missed_ground_truth"][record["entity_type"]].append(
            {
                "ground_truth_entity": record["ground_truth_entity"],
                "normalized_entity": record["normalized_entity"],
                "parsed_presence": record["parsed_presence"],
                "downstream_status": record["downstream_status"],
                "downstream_match": record["downstream_match"],
            }
        )
    return grouped


def calculate_ground_truth_present_only_recall(
    comparison: dict[str, Any],
    parsed_dir: Path,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {
        entity_type: {
            "ground_truth_present_in_parsed": 0,
            "correct_present_in_parsed": 0,
            "missed_present_in_parsed": 0,
            "ground_truth_absent_from_parsed": 0,
            "recall_on_present_ground_truth": 0.0,
        }
        for entity_type in ENTITY_TYPES
    }
    metrics["micro"] = {
        "ground_truth_present_in_parsed": 0,
        "correct_present_in_parsed": 0,
        "missed_present_in_parsed": 0,
        "ground_truth_absent_from_parsed": 0,
        "recall_on_present_ground_truth": 0.0,
    }
    records: list[dict[str, Any]] = []
    parsed_cache: dict[str, str | None] = {}

    for paper_id, paper_record in sorted(comparison.items()):
        if paper_id not in parsed_cache:
            parsed_path = parsed_dir / f"{paper_id}.md"
            parsed_cache[paper_id] = (
                parsed_path.read_text(encoding="utf-8")
                if parsed_path.exists()
                else None
            )
        parsed_text = parsed_cache[paper_id]

        for entity_type in ENTITY_TYPES:
            correct_norms = {
                item.get("normalized_entity")
                or norm(item.get("ground_truth", ""), entity_type)
                for item in (
                    paper_record.get("comparison_by_type", {})
                    .get(entity_type, {})
                    .get("correctly_extracted", [])
                )
            }
            for ground_truth_label in (
                paper_record.get("ground_truth", {}).get(entity_type, []) or []
            ):
                normalized = norm(ground_truth_label, entity_type)
                if parsed_text is None:
                    parsed_presence = {
                        "matched": False,
                        "match_type": "missing_parsed_file",
                        "count": 0,
                        "snippet": None,
                    }
                else:
                    parsed_presence = find_surface_form(
                        parsed_text,
                        ground_truth_label,
                    )
                correctly_extracted = normalized in correct_norms

                for metric_key in (entity_type, "micro"):
                    bucket = metrics[metric_key]
                    if parsed_presence["matched"]:
                        bucket["ground_truth_present_in_parsed"] += 1
                        if correctly_extracted:
                            bucket["correct_present_in_parsed"] += 1
                        else:
                            bucket["missed_present_in_parsed"] += 1
                    else:
                        bucket["ground_truth_absent_from_parsed"] += 1

                records.append(
                    {
                        "paper_id": paper_id,
                        "title": paper_record.get("title"),
                        "doi": paper_record.get("doi"),
                        "site_id": paper_record.get("site_id"),
                        "entity_type": entity_type,
                        "ground_truth_entity": ground_truth_label,
                        "normalized_entity": normalized,
                        "present_in_parsed": parsed_presence["matched"],
                        "match_type": parsed_presence["match_type"],
                        "correctly_extracted": correctly_extracted,
                        "snippet": parsed_presence.get("snippet"),
                    }
                )

    for bucket in metrics.values():
        present = bucket["ground_truth_present_in_parsed"]
        bucket["recall_on_present_ground_truth"] = (
            bucket["correct_present_in_parsed"] / present if present else 0.0
        )
        bucket["present_denominator_check"] = (
            bucket["correct_present_in_parsed"] + bucket["missed_present_in_parsed"]
        )

    return {
        "definition": (
            "Recall after excluding ground-truth entities whose ground-truth "
            "surface form is absent from the parsed markdown paper."
        ),
        "parsed_dir": str(parsed_dir),
        "missing_parsed_files": sum(
            1 for parsed_text in parsed_cache.values() if parsed_text is None
        ),
        "metrics": metrics,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search parsed Markdown papers for false-negative CMIP More Info "
            "ground-truth entities."
        )
    )
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        action="append",
        dest="pipeline_dirs",
        help=(
            "A per-paper JSON output directory to inspect for downstream labels. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument("--near-threshold", type=float, default=0.82)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = read_json(args.comparison)
    records = missed_ground_truth_records(comparison)
    paper_ids = {record["paper_id"] for record in records}
    pipeline_dirs = args.pipeline_dirs or DEFAULT_PIPELINE_DIRS
    pipeline_typed_labels, pipeline_raw_text = load_pipeline_context(
        paper_ids,
        pipeline_dirs,
    )

    parsed_cache: dict[str, str | None] = {}
    for record in records:
        paper_id = record["paper_id"]
        if paper_id not in parsed_cache:
            parsed_path = args.parsed_dir / f"{paper_id}.md"
            parsed_cache[paper_id] = (
                parsed_path.read_text(encoding="utf-8")
                if parsed_path.exists()
                else None
            )
        parsed_text = parsed_cache[paper_id]
        if parsed_text is None:
            parsed_presence = {
                "matched": False,
                "match_type": "missing_parsed_file",
                "count": 0,
                "snippet": None,
            }
        else:
            parsed_presence = find_surface_form(
                parsed_text,
                record["ground_truth_entity"],
            )

        paper_record = comparison[paper_id]
        downstream = classify_downstream_status(
            paper_record=paper_record,
            entity_type=record["entity_type"],
            missed_label=record["ground_truth_entity"],
            pipeline_typed_labels=pipeline_typed_labels[paper_id],
            pipeline_raw_text=pipeline_raw_text[paper_id],
            near_threshold=args.near_threshold,
        )
        record["parsed_presence"] = parsed_presence
        record["downstream_status"] = downstream["status"]
        record["downstream_match"] = downstream["match"]

    summary = summarize(records)
    cause1_records = [
        record
        for record in records
        if record["downstream_status"] == "not_found_exact_or_near_in_pipeline_outputs"
    ]
    summary["cause1_not_found_exact_or_near_in_pipeline_outputs"] = summarize(
        cause1_records
    )
    summary["input"] = {
        "comparison": str(args.comparison),
        "parsed_dir": str(args.parsed_dir),
        "pipeline_dirs": [str(path) for path in pipeline_dirs],
        "near_threshold": args.near_threshold,
    }
    summary["parsed_files"] = {
        "papers_with_false_negatives": len(paper_ids),
        "found": sum(1 for value in parsed_cache.values() if value is not None),
        "missing": sum(1 for value in parsed_cache.values() if value is None),
    }

    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "missed_ground_truth_parsed_presence.json", records)
    write_json(args.output_dir / "paper_missing_ground_truth_parsed_presence.json", group_by_paper(records))
    write_json(args.output_dir / "cause1_records.json", cause1_records)
    present_only_recall = calculate_ground_truth_present_only_recall(
        comparison,
        args.parsed_dir,
    )
    present_only_records = present_only_recall.pop("records")
    write_json(
        args.output_dir / "recall_ground_truth_present_only.json",
        present_only_recall,
    )
    write_json(
        args.output_dir / "ground_truth_parsed_presence_all.json",
        present_only_records,
    )

    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
