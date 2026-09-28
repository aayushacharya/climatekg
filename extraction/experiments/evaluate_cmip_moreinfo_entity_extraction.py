#!/usr/bin/env python3
"""Evaluate extracted paper entities against CMIP Publication Hub More Info.

Ground truth is the More Info extraction produced by
`extract_cmip_publication_moreinfo_entities.py`. Evaluation is restricted to:
- the OpenAlex papers present in that ground-truth file
- the More Info entity families: activities, experiments, sources, frequencies
- extracted paper entities under `entities_extracted`

The primary metric is paper-level entity matching: an entity is correct only if
the same normalized entity appears for the same paper and entity type.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXTRACTION_DIR.parent

DEFAULT_GROUND_TRUTH_ROOT = (
    SCRIPT_DIR / "output" / "cmip_publication_moreinfo_entities"
)
DEFAULT_PREDICTIONS_PATH = (
    EXTRACTION_DIR
    / "output"
    / "entity_linked_papers"
    / "v2_processed_papers_entity_linking_post_llm.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "cmip_moreinfo_entity_evaluation"

GT_TO_PRED_TYPE = {
    "activities": "Activity",
    "experiments": "Experiment",
    "sources": "Source",
    "frequencies": "Frequency",
}
ENTITY_TYPES = ["activities", "experiments", "sources", "frequencies"]

SPACE_RE = re.compile(r"\s+")
OUTER_PUNCT_RE = re.compile(r"^[\s\"'`.,;:()\[\]{}<>]+|[\s\"'`.,;:()\[\]{}<>]+$")

FREQUENCY_ALIASES = {
    "mon": "monthly",
    "month": "monthly",
    "monthly": "monthly",
    "monthly mean": "monthly",
    "monthly means": "monthly",
    "day": "daily",
    "1 day": "daily",
    "daily": "daily",
    "daily mean": "daily",
    "yr": "yearly",
    "year": "yearly",
    "yearly": "yearly",
    "annual": "yearly",
    "ann": "yearly",
    "3 h": "3 hourly",
    "3 hr": "3 hourly",
    "3hr": "3 hourly",
    "3 hourly": "3 hourly",
    "3 hourly mean": "3 hourly",
    "6 h": "6 hourly",
    "6 hr": "6 hourly",
    "6hr": "6 hourly",
    "6 hourly": "6 hourly",
    "6 hourly mean": "6 hourly",
    "fx": "fixed",
    "fixed": "fixed",
}


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
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_base(value: Any) -> str:
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
    text = OUTER_PUNCT_RE.sub("", text)
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_experiment(value: Any) -> str:
    text = normalize_base(value)
    text = re.sub(r"\brcp\s*([0-9])\s*([0-9])\b", r"rcp\1\2", text)
    text = re.sub(r"\bssp\s*([0-9])\s*([0-9])\s*([0-9])\b", r"ssp\1\2\3", text)
    return text


def normalize_frequency(value: Any, use_aliases: bool) -> str:
    text = normalize_base(value)
    if not use_aliases:
        return text
    return FREQUENCY_ALIASES.get(text, text)


def normalize_entity(value: Any, entity_type: str, use_frequency_aliases: bool) -> str:
    if entity_type == "experiments":
        return normalize_experiment(value)
    if entity_type == "frequencies":
        return normalize_frequency(value, use_frequency_aliases)
    return normalize_base(value)


def bare_openalex_id(value: str) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def resolve_latest_ground_truth(root: Path) -> Path:
    candidates = sorted(
        root.glob("*/paper_moreinfo_records.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [path for path in candidates if path.parent.name != "smoke"]
    if not candidates:
        raise FileNotFoundError(
            f"No paper_moreinfo_records.json files found under {root}"
        )
    return candidates[0]


def load_ground_truth(
    ground_truth_path: Path,
    use_frequency_aliases: bool,
) -> tuple[
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    records = read_json(ground_truth_path)
    if not isinstance(records, list):
        raise ValueError(f"Ground truth must be a list: {ground_truth_path}")

    truth: dict[str, dict[str, dict[str, str]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    excluded_without_ground_truth: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        paper_id = bare_openalex_id(record.get("openalex_id", ""))
        if not paper_id:
            continue
        more_info = record.get("more_info") or {}
        typed_truth: dict[str, dict[str, str]] = {entity_type: {} for entity_type in ENTITY_TYPES}
        for entity_type in ENTITY_TYPES:
            values = more_info.get(entity_type) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                name = display_text(value)
                normalized = normalize_entity(name, entity_type, use_frequency_aliases)
                if normalized and normalized not in typed_truth[entity_type]:
                    typed_truth[entity_type][normalized] = name
        if not any(typed_truth[entity_type] for entity_type in ENTITY_TYPES):
            excluded_without_ground_truth.append(
                {
                    "paper_id": paper_id,
                    "openalex_id": record.get("openalex_id"),
                    "title": record.get("title"),
                    "doi": record.get("doi"),
                    "site_id": record.get("site_id"),
                }
            )
            continue
        truth[paper_id] = typed_truth
        metadata[paper_id] = {
            "openalex_id": record.get("openalex_id"),
            "title": record.get("title"),
            "doi": record.get("doi"),
            "site_id": record.get("site_id"),
            "site_title": record.get("site_title"),
            "site_doi": record.get("site_doi"),
        }
    return truth, metadata, excluded_without_ground_truth


def load_predictions(
    predictions_path: Path,
    paper_ids: set[str],
    use_frequency_aliases: bool,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, int]]]:
    data = read_json(predictions_path)
    if not isinstance(data, dict):
        raise ValueError(f"Predictions must be a dictionary: {predictions_path}")

    pred_type_to_gt = {pred_type: gt_type for gt_type, pred_type in GT_TO_PRED_TYPE.items()}
    predictions: dict[str, dict[str, dict[str, str]]] = {
        paper_id: {entity_type: {} for entity_type in ENTITY_TYPES} for paper_id in paper_ids
    }
    skipped_counts: dict[str, Counter[str]] = {
        paper_id: Counter() for paper_id in paper_ids
    }
    for paper_id in paper_ids:
        record = data.get(paper_id) or {}
        entities = record.get("entities_extracted") or []
        if not isinstance(entities, list):
            skipped_counts[paper_id]["entities_extracted_not_list"] += 1
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                skipped_counts[paper_id]["entity_not_object"] += 1
                continue
            pred_type = str(entity.get("entity_type") or "")
            gt_type = pred_type_to_gt.get(pred_type)
            if gt_type is None:
                skipped_counts[paper_id]["ignored_other_type"] += 1
                continue
            name = display_text(entity.get("entity_instance"))
            normalized = normalize_entity(name, gt_type, use_frequency_aliases)
            if not normalized:
                skipped_counts[paper_id]["empty_relevant_entity"] += 1
                continue
            predictions[paper_id][gt_type].setdefault(normalized, name)
    skipped = {
        paper_id: dict(counter)
        for paper_id, counter in skipped_counts.items()
        if counter
    }
    return predictions, skipped


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    truth: dict[str, dict[str, dict[str, str]]],
    predictions: dict[str, dict[str, dict[str, str]]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_type_counts = {entity_type: Counter() for entity_type in ENTITY_TYPES}
    per_paper_records: list[dict[str, Any]] = []
    paper_comparisons: dict[str, dict[str, Any]] = {}
    detailed_records: list[dict[str, Any]] = []
    global_truth: dict[str, dict[str, str]] = {entity_type: {} for entity_type in ENTITY_TYPES}
    global_predictions: dict[str, dict[str, str]] = {
        entity_type: {} for entity_type in ENTITY_TYPES
    }

    for paper_id in sorted(truth):
        paper_truth = truth[paper_id]
        paper_predictions = predictions.get(paper_id) or {
            entity_type: {} for entity_type in ENTITY_TYPES
        }
        paper_counts = Counter()
        paper_by_type: dict[str, dict[str, Any]] = {}
        paper_entity_comparison: dict[str, dict[str, Any]] = {}
        paper_ground_truth: dict[str, list[str]] = {}
        paper_extracted: dict[str, list[str]] = {}
        for entity_type in ENTITY_TYPES:
            truth_entities = paper_truth.get(entity_type, {})
            pred_entities = paper_predictions.get(entity_type, {})
            truth_set = set(truth_entities)
            pred_set = set(pred_entities)
            tp_set = truth_set & pred_set
            fp_set = pred_set - truth_set
            fn_set = truth_set - pred_set

            for key in truth_set:
                global_truth[entity_type].setdefault(key, truth_entities[key])
            for key in pred_set:
                global_predictions[entity_type].setdefault(key, pred_entities[key])

            counts = {
                "true_positive": len(tp_set),
                "false_positive": len(fp_set),
                "false_negative": len(fn_set),
            }
            by_type_counts[entity_type].update(counts)
            paper_counts.update(counts)
            paper_by_type[entity_type] = {
                **prf(len(tp_set), len(fp_set), len(fn_set)),
                "ground_truth_count": len(truth_set),
                "predicted_count": len(pred_set),
            }
            paper_ground_truth[entity_type] = sorted(
                (truth_entities[key] for key in truth_set),
                key=str.casefold,
            )
            paper_extracted[entity_type] = sorted(
                (pred_entities[key] for key in pred_set),
                key=str.casefold,
            )
            paper_entity_comparison[entity_type] = {
                "correctly_extracted": [
                    {
                        "ground_truth": truth_entities[key],
                        "extracted": pred_entities[key],
                        "normalized_entity": key,
                    }
                    for key in sorted(tp_set)
                ],
                "extra_extracted": sorted(
                    (pred_entities[key] for key in fp_set),
                    key=str.casefold,
                ),
                "missed_ground_truth": sorted(
                    (truth_entities[key] for key in fn_set),
                    key=str.casefold,
                ),
                "metrics": paper_by_type[entity_type],
            }

            for label, keys, source in (
                ("true_positive", tp_set, "both"),
                ("false_positive", fp_set, "prediction"),
                ("false_negative", fn_set, "ground_truth"),
            ):
                for normalized in sorted(keys):
                    detail: dict[str, Any] = {
                        "paper_id": paper_id,
                        "openalex_id": metadata.get(paper_id, {}).get("openalex_id"),
                        "title": metadata.get(paper_id, {}).get("title"),
                        "entity_type": entity_type,
                        "status": label,
                        "normalized_entity": normalized,
                    }
                    if source in {"both", "ground_truth"}:
                        detail["ground_truth_entity"] = truth_entities.get(normalized)
                    if source in {"both", "prediction"}:
                        detail["predicted_entity"] = pred_entities.get(normalized)
                    detailed_records.append(detail)

        per_paper_records.append(
            {
                "paper_id": paper_id,
                "openalex_id": metadata.get(paper_id, {}).get("openalex_id"),
                "title": metadata.get(paper_id, {}).get("title"),
                "doi": metadata.get(paper_id, {}).get("doi"),
                "site_id": metadata.get(paper_id, {}).get("site_id"),
                "metrics": prf(
                    paper_counts["true_positive"],
                    paper_counts["false_positive"],
                    paper_counts["false_negative"],
                ),
                "ground_truth_count": paper_counts["true_positive"]
                + paper_counts["false_negative"],
                "predicted_count": paper_counts["true_positive"]
                + paper_counts["false_positive"],
                "by_type": paper_by_type,
            }
        )
        paper_comparisons[paper_id] = {
            "paper_id": paper_id,
            "openalex_id": metadata.get(paper_id, {}).get("openalex_id"),
            "title": metadata.get(paper_id, {}).get("title"),
            "doi": metadata.get(paper_id, {}).get("doi"),
            "site_id": metadata.get(paper_id, {}).get("site_id"),
            "site_title": metadata.get(paper_id, {}).get("site_title"),
            "site_doi": metadata.get(paper_id, {}).get("site_doi"),
            "overall_metrics": prf(
                paper_counts["true_positive"],
                paper_counts["false_positive"],
                paper_counts["false_negative"],
            ),
            "ground_truth_count": paper_counts["true_positive"]
            + paper_counts["false_negative"],
            "extracted_count": paper_counts["true_positive"]
            + paper_counts["false_positive"],
            "ground_truth": paper_ground_truth,
            "extracted": paper_extracted,
            "comparison_by_type": paper_entity_comparison,
        }

    metrics_by_type = {
        entity_type: prf(
            counts["true_positive"],
            counts["false_positive"],
            counts["false_negative"],
        )
        for entity_type, counts in by_type_counts.items()
    }
    total_tp = sum(counts["true_positive"] for counts in by_type_counts.values())
    total_fp = sum(counts["false_positive"] for counts in by_type_counts.values())
    total_fn = sum(counts["false_negative"] for counts in by_type_counts.values())
    macro_precision = sum(item["precision"] for item in metrics_by_type.values()) / len(ENTITY_TYPES)
    macro_recall = sum(item["recall"] for item in metrics_by_type.values()) / len(ENTITY_TYPES)
    macro_f1 = sum(item["f1"] for item in metrics_by_type.values()) / len(ENTITY_TYPES)

    unique_by_type: dict[str, dict[str, Any]] = {}
    unique_total_tp = unique_total_fp = unique_total_fn = 0
    for entity_type in ENTITY_TYPES:
        truth_set = set(global_truth[entity_type])
        pred_set = set(global_predictions[entity_type])
        tp = len(truth_set & pred_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        unique_total_tp += tp
        unique_total_fp += fp
        unique_total_fn += fn
        unique_by_type[entity_type] = {
            **prf(tp, fp, fn),
            "ground_truth_unique_count": len(truth_set),
            "predicted_unique_count": len(pred_set),
        }

    return {
        "summary": {
            "paper_count": len(truth),
            "entity_types_evaluated": ENTITY_TYPES,
            "paper_level_micro": prf(total_tp, total_fp, total_fn),
            "paper_level_macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "global_unique_micro": prf(
                unique_total_tp,
                unique_total_fp,
                unique_total_fn,
            ),
        },
        "metrics_by_type": metrics_by_type,
        "global_unique_by_type": unique_by_type,
        "per_paper": per_paper_records,
        "paper_comparisons": paper_comparisons,
        "details": detailed_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate entities_extracted against CMIP Publication Hub More Info ground truth."
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help=(
            "Path to paper_moreinfo_records.json. Defaults to the newest non-smoke "
            "cmip_publication_moreinfo_entities output."
        ),
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Root used to auto-discover the newest ground-truth output.",
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--strict-frequency",
        action="store_true",
        help="Disable small aliases such as mon->Monthly and 1 day->Daily.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground_truth_path = args.ground_truth_path or resolve_latest_ground_truth(
        args.ground_truth_root
    )
    use_frequency_aliases = not args.strict_frequency
    truth, metadata, excluded_without_ground_truth = load_ground_truth(
        ground_truth_path,
        use_frequency_aliases,
    )
    predictions, skipped_counts = load_predictions(
        args.predictions_path,
        set(truth),
        use_frequency_aliases,
    )
    result = evaluate(truth, predictions, metadata)
    result["config"] = {
        "ground_truth_path": str(ground_truth_path),
        "predictions_path": str(args.predictions_path),
        "use_frequency_aliases": use_frequency_aliases,
        "type_mapping": GT_TO_PRED_TYPE,
    }
    result["skipped_entity_counts"] = skipped_counts
    result["excluded_without_ground_truth"] = excluded_without_ground_truth
    result["summary"]["excluded_without_ground_truth_count"] = len(
        excluded_without_ground_truth
    )

    run_id = args.run_id or now_run_id()
    output_dir = args.output_root / run_id
    write_json(output_dir / "config.json", result["config"])
    write_json(output_dir / "summary.json", result["summary"])
    write_json(output_dir / "metrics_by_type.json", result["metrics_by_type"])
    write_json(output_dir / "global_unique_by_type.json", result["global_unique_by_type"])
    write_json(output_dir / "per_paper_metrics.json", result["per_paper"])
    write_json(output_dir / "paper_entity_comparison.json", result["paper_comparisons"])
    write_json(output_dir / "details.json", result["details"])
    write_jsonl(output_dir / "details.jsonl", result["details"])
    write_json(output_dir / "skipped_entity_counts.json", skipped_counts)
    write_json(
        output_dir / "excluded_without_ground_truth.json",
        excluded_without_ground_truth,
    )
    write_json(output_dir / "result.json", result)

    print(f"Output directory: {output_dir}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
