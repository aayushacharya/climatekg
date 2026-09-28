#!/usr/bin/env python3
"""Run the full Gemini ClimateIE human-corpus comparison.

This orchestrates:
- our ClimateIE-ontology PDF extraction with Gemini Pro
- Gemini Flash validation for our extracted triplets
- Gemini Flash validation for ClimateIE's existing extracted triplets
- stored comparison metrics recomputed from JSONL artifacts
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXTRACTION_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_climateie_triplet_judge as judge  # noqa: E402
import extract_climateie_pdf_gemini as extraction  # noqa: E402


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "climateie_full_gemini_analysis"
DEFAULT_GOLD_DIR = judge.DEFAULT_GOLD_DIR
DEFAULT_PDF_DIR = extraction.DEFAULT_PDF_DIR
DEFAULT_CLIMATEIE_PRED_DIR = judge.DEFAULT_PRED_DIR
DEFAULT_OURS_EXTRACTION_MODEL = "gemini-2.5-pro"
DEFAULT_VALIDATION_MODEL = "gemini-2.5-flash"
DEFAULT_SMOKE_DOC_ID = "3b075d631213761324ab653a43c330d6a3609402"


def now_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return judge.read_jsonl(path)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def selected_doc_ids(gold_dir: Path, requested_doc_ids: list[str] | None) -> list[str]:
    all_ids = sorted(path.stem for path in gold_dir.glob("*.json"))
    if not requested_doc_ids:
        return all_ids
    allowed = set(requested_doc_ids)
    return [doc_id for doc_id in all_ids if doc_id in allowed]


def climateie_relationship_count(pred_path: Path) -> int:
    data = read_json(pred_path)
    if not isinstance(data, dict):
        raise ValueError(f"ClimateIE prediction file is not a chunk dictionary: {pred_path}")
    total = 0
    for chunk in data.values():
        if isinstance(chunk, dict):
            relationships = chunk.get("relationships") or []
            if isinstance(relationships, list):
                total += len(relationships)
    return total


def structural_check(
    gold_dir: Path,
    pdf_dir: Path,
    climateie_pred_dir: Path,
    requested_doc_ids: list[str] | None,
) -> dict[str, Any]:
    docs = selected_doc_ids(gold_dir, requested_doc_ids)
    missing_pdf = [doc_id for doc_id in docs if not (pdf_dir / f"{doc_id}.pdf").exists()]
    missing_climateie_pred = [
        doc_id for doc_id in docs if not (climateie_pred_dir / f"{doc_id}.json").exists()
    ]
    relationship_counts: dict[str, int] = {}
    for doc_id in docs:
        pred_path = climateie_pred_dir / f"{doc_id}.json"
        if pred_path.exists():
            relationship_counts[doc_id] = climateie_relationship_count(pred_path)

    return {
        "doc_ids": docs,
        "doc_count": len(docs),
        "gold_json_count": len(list(gold_dir.glob("*.json"))),
        "pdf_count": sum((pdf_dir / f"{doc_id}.pdf").exists() for doc_id in docs),
        "climateie_prediction_file_count": sum(
            (climateie_pred_dir / f"{doc_id}.json").exists() for doc_id in docs
        ),
        "missing_pdf": missing_pdf,
        "missing_climateie_prediction": missing_climateie_pred,
        "climateie_relationship_counts_by_doc": relationship_counts,
        "climateie_relationship_count": sum(relationship_counts.values()),
        "complete": not missing_pdf and not missing_climateie_pred and bool(docs),
    }


def exact_match_metrics(
    records: list[dict[str, Any]],
    gold_dir: Path,
    doc_ids: list[str],
) -> dict[str, Any]:
    pred_counter = Counter(tuple(record["normalized_text_triple"]) for record in records)
    gold_counter = judge.load_gold_text_triples(gold_dir, set(doc_ids))
    correct = sum((pred_counter & gold_counter).values())
    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())
    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "doc_count": len(doc_ids),
        "predicted": pred_total,
        "gold": gold_total,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def relation_exact_metrics(
    records: list[dict[str, Any]],
    gold_dir: Path,
    doc_ids: list[str],
) -> dict[str, dict[str, Any]]:
    pred_counter = Counter(tuple(record["normalized_text_triple"]) for record in records)
    gold_counter = judge.load_gold_text_triples(gold_dir, set(doc_ids))
    pred_by_relation: Counter[str] = Counter()
    gold_by_relation: Counter[str] = Counter()
    correct_by_relation: Counter[str] = Counter()

    for (_, relation, _), count in pred_counter.items():
        pred_by_relation[str(relation)] += count
    for (_, relation, _), count in gold_counter.items():
        gold_by_relation[str(relation)] += count
    for (_, relation, _), count in (pred_counter & gold_counter).items():
        correct_by_relation[str(relation)] += count

    result: dict[str, dict[str, Any]] = {}
    for relation in sorted(set(pred_by_relation) | set(gold_by_relation)):
        predicted = pred_by_relation[relation]
        gold = gold_by_relation[relation]
        correct = correct_by_relation[relation]
        precision = correct / predicted if predicted else 0.0
        recall = correct / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[relation] = {
            "predicted": predicted,
            "gold": gold,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def label_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record.get("parsed_label", "parse_error") for record in records)
    total = len(records)
    valid = counts.get("valid", 0)
    invalid = counts.get("invalid", 0)
    parse_errors = counts.get("parse_error", 0)
    judged = valid + invalid
    return {
        "total_records": total,
        "judged_triplets": judged,
        "valid_triplets": valid,
        "invalid_triplets": invalid,
        "parse_errors": parse_errors,
        "valid_rate": valid / total if total else 0.0,
        "valid_rate_excluding_parse_errors": valid / judged if judged else 0.0,
        "label_counts": dict(sorted(counts.items())),
    }


def system_metrics(
    pred_records: list[dict[str, Any]],
    judge_records: list[dict[str, Any]],
    gold_dir: Path,
    doc_ids: list[str],
) -> dict[str, Any]:
    valid_records = [record for record in judge_records if record.get("parsed_label") == "valid"]
    return {
        "predicted_triplets": len(pred_records),
        **label_metrics(judge_records),
        "exact_match_all_predictions": exact_match_metrics(pred_records, gold_dir, doc_ids),
        "exact_match_judge_valid_predictions": exact_match_metrics(valid_records, gold_dir, doc_ids),
    }


def by_doc_metrics(
    pred_records: list[dict[str, Any]],
    judge_records: list[dict[str, Any]],
    gold_dir: Path,
    doc_ids: list[str],
) -> dict[str, Any]:
    pred_by_doc: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    judge_by_doc: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pred_records:
        pred_by_doc[str(record.get("doc_id") or "")].append(record)
    for record in judge_records:
        judge_by_doc[str(record.get("doc_id") or "")].append(record)

    result: dict[str, Any] = {}
    for doc_id in doc_ids:
        doc_pred = pred_by_doc[doc_id]
        doc_judge = judge_by_doc[doc_id]
        doc_valid = [record for record in doc_judge if record.get("parsed_label") == "valid"]
        result[doc_id] = {
            "predicted_triplets": len(doc_pred),
            **label_metrics(doc_judge),
            "exact_match_all_predictions": exact_match_metrics(doc_pred, gold_dir, [doc_id]),
            "exact_match_judge_valid_predictions": exact_match_metrics(doc_valid, gold_dir, [doc_id]),
        }
    return result


def by_relation_metrics(
    pred_records: list[dict[str, Any]],
    judge_records: list[dict[str, Any]],
    gold_dir: Path,
    doc_ids: list[str],
) -> dict[str, Any]:
    pred_by_relation: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    judge_by_relation: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pred_records:
        pred_by_relation[str(record.get("relation") or "")].append(record)
    for record in judge_records:
        judge_by_relation[str(record.get("relation") or "")].append(record)

    exact_all = relation_exact_metrics(pred_records, gold_dir, doc_ids)
    valid_records = [record for record in judge_records if record.get("parsed_label") == "valid"]
    exact_valid = relation_exact_metrics(valid_records, gold_dir, doc_ids)
    relations = sorted(set(pred_by_relation) | set(judge_by_relation) | set(exact_all) | set(exact_valid))

    result: dict[str, Any] = {}
    for relation in relations:
        result[relation] = {
            "predicted_triplets": len(pred_by_relation[relation]),
            **label_metrics(judge_by_relation[relation]),
            "exact_match_all_predictions": exact_all.get(
                relation,
                {"predicted": 0, "gold": 0, "correct": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            ),
            "exact_match_judge_valid_predictions": exact_valid.get(
                relation,
                {"predicted": 0, "gold": 0, "correct": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            ),
        }
    return result


def delta(value_a: float | int, value_b: float | int) -> float | int:
    return value_a - value_b


def comparison_delta(ours: dict[str, Any], climateie: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_triplets_delta_ours_minus_climateie": delta(
            ours["predicted_triplets"], climateie["predicted_triplets"]
        ),
        "valid_triplets_delta_ours_minus_climateie": delta(
            ours["valid_triplets"], climateie["valid_triplets"]
        ),
        "valid_rate_delta_ours_minus_climateie": delta(
            ours["valid_rate"], climateie["valid_rate"]
        ),
        "exact_all_f1_delta_ours_minus_climateie": delta(
            ours["exact_match_all_predictions"]["f1"],
            climateie["exact_match_all_predictions"]["f1"],
        ),
        "exact_valid_f1_delta_ours_minus_climateie": delta(
            ours["exact_match_judge_valid_predictions"]["f1"],
            climateie["exact_match_judge_valid_predictions"]["f1"],
        ),
    }


def relation_deltas(ours_by_relation: dict[str, Any], climateie_by_relation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relation in sorted(set(ours_by_relation) | set(climateie_by_relation)):
        ours = ours_by_relation.get(relation, {})
        climateie = climateie_by_relation.get(relation, {})
        result[relation] = comparison_delta(
            {
                "predicted_triplets": ours.get("predicted_triplets", 0),
                "valid_triplets": ours.get("valid_triplets", 0),
                "valid_rate": ours.get("valid_rate", 0.0),
                "exact_match_all_predictions": ours.get("exact_match_all_predictions", {"f1": 0.0}),
                "exact_match_judge_valid_predictions": ours.get(
                    "exact_match_judge_valid_predictions", {"f1": 0.0}
                ),
            },
            {
                "predicted_triplets": climateie.get("predicted_triplets", 0),
                "valid_triplets": climateie.get("valid_triplets", 0),
                "valid_rate": climateie.get("valid_rate", 0.0),
                "exact_match_all_predictions": climateie.get("exact_match_all_predictions", {"f1": 0.0}),
                "exact_match_judge_valid_predictions": climateie.get(
                    "exact_match_judge_valid_predictions", {"f1": 0.0}
                ),
            },
        )
    return result


def doc_deltas(ours_by_doc: dict[str, Any], climateie_by_doc: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for doc_id in sorted(set(ours_by_doc) | set(climateie_by_doc)):
        result[doc_id] = comparison_delta(ours_by_doc[doc_id], climateie_by_doc[doc_id])
    return result


def completion_status(
    run_dir: Path,
    structural: dict[str, Any],
    ours_extraction_records: list[dict[str, Any]],
    ours_pred_records: list[dict[str, Any]],
    ours_judge_records: list[dict[str, Any]],
    climateie_pred_records: list[dict[str, Any]],
    climateie_judge_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    ours_extraction_metrics_path = run_dir / "ours_extraction" / "metrics.json"
    ours_extraction_metrics = (
        read_json(ours_extraction_metrics_path) if ours_extraction_metrics_path.exists() else {}
    )

    if not structural.get("complete"):
        reasons.append("Structural check is incomplete.")
    if ours_extraction_metrics.get("documents_attempted") != structural.get("doc_count"):
        reasons.append("Our extraction did not attempt every selected document.")
    if ours_extraction_metrics.get("documents_failed", 0) != 0:
        reasons.append("Our extraction has failed documents.")
    if len(ours_extraction_records) != len(ours_pred_records):
        reasons.append("Our validation pred_triplets count does not match our extraction triplet count.")
    if len(ours_judge_records) != len(ours_pred_records):
        reasons.append("Our validation does not have one judge record per extracted triplet.")
    if len(climateie_pred_records) != structural.get("climateie_relationship_count"):
        reasons.append("ClimateIE loaded prediction count does not match structural count.")
    if len(climateie_judge_records) != structural.get("climateie_relationship_count"):
        reasons.append("ClimateIE validation does not have one judge record per predicted triplet.")
    if label_metrics(ours_judge_records)["parse_errors"] != 0:
        reasons.append("Our validation has parse errors.")
    if label_metrics(climateie_judge_records)["parse_errors"] != 0:
        reasons.append("ClimateIE validation has parse errors.")
    if count_lines(run_dir / "ours_validation" / "errors.jsonl") != 0:
        reasons.append("Our validation errors.jsonl is not empty.")
    if count_lines(run_dir / "climateie_validation" / "errors.jsonl") != 0:
        reasons.append("ClimateIE validation errors.jsonl is not empty.")

    return {
        "complete": not reasons,
        "reasons": reasons,
    }


def write_summary(run_dir: Path, metrics: dict[str, Any]) -> None:
    ours = metrics["systems"]["ours"]
    climateie = metrics["systems"]["climateie"]
    completion = metrics["completion"]
    lines = [
        "# ClimateIE Full Gemini Analysis",
        "",
        f"Run complete: `{completion['complete']}`",
        "",
        "| System | Predicted | Valid | Invalid | Parse Errors | Valid Rate | Exact F1 | Valid Exact F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Ours | {ours['predicted_triplets']} | {ours['valid_triplets']} | "
            f"{ours['invalid_triplets']} | {ours['parse_errors']} | {ours['valid_rate']:.4f} | "
            f"{ours['exact_match_all_predictions']['f1']:.4f} | "
            f"{ours['exact_match_judge_valid_predictions']['f1']:.4f} |"
        ),
        (
            f"| ClimateIE | {climateie['predicted_triplets']} | {climateie['valid_triplets']} | "
            f"{climateie['invalid_triplets']} | {climateie['parse_errors']} | "
            f"{climateie['valid_rate']:.4f} | {climateie['exact_match_all_predictions']['f1']:.4f} | "
            f"{climateie['exact_match_judge_valid_predictions']['f1']:.4f} |"
        ),
        "",
        "## Completion Notes",
    ]
    if completion["reasons"]:
        lines.extend(f"- {reason}" for reason in completion["reasons"])
    else:
        lines.append("- All expected artifacts and judge records are present.")
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_comparison(run_dir: Path, gold_dir: Path, structural: dict[str, Any] | None = None) -> dict[str, Any]:
    structural = structural or read_json(run_dir / "structural_check.json")
    doc_ids = structural["doc_ids"]

    ours_extraction_records = read_jsonl(run_dir / "ours_extraction" / "triplets.jsonl")
    ours_pred = read_jsonl(run_dir / "ours_validation" / "pred_triplets.jsonl")
    ours_judge = read_jsonl(run_dir / "ours_validation" / "judge_records.jsonl")
    climateie_pred = read_jsonl(run_dir / "climateie_validation" / "pred_triplets.jsonl")
    climateie_judge = read_jsonl(run_dir / "climateie_validation" / "judge_records.jsonl")

    ours_metrics = system_metrics(ours_pred, ours_judge, gold_dir, doc_ids)
    climateie_metrics = system_metrics(climateie_pred, climateie_judge, gold_dir, doc_ids)
    ours_by_doc = by_doc_metrics(ours_pred, ours_judge, gold_dir, doc_ids)
    climateie_by_doc = by_doc_metrics(climateie_pred, climateie_judge, gold_dir, doc_ids)
    ours_by_relation = by_relation_metrics(ours_pred, ours_judge, gold_dir, doc_ids)
    climateie_by_relation = by_relation_metrics(climateie_pred, climateie_judge, gold_dir, doc_ids)

    metrics = {
        "run_dir": str(run_dir),
        "structural_check": structural,
        "completion": completion_status(
            run_dir,
            structural,
            ours_extraction_records,
            ours_pred,
            ours_judge,
            climateie_pred,
            climateie_judge,
        ),
        "systems": {
            "ours": ours_metrics,
            "climateie": climateie_metrics,
        },
        "comparison": comparison_delta(ours_metrics, climateie_metrics),
    }
    by_doc = {
        doc_id: {
            "ours": ours_by_doc[doc_id],
            "climateie": climateie_by_doc[doc_id],
            "delta": doc_deltas(ours_by_doc, climateie_by_doc)[doc_id],
        }
        for doc_id in doc_ids
    }
    by_relation = {
        relation: {
            "ours": ours_by_relation.get(relation),
            "climateie": climateie_by_relation.get(relation),
            "delta": relation_deltas(ours_by_relation, climateie_by_relation)[relation],
        }
        for relation in sorted(set(ours_by_relation) | set(climateie_by_relation))
    }

    write_json(run_dir / "comparison_metrics.json", metrics)
    write_json(run_dir / "comparison_by_doc.json", by_doc)
    write_json(run_dir / "comparison_by_relation.json", by_relation)
    write_jsonl(
        run_dir / "ours_invalid_examples.jsonl",
        [record for record in ours_judge if record.get("parsed_label") == "invalid"],
    )
    write_jsonl(
        run_dir / "climateie_invalid_examples.jsonl",
        [record for record in climateie_judge if record.get("parsed_label") == "invalid"],
    )
    write_summary(run_dir, metrics)
    return metrics


def run_analysis(args: argparse.Namespace) -> Path:
    gold_dir = args.gold_dir.expanduser().resolve()
    pdf_dir = args.pdf_dir.expanduser().resolve()
    climateie_pred_dir = args.climateie_pred_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or now_run_id()
    run_dir = output_root / run_id
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"Run directory already exists; pass --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_doc_ids = args.doc_id
    if args.smoke:
        requested_doc_ids = requested_doc_ids or [DEFAULT_SMOKE_DOC_ID]

    config = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "gold_dir": str(gold_dir),
        "pdf_dir": str(pdf_dir),
        "climateie_pred_dir": str(climateie_pred_dir),
        "output_dir": str(run_dir),
        "doc_id_filter": requested_doc_ids,
        "smoke": args.smoke,
        "ours_extraction_model": args.ours_extraction_model,
        "ours_extraction_max_output_tokens": args.ours_extraction_max_output_tokens,
        "validation_model": args.validation_model,
        "validation_max_workers": args.validation_max_workers,
        "gemini_max_output_tokens": args.gemini_max_output_tokens,
        "max_retries": args.max_retries,
        "retry_sleep_seconds": args.retry_sleep_seconds,
    }
    write_json(run_dir / "config.json", config)

    structural = structural_check(gold_dir, pdf_dir, climateie_pred_dir, requested_doc_ids)
    write_json(run_dir / "structural_check.json", structural)
    if not structural["complete"]:
        raise RuntimeError(f"Structural check failed. See {run_dir / 'structural_check.json'}")
    if args.structural_only:
        return run_dir

    extraction.run(
        argparse.Namespace(
            gold_dir=gold_dir,
            pdf_dir=pdf_dir,
            output_root=run_dir,
            run_id="ours_extraction",
            doc_id=requested_doc_ids,
            limit_docs=None,
            sample_seed=args.sample_seed,
            model=args.ours_extraction_model,
            api_key_env=args.api_key_env,
            temperature=0.0,
            max_output_tokens=args.ours_extraction_max_output_tokens,
            request_sleep_seconds=args.request_sleep_seconds,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            force=args.force,
        )
    )

    judge.run_experiment(
        argparse.Namespace(
            gold_dir=gold_dir,
            pred_dir=climateie_pred_dir,
            source_system="ours_v2_text",
            input_triplets_jsonl=run_dir / "ours_extraction" / "triplets.jsonl",
            output_root=run_dir,
            run_id="ours_validation",
            doc_id=requested_doc_ids,
            limit=None,
            sample_seed=args.sample_seed,
            judge_provider="gemini",
            model=args.validation_model,
            api_key_env=args.api_key_env,
            gemini_max_output_tokens=args.gemini_max_output_tokens,
            ollama_base_url=judge.DEFAULT_OLLAMA_BASE_URL,
            timeout=args.timeout,
            max_workers=args.validation_max_workers,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            force=args.force,
            dry_run=False,
        )
    )

    judge.run_experiment(
        argparse.Namespace(
            gold_dir=gold_dir,
            pred_dir=climateie_pred_dir,
            source_system="climateie_predictions",
            input_triplets_jsonl=None,
            output_root=run_dir,
            run_id="climateie_validation",
            doc_id=requested_doc_ids,
            limit=None,
            sample_seed=args.sample_seed,
            judge_provider="gemini",
            model=args.validation_model,
            api_key_env=args.api_key_env,
            gemini_max_output_tokens=args.gemini_max_output_tokens,
            ollama_base_url=judge.DEFAULT_OLLAMA_BASE_URL,
            timeout=args.timeout,
            max_workers=args.validation_max_workers,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            force=args.force,
            dry_run=False,
        )
    )

    build_comparison(run_dir, gold_dir, structural)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Gemini comparison for ClimateIE human-corpus papers."
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--climateie-pred-dir", type=Path, default=DEFAULT_CLIMATEIE_PRED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--doc-id", action="append", help="Restrict to a doc id. May be repeated.")
    parser.add_argument("--smoke", action="store_true", help="Run only the default one-document smoke case.")
    parser.add_argument("--structural-only", action="store_true", help="Only write config and structural_check.json.")
    parser.add_argument("--recompute-from", type=Path, help="Recompute comparison metrics from an existing run dir.")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--ours-extraction-model", default=DEFAULT_OURS_EXTRACTION_MODEL)
    parser.add_argument("--ours-extraction-max-output-tokens", type=int, default=32768)
    parser.add_argument("--validation-model", default=DEFAULT_VALIDATION_MODEL)
    parser.add_argument("--validation-max-workers", type=int, default=4)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=512)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recompute_from is not None:
        run_dir = args.recompute_from.expanduser().resolve()
        metrics = build_comparison(run_dir, args.gold_dir.expanduser().resolve())
    else:
        run_dir = run_analysis(args)
        if args.structural_only:
            metrics = {"completion": {"complete": True, "reasons": []}}
        else:
            metrics = read_json(run_dir / "comparison_metrics.json")

    print("\nClimateIE Full Gemini Analysis")
    print(f"  Run dir: {run_dir}")
    if (run_dir / "structural_check.json").exists():
        structural = read_json(run_dir / "structural_check.json")
        print(f"  Documents: {structural['doc_count']}")
        print(f"  ClimateIE triplets: {structural['climateie_relationship_count']}")
    if not args.structural_only:
        ours = metrics["systems"]["ours"]
        climateie = metrics["systems"]["climateie"]
        print(f"  Ours valid: {ours['valid_triplets']}/{ours['judged_triplets']}")
        print(f"  ClimateIE valid: {climateie['valid_triplets']}/{climateie['judged_triplets']}")
        print(f"  Complete: {metrics['completion']['complete']}")


if __name__ == "__main__":
    main()
