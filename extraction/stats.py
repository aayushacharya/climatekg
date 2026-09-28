#!/usr/bin/env python3
"""Summarize cumulative v2 extraction pipeline attrition and correctness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR /  "output"

STAGE_DIRS = {
    "extraction": "extracted_papers",
    "ontology": "ontology_conformed_papers",
    "evidence": "evidence_validated_papers",
    "validation": "validated_papers",
    "evaluation": "evaluated_papers",
}


@dataclass
class FilterStats:
    files: int = 0
    triplets: int = 0
    passed: int = 0
    filtered: int = 0
    missing_flag: int = 0


@dataclass
class EvalStats:
    triplets: int = 0
    eval_true: int = 0
    eval_false: int = 0
    missing_eval: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report v2 cumulative pipeline attrition stats and evaluator true rates. "
            "If --date is omitted, all dates are combined."
        )
    )
    parser.add_argument(
        "--date",
        help="Optional date partition such as 2026-02-01. Defaults to all dates.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_DIR,
        help="Root output directory. Defaults to v2/extraction/output.",
    )
    return parser.parse_args()


def json_files(stage_dir: Path, date: str) -> list[Path]:
    date_dir = stage_dir / date
    if not date_dir.exists():
        return []
    return sorted(path for path in date_dir.glob("*.json") if path.is_file())


def available_dates(output_root: Path) -> list[str]:
    dates: set[str] = set()
    for stage_name in STAGE_DIRS.values():
        stage_dir = output_root / stage_name
        if not stage_dir.exists():
            continue
        dates.update(path.name for path in stage_dir.iterdir() if path.is_dir())
    return sorted(dates)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def triplets_from_file(path: Path) -> list[dict]:
    data = load_json(path)
    triplets = data.get("kg_extraction_triplets", [])
    if not isinstance(triplets, list):
        return []
    return [triplet for triplet in triplets if isinstance(triplet, dict)]


def count_generated(stage_dir: Path, date: str) -> FilterStats:
    stats = FilterStats()
    for path in json_files(stage_dir, date):
        stats.files += 1
        stats.triplets += len(triplets_from_file(path))
    stats.passed = stats.triplets
    return stats


def flag_is_true(triplet: dict, flag: str) -> bool:
    return bool(triplet.get(flag))


def all_flags_true(*flags: str) -> Callable[[dict], bool]:
    return lambda triplet: all(flag_is_true(triplet, flag) for flag in flags)


def count_cumulative_stage(
    stage_dir: Path,
    date: str,
    input_predicate: Callable[[dict], bool],
    output_predicate: Callable[[dict], bool],
    current_flag: str | None,
) -> FilterStats:
    stats = FilterStats()
    for path in json_files(stage_dir, date):
        stats.files += 1
        for triplet in triplets_from_file(path):
            if not input_predicate(triplet):
                continue

            stats.triplets += 1
            if current_flag is not None and current_flag not in triplet:
                stats.missing_flag += 1
            elif output_predicate(triplet):
                stats.passed += 1
            else:
                stats.filtered += 1
    return stats


def count_eval_subset(
    stage_dir: Path,
    date: str,
    predicate: Callable[[dict], bool],
) -> EvalStats:
    stats = EvalStats()
    for path in json_files(stage_dir, date):
        for triplet in triplets_from_file(path):
            if not predicate(triplet):
                continue
            stats.triplets += 1
            if "eval" not in triplet:
                stats.missing_eval += 1
            elif bool(triplet.get("eval")):
                stats.eval_true += 1
            else:
                stats.eval_false += 1
    return stats


def merge_filter_stats(stats_list: list[FilterStats]) -> FilterStats:
    merged = FilterStats()
    for stats in stats_list:
        merged.files += stats.files
        merged.triplets += stats.triplets
        merged.passed += stats.passed
        merged.filtered += stats.filtered
        merged.missing_flag += stats.missing_flag
    return merged


def merge_eval_stats(stats_list: list[EvalStats]) -> EvalStats:
    merged = EvalStats()
    for stats in stats_list:
        merged.triplets += stats.triplets
        merged.eval_true += stats.eval_true
        merged.eval_false += stats.eval_false
        merged.missing_eval += stats.missing_eval
    return merged


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def filter_stats_for_date(output_root: Path, date: str) -> dict[str, FilterStats]:
    return {
        "Extraction": count_generated(output_root / STAGE_DIRS["extraction"], date),
        "Ontology Conformance": count_cumulative_stage(
            output_root / STAGE_DIRS["ontology"],
            date,
            lambda triplet: True,
            all_flags_true("ontology_conformed"),
            "ontology_conformed",
        ),
        "Evidence Validation": count_cumulative_stage(
            output_root / STAGE_DIRS["evidence"],
            date,
            all_flags_true("ontology_conformed"),
            all_flags_true("ontology_conformed", "evidence_validated"),
            "evidence_validated",
        ),
        "Validation": count_cumulative_stage(
            output_root / STAGE_DIRS["validation"],
            date,
            all_flags_true("ontology_conformed", "evidence_validated"),
            all_flags_true("ontology_conformed", "evidence_validated", "valid"),
            "valid",
        ),
        "Evaluation": count_cumulative_stage(
            output_root / STAGE_DIRS["evaluation"],
            date,
            all_flags_true("ontology_conformed", "evidence_validated", "valid"),
            all_flags_true("ontology_conformed", "evidence_validated", "valid", "eval"),
            "eval",
        ),
    }


def eval_stats_for_date(output_root: Path, date: str) -> dict[str, EvalStats]:
    evaluation_dir = output_root / STAGE_DIRS["evaluation"]
    return {
        "Extraction": count_eval_subset(evaluation_dir, date, lambda triplet: True),
        "Ontology Conformance": count_eval_subset(
            evaluation_dir,
            date,
            lambda triplet: bool(triplet.get("ontology_conformed")),
        ),
        "Evidence Validation": count_eval_subset(
            evaluation_dir,
            date,
            all_flags_true("ontology_conformed", "evidence_validated"),
        ),
        "Validation": count_eval_subset(
            evaluation_dir,
            date,
            all_flags_true("ontology_conformed", "evidence_validated", "valid"),
        ),
    }


def print_filter_table(stats_by_stage: dict[str, FilterStats]) -> None:
    print("\nCumulative pipeline attrition stats")
    print(
        f"{'Stage':<24} {'Files':>7} {'Input':>10} {'Passed':>10} "
        f"{'Filtered':>10} {'Pass Rate':>10} {'Missing Flag':>12}"
    )
    for stage, stats in stats_by_stage.items():
        print(
            f"{stage:<24} {stats.files:>7} {stats.triplets:>10} "
            f"{stats.passed:>10} {stats.filtered:>10} "
            f"{pct(stats.passed, stats.triplets):>10} {stats.missing_flag:>12}"
        )


def print_eval_table(stats_by_stage: dict[str, EvalStats]) -> None:
    print("\nEvaluation true rate by stage subset")
    print(
        f"{'Stage Subset':<24} {'Triplets':>10} {'Eval True':>10} "
        f"{'Eval False':>10} {'Eval True %':>12} {'Missing Eval':>12}"
    )
    for stage, stats in stats_by_stage.items():
        print(
            f"{stage:<24} {stats.triplets:>10} {stats.eval_true:>10} "
            f"{stats.eval_false:>10} {pct(stats.eval_true, stats.triplets):>12} "
            f"{stats.missing_eval:>12}"
        )


def combine_filter_stats(
    output_root: Path,
    dates: list[str],
) -> dict[str, FilterStats]:
    per_date = [filter_stats_for_date(output_root, date) for date in dates]
    stage_names = per_date[0].keys() if per_date else []
    return {
        stage: merge_filter_stats([date_stats[stage] for date_stats in per_date])
        for stage in stage_names
    }


def combine_eval_stats(
    output_root: Path,
    dates: list[str],
) -> dict[str, EvalStats]:
    per_date = [eval_stats_for_date(output_root, date) for date in dates]
    stage_names = per_date[0].keys() if per_date else []
    return {
        stage: merge_eval_stats([date_stats[stage] for date_stats in per_date])
        for stage in stage_names
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    dates = [args.date] if args.date else available_dates(output_root)
    if not dates:
        raise FileNotFoundError(f"No date partitions found under {output_root}")

    title = args.date if args.date else "all dates"
    print(f"Stats for {title}")
    if not args.date:
        print(f"Dates included: {', '.join(dates)}")

    print_filter_table(combine_filter_stats(output_root, dates))
    print_eval_table(combine_eval_stats(output_root, dates))


if __name__ == "__main__":
    main()
