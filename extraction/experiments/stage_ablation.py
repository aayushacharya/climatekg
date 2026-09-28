"""Reproduce the per-stage attrition and stage-ablation tables from the stored stage flags.

Every triplet in output/evaluated_papers/<date>/*.json carries the verdict of each stage,
whether or not an earlier stage removed it:
  ontology_conformed  - ontology conformance agent (O)
  evidence_validated  - evidence validator agent (E)
  valid               - validator agent (V)
  eval                - C_triplet judge (Gemini-2.5-flash, evaluation_agent.py)
so every combination of stages can be evaluated without new LLM calls.
"""
import argparse
import glob
import itertools
import json
import os

STAGES = [("O", "ontology_conformed"), ("E", "evidence_validated"), ("V", "valid")]
DEFAULT_DATES = ["2026-02-01", "2026-04-24"]


def load_triplets(eval_root: str, dates: list[str]) -> list[dict]:
    triplets = []
    for date in dates:
        for path in sorted(glob.glob(os.path.join(eval_root, date, "*.json"))):
            with open(path, "r") as f:
                triplets.extend(json.load(f).get("kg_extraction_triplets", []))
    return triplets


def correctness(kept: list[dict]) -> float:
    return 100 * sum(bool(t["eval"]) for t in kept) / len(kept) if kept else 0.0


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", default=os.path.join(here, "..", "output", "evaluated_papers"))
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    args = parser.parse_args()

    triplets = load_triplets(args.eval_root, args.dates)
    print(f"{len(triplets):,} candidate triplets from {', '.join(args.dates)}\n")

    print("Per-stage attrition (cumulative pipeline)")
    print(f"{'Stage':<34}{'Input':>9}{'Filtered':>10}{'Output':>9}{'C_triplet (%)':>15}")
    kept = triplets
    print(f"{'Triplet and evidence extractor':<34}{'-':>9}{'-':>10}{len(kept):>9,}{correctness(kept):>15.2f}")
    for name, flag in STAGES:
        survivors = [t for t in kept if t[flag]]
        print(f"{name + ' (' + flag + ')':<34}{len(kept):>9,}{len(kept) - len(survivors):>10,}"
              f"{len(survivors):>9,}{correctness(survivors):>15.2f}")
        kept = survivors

    print("\nStage ablation (every subset of O, E, V)")
    print(f"{'Stages applied':<18}{'Triplets kept':>15}{'C_triplet (%)':>15}")
    for size in range(len(STAGES) + 1):
        for subset in itertools.combinations(STAGES, size):
            survivors = [t for t in triplets if all(t[flag] for _, flag in subset)]
            label = "+".join(name for name, _ in subset) or "none (extractor)"
            print(f"{label:<18}{len(survivors):>15,}{correctness(survivors):>15.2f}")


if __name__ == "__main__":
    main()
