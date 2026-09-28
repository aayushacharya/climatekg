"""
Evidence validation pass for ontology-conformed paper JSON files.

Default input:
  output/ontology_conformed_papers/<date>/*.json

Default markdown:
  papers/parsed/<date>/*.md

Default output:
  output/evidence_validated_papers/<date>/*.json
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 80.0


def today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def latest_ontology_conformance_date() -> str:
    output_root = SCRIPT_DIR / "output" / "ontology_conformed_papers"
    if not output_root.exists():
        return today()

    date_dirs = sorted(path.name for path in output_root.iterdir() if path.is_dir())
    if not date_dirs:
        return today()
    return date_dirs[-1]


def default_input_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "ontology_conformed_papers" / date


def default_markdown_dir(date: str) -> Path:
    return SCRIPT_DIR / "papers" / "parsed" / date


def default_output_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "evidence_validated_papers" / date


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def is_figure_or_layout_description(paragraph: str) -> bool:
    stripped = paragraph.lstrip()
    return stripped.startswith("Figure") or stripped.startswith("Table")


def partial_ratio_score(paragraph: str, markdown_content: str) -> float:
    result = fuzz.partial_ratio_alignment(paragraph, markdown_content)
    if result is None:
        return 100.0
    return result.score


def validate_triplet_evidence(
    triplet: dict, markdown_content: str, threshold: float
) -> bool:
    paragraph = triplet.get("paragraph", "")
    if not isinstance(paragraph, str) or not paragraph:
        return False

    if partial_ratio_score(paragraph, markdown_content) < threshold:
        return is_figure_or_layout_description(paragraph)
    return True


def mark_all_triplets_valid(data: dict) -> int:
    triplets = data.get("kg_extraction_triplets", [])
    if not isinstance(triplets, list):
        return 0

    count = 0
    for triplet in triplets:
        if isinstance(triplet, dict):
            triplet["evidence_validated"] = True
            count += 1
    return count


def validate_paper_evidence(
    paper_data: dict,
    markdown_path: Path,
    threshold: float,
) -> tuple[int, int, bool]:
    triplets = paper_data.get("kg_extraction_triplets", [])
    if not isinstance(triplets, list):
        raise ValueError("kg_extraction_triplets must be a list")

    if not markdown_path.exists():
        triplet_count = mark_all_triplets_valid(paper_data)
        return triplet_count, 0, False

    markdown_content = markdown_path.read_text(encoding="utf-8")
    invalid_count = 0

    for triplet in triplets:
        if not isinstance(triplet, dict):
            raise ValueError("kg_extraction_triplets must contain objects")

        evidence_validated = validate_triplet_evidence(
            triplet=triplet,
            markdown_content=markdown_content,
            threshold=threshold,
        )
        triplet["evidence_validated"] = evidence_validated
        if not evidence_validated:
            invalid_count += 1

    return len(triplets), invalid_count, True


def validate_evidence(
    input_dir: Path,
    markdown_dir: Path,
    output_dir: Path,
    threshold: float,
) -> dict[str, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Ontology-conformed paper directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "processed": 0,
        "triplets": 0,
        "invalid_triplets": 0,
        "missing_markdown": 0,
        "failed": 0,
    }

    for json_path in tqdm(iter_json_files(input_dir), desc="Validating evidence"):
        try:
            paper_data = json.loads(json_path.read_text(encoding="utf-8"))
            markdown_path = markdown_dir / f"{json_path.stem}.md"
            triplets, invalid, had_markdown = validate_paper_evidence(
                paper_data=paper_data,
                markdown_path=markdown_path,
                threshold=threshold,
            )

            if not had_markdown:
                print(f"Markdown file not found for paper {json_path.stem}, will mark all as true.")
                stats["missing_markdown"] += 1

            output_path = output_dir / json_path.name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(paper_data, f, indent=4)
                f.write("\n")

            stats["processed"] += 1
            stats["triplets"] += triplets
            stats["invalid_triplets"] += invalid
        except Exception as exc:
            stats["failed"] += 1
            print(f"Failed: {json_path} - {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    default_date = latest_ontology_conformance_date()
    parser = argparse.ArgumentParser(
        description="Add evidence validation flags to ontology-conformed paper JSON files."
    )
    parser.add_argument(
        "--date",
        default=default_date,
        help=(
            "Date partition to process. Defaults to the latest "
            "output/ontology_conformed_papers/<date> directory."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing ontology-conformed paper JSON files.",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        help="Directory containing parsed paper markdown files. Defaults to papers/parsed/<date>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to output/evidence_validated_papers/<date>.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="RapidFuzz partial-ratio threshold below which evidence is marked invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or default_input_dir(args.date)
    markdown_dir = args.markdown_dir or default_markdown_dir(args.date)
    output_dir = args.output_dir or default_output_dir(args.date)

    stats = validate_evidence(
        input_dir=input_dir,
        markdown_dir=markdown_dir,
        output_dir=output_dir,
        threshold=args.threshold,
    )

    print("\nEvidence Validation Summary")
    print(f"  Input dir: {input_dir}")
    print(f"  Markdown dir: {markdown_dir}")
    print(f"  Output dir: {output_dir}")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if stats["triplets"] > 0:
        invalid_percent = (stats["invalid_triplets"] / stats["triplets"]) * 100
        print(f"  invalid_triplet_percentage: {invalid_percent:.2f}%")


if __name__ == "__main__":
    main()
