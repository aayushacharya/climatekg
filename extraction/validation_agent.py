"""
LLM validation pass for evidence-validated paper JSON files.

Default input:
  output/evidence_validated_papers/<date>/*.json

Default output:
  output/validated_papers/<date>/*.json
"""

import argparse
import concurrent.futures
import datetime as dt
import json
from functools import partial
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from prompts_deprecated import VALIDATION_PROMPT
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_GPT_MODEL = "gpt-5-mini-2025-08-07"


def load_environment() -> None:
    load_dotenv()
    load_dotenv(SCRIPT_DIR / ".env")


def today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def latest_evidence_validation_date() -> str:
    output_root = SCRIPT_DIR / "output" / "evidence_validated_papers"
    if not output_root.exists():
        return today()

    date_dirs = sorted(path.name for path in output_root.iterdir() if path.is_dir())
    if not date_dirs:
        return today()
    return date_dirs[-1]


def default_input_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "evidence_validated_papers" / date


def default_output_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "validated_papers" / date


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def triplet_value(triplet: dict, key: str, fallback_key: str | None = None) -> str:
    value = triplet.get(key)
    if value is None and fallback_key is not None:
        value = triplet.get(fallback_key)
    return "" if value is None else str(value)


def resolve_model(model_arg: str) -> tuple[str, str]:
    if model_arg.lower() == "gpt":
        return "gpt", DEFAULT_GPT_MODEL
    return "gemini", model_arg


def create_client(provider: str):
    if provider == "gpt":
        from openai import OpenAI

        return OpenAI()
    from google import genai

    return genai.Client()


def generate_label(client, provider: str, model: str, prompt: str) -> str:
    if provider == "gpt":
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text.strip()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()


def validate_triplet(client, provider: str, model: str, triplet: dict) -> str:
    prompt = VALIDATION_PROMPT.format(
        source=triplet_value(
            triplet,
            "source_entity_instance_anaphora_resolved",
            "source_entity_instance",
        ),
        source_type=triplet_value(triplet, "source_entity_type"),
        relation=triplet_value(triplet, "relation"),
        target=triplet_value(
            triplet,
            "target_entity_instance_anaphora_resolved",
            "target_entity_instance",
        ),
        target_type=triplet_value(triplet, "target_entity_type"),
        paragraph=triplet_value(triplet, "paragraph"),
    )

    return generate_label(client, provider, model, prompt)


def process_file(
    json_path: Path,
    output_dir: Path,
    provider: str,
    model: str,
) -> tuple[int, int]:
    client = create_client(provider)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    triplets = data.get("kg_extraction_triplets", [])
    if not isinstance(triplets, list):
        raise ValueError("kg_extraction_triplets must be a list")

    valid_count = 0
    for triplet in triplets:
        if not isinstance(triplet, dict):
            raise ValueError("kg_extraction_triplets must contain objects")

        validation_result = validate_triplet(client, provider, model, triplet)
        is_valid = validation_result.lower() == "valid"
        triplet["valid"] = is_valid
        if is_valid:
            valid_count += 1

    output_path = output_dir / json_path.name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    return len(triplets), valid_count


def validate_papers(
    input_dir: Path,
    output_dir: Path,
    provider: str,
    model: str,
    max_workers: int,
    force: bool,
) -> dict[str, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Evidence-validated paper directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_json_paths = list(iter_json_files(input_dir))
    json_paths = [
        path
        for path in all_json_paths
        if force or not (output_dir / path.name).exists()
    ]

    stats = {
        "processed": 0,
        "skipped_existing": len(all_json_paths) - len(json_paths),
        "triplets": 0,
        "valid_triplets": 0,
        "invalid_triplets": 0,
        "failed": 0,
    }

    worker = partial(
        process_file,
        output_dir=output_dir,
        provider=provider,
        model=model,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, json_path): json_path for json_path in json_paths}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Validating triplets",
        ):
            json_path = futures[future]
            try:
                triplet_count, valid_count = future.result()
                stats["processed"] += 1
                stats["triplets"] += triplet_count
                stats["valid_triplets"] += valid_count
                stats["invalid_triplets"] += triplet_count - valid_count
            except Exception as exc:
                stats["failed"] += 1
                print(f"Failed: {json_path} - {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    default_date = latest_evidence_validation_date()
    parser = argparse.ArgumentParser(
        description="Validate extracted triplets in evidence-validated paper JSON files."
    )
    parser.add_argument(
        "--date",
        default=default_date,
        help=(
            "Date partition to process. Defaults to the latest "
            "output/evidence_validated_papers/<date> directory."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing evidence-validated paper JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to output/validated_papers/<date>.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model selector. Defaults to gemini-2.5-flash. "
            "Pass 'gpt' to use gpt-5-mini-2025-08-07."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of concurrent files to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files that already exist in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    input_dir = args.input_dir or default_input_dir(args.date)
    output_dir = args.output_dir or default_output_dir(args.date)
    provider, resolved_model = resolve_model(args.model)

    stats = validate_papers(
        input_dir=input_dir,
        output_dir=output_dir,
        provider=provider,
        model=resolved_model,
        max_workers=args.max_workers,
        force=args.force,
    )

    print("\nValidation Summary")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Provider: {provider}")
    print(f"  Model: {resolved_model}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
