"""
Ontology conformance pass for extracted paper JSON files.

Reads files produced by extraction_agent.py and adds an
"ontology_conformed": true/false flag to each extracted triple.
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable

from consts import FILTERED_RELATIONSHIP_SOURCE_TO_TARGET


SCRIPT_DIR = Path(__file__).resolve().parent
ONTOLOGY_CONFORMANCE_KEY = "ontology_conformed"
LEGACY_ONTOLOGY_CONFORMANCE_KEY = "ontology conformed"

REALM_VALUES = {
    "aer",
    "aerosol",
    "atmos",
    "atmosphere",
    "atmoschem",
    "atmospheric chemistry",
    "land",
    "landice",
    "land ice",
    "ocean",
    "ocnbgchem",
    "ocean biogeochemistry",
    "seaice",
    "sea ice",
}

EXPERIMENT_VALUES = {
    "historical",
    "picontrol",
    "amip",
    "abrupt-4xco2",
    "1pctco2",
    "esm-hist",
    "esm-picontrol",
}

ACTIVITY_OR_PROJECT_VALUES = {
    "cmip",
    "cmip5",
    "cmip6",
    "cordex",
    "obs4mips",
    "input4mips",
    "scenariomip",
    "aerchemmip",
    "c4mip",
    "damip",
    "fafmip",
    "geomip",
    "ismip6",
    "ls3mip",
    "lumip",
    "omip",
    "pmip",
}

COMMON_VARIABLE_VALUES = {
    "tas",
    "tasmax",
    "tasmin",
    "pr",
    "prc",
    "ps",
    "psl",
    "uas",
    "vas",
    "ua",
    "va",
    "wap",
    "hus",
    "hurs",
    "huss",
    "rsds",
    "rsus",
    "rlus",
    "rlut",
    "tos",
    "siconc",
    "sic",
    "zg",
    "air_temperature",
    "precipitation",
    "sea surface temperature",
    "surface air temperature",
}

PHYSICAL_SCHEME_CUES = {
    "scheme",
    "parameterization",
    "parameterisation",
    "closure",
    "formulation",
    "algorithm",
    "representation",
    "physics package",
}

KNOWN_PHYSICAL_SCHEME_NAMES = {
    "arakawa-schubert",
    "betts-miller",
    "chikira",
    "ecmwf radiation",
    "ecrad",
    "edmf",
    "gregory-rowntree",
    "k-profile",
    "kpp",
    "kain-fritsch",
    "mellor-yamada",
    "monin-obukhov",
    "park",
    "ras",
    "tiedtke",
    "tokioka",
    "zhang-mcfarlane",
}

MEMBER_RE = re.compile(r"^r\d+i\d+p\d+f\d+$", re.IGNORECASE)
SSP_RE = re.compile(r"^ssp\d{3,4}$", re.IGNORECASE)
RESOLUTION_RE = re.compile(
    r"^\s*((nominal\s+)?resolution\s*:?\s*)?\d+(\.\d+)?\s*(km|kilometer|kilometre|degree|degrees|deg)\s*$",
    re.IGNORECASE,
)


def today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def latest_extraction_date() -> str:
    output_root = SCRIPT_DIR / "output" / "extracted_papers"
    if not output_root.exists():
        return today()

    date_dirs = sorted(path.name for path in output_root.iterdir() if path.is_dir())
    if not date_dirs:
        return today()
    return date_dirs[-1]


def default_input_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "extracted_papers" / date


def default_output_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "ontology_conformed_papers" / date


def normalize_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.strip("\"'`.,;:()[]{}")


def entity_instances_for_type(triplet: dict, entity_type: str) -> list[str]:
    instances = []
    if triplet.get("source_entity_type") == entity_type:
        instances.append(triplet.get("source_entity_instance", ""))
    if triplet.get("target_entity_type") == entity_type:
        instances.append(triplet.get("target_entity_instance", ""))
    return instances


def is_realm_label(value: str) -> bool:
    normalized = normalize_value(value).replace("_", " ")
    compact = normalized.replace(" ", "")
    return normalized in REALM_VALUES or compact in REALM_VALUES


def is_experiment_label(value: str) -> bool:
    normalized = normalize_value(value)
    return normalized in EXPERIMENT_VALUES or SSP_RE.match(normalized) is not None


def looks_like_member(value: str) -> bool:
    return MEMBER_RE.match(normalize_value(value)) is not None


def looks_like_resolution(value: str) -> bool:
    normalized = normalize_value(value)
    return RESOLUTION_RE.match(normalized) is not None


def looks_like_variable(value: str) -> bool:
    normalized = normalize_value(value)
    if normalized in COMMON_VARIABLE_VALUES:
        return True
    if "_" in normalized and any(
        token in normalized
        for token in (
            "temperature",
            "precipitation",
            "pressure",
            "humidity",
            "wind",
            "flux",
            "concentration",
        )
    ):
        return True
    return False


def physical_scheme_has_support(triplet: dict, value: str) -> bool:
    value_text = normalize_value(value)
    paragraph_text = normalize_value(triplet.get("paragraph", ""))
    combined_text = f"{value_text} {paragraph_text}"
    return any(cue in combined_text for cue in PHYSICAL_SCHEME_CUES) or any(
        name in combined_text for name in KNOWN_PHYSICAL_SCHEME_NAMES
    )


def physical_feature_boundary_valid(value: str) -> bool:
    return not (
        is_realm_label(value)
        or is_experiment_label(value)
        or looks_like_member(value)
        or looks_like_resolution(value)
        or looks_like_variable(value)
        or normalize_value(value) in ACTIVITY_OR_PROJECT_VALUES
    )


def semantic_boundaries_conform(triplet: dict) -> bool:
    for value in entity_instances_for_type(triplet, "SourceComponent"):
        if is_realm_label(value):
            return False

    for value in entity_instances_for_type(triplet, "PhysicalScheme"):
        if not physical_scheme_has_support(triplet, value):
            return False

    for value in entity_instances_for_type(triplet, "PhysicalFeature"):
        if not physical_feature_boundary_valid(value):
            return False

    return True


def triplet_conforms(triplet: dict, ontology: set[tuple[str, str, str]]) -> bool:
    source_type = triplet.get("source_entity_type")
    target_type = triplet.get("target_entity_type")
    relation = triplet.get("relation")
    return (source_type, target_type, relation) in ontology and semantic_boundaries_conform(triplet)


def annotate_triplet_conformance(
    data: dict, ontology: set[tuple[str, str, str]]
) -> tuple[int, int]:
    triplets = data.get("kg_extraction_triplets", [])
    if not isinstance(triplets, list):
        raise ValueError("kg_extraction_triplets must be a list")

    conformed = 0
    not_conformed = 0
    for triplet in triplets:
        if not isinstance(triplet, dict):
            raise ValueError("kg_extraction_triplets must contain objects")

        is_conformed = triplet_conforms(triplet, ontology)
        triplet[ONTOLOGY_CONFORMANCE_KEY] = is_conformed
        if is_conformed:
            conformed += 1
        else:
            not_conformed += 1

    return conformed, not_conformed


def remove_paper_conformance_flags(data: dict) -> None:
    data.pop(ONTOLOGY_CONFORMANCE_KEY, None)
    data.pop(LEGACY_ONTOLOGY_CONFORMANCE_KEY, None)


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def add_ontology_conformance(input_dir: Path, output_dir: Path) -> dict[str, int]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Extraction output directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ontology = set(FILTERED_RELATIONSHIP_SOURCE_TO_TARGET)
    stats = {
        "processed": 0,
        "triplets": 0,
        "conformed_triplets": 0,
        "not_conformed_triplets": 0,
        "failed": 0,
    }

    for json_path in iter_json_files(input_dir):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            remove_paper_conformance_flags(data)
            conformed, not_conformed = annotate_triplet_conformance(data, ontology)

            output_path = output_dir / json_path.name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.write("\n")

            stats["processed"] += 1
            stats["triplets"] += conformed + not_conformed
            stats["conformed_triplets"] += conformed
            stats["not_conformed_triplets"] += not_conformed
        except Exception as exc:
            stats["failed"] += 1
            print(f"Failed: {json_path} - {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    default_date = latest_extraction_date()
    parser = argparse.ArgumentParser(
        description="Add ontology conformance flags to extracted paper JSON files."
    )
    parser.add_argument(
        "--date",
        default=default_date,
        help="Extraction date partition to process. Defaults to the latest output/extracted_papers/<date> directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing extracted paper JSON files. Defaults to output/extracted_papers/<date>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for ontology-conformed paper JSON files. Defaults to output/ontology_conformed_papers/<date>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or default_input_dir(args.date)
    output_dir = args.output_dir or default_output_dir(args.date)
    stats = add_ontology_conformance(input_dir, output_dir)

    print("\nOntology Conformance Summary")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
