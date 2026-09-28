#!/usr/bin/env python3
"""Run ClimateIE-ontology triplet extraction on human-corpus PDFs with Gemini.

This is the Gemini/PDF counterpart to extract_climateie_text_deepseek.py. It
uses the ClimateIE ontology only, so the output can be judged against the same
relation/entity space as ClimateIE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXTRACTION_DIR.parent
if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))

from models import Kg_Extraction_Model  # noqa: E402


DEFAULT_GOLD_DIR = SCRIPT_DIR / "data" / "ClimateIE" / "human_corpus"
DEFAULT_PDF_DIR = SCRIPT_DIR / "data" / "climateie_human_pdfs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "climateie_pdf_gemini_extraction"
DEFAULT_MODEL = "gemini-2.5-pro"

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

CLIMATEIE_ENTITY_TYPES = [
    "project",
    "location",
    "model",
    "experiment",
    "platform",
    "instrument",
    "provider",
    "variable",
    "weather event",
    "natural hazard",
    "teleconnection",
    "ocean circulation",
]

CLIMATEIE_RELATION_TYPES = [
    "ComparedTo",
    "Outputs",
    "RunBy",
    "ProvidedBy",
    "ValidatedBy",
    "UsedIn",
    "MeasuredAt",
    "MountedOn",
    "TargetsLocation",
]

CLIMATEIE_KG_LINKING_PROMPT = """Goal:
Given the attached climate or Earth-system science paper PDF, identify entities of the specified ClimateIE entity types and determine relationships among the detected entities.

Entity Types and Definitions:
- project: A scientific program, field campaign, or project from which data were collected.
- location: A place on Earth, a location within Earth, a vertical location, or a location outside Earth.
- model: A computer simulation integrating physical, chemical, biological, or dynamical processes to represent or predict Earth's climate system.
- experiment: A structured simulation designed to test hypotheses, investigate climate processes, or assess forcings.
- platform: A system, theory, phenomenon, vehicle, satellite, station, or other carrier/context used for study or observation.
- instrument: A device used to measure, observe, or calculate.
- provider: An organization, academic institution, agency, company, or other data/model/study provider.
- variable: A quantity or characteristic that can be measured or observed in climate experiments.
- weather event: A meteorological occurrence that impacts Earth's atmosphere and surface over short timescales.
- natural hazard: A phenomenon with potential to cause significant harm to life, property, or the environment.
- teleconnection: A large-scale pattern of climate variability linking weather and climate phenomena across distant regions.
- ocean circulation: Large-scale movement of water masses in Earth's oceans.

Allowed entity types:
project, location, model, experiment, platform, instrument, provider, variable, weather event, natural hazard, teleconnection, ocean circulation

Relationship Types and Definitions:
- ComparedTo: The source entity is compared to the target entity.
- Outputs: A climate model, experiment, or project outputs data.
- RunBy: Experiments or scenarios are run by a climate model.
- ProvidedBy: A dataset, instrument, or model is created or managed by an organization.
- ValidatedBy: The accuracy or reliability of model simulations is confirmed by datasets or analyses.
- UsedIn: An entity such as a model, simulation tool, experiment, or instrument is utilized within a project.
- MeasuredAt: A variable or parameter is quantified or recorded at a geographic location.
- MountedOn: An instrument or measurement device is physically attached or installed on a platform.
- TargetsLocation: An experiment, project, model, weather event, natural hazard, teleconnection, or ocean circulation is designed to study, simulate, or focus on a specific geographic location.

Allowed relationship types:
ComparedTo, Outputs, RunBy, ProvidedBy, ValidatedBy, UsedIn, MeasuredAt, MountedOn, TargetsLocation

Extraction rules:
- Extract only relationships clearly supported by the PDF.
- Use only the allowed entity types and relationship types above.
- Prefer omission over weak inference.
- Keep entity strings close to the paper text.
- Resolve direct anaphora when the paper makes the referent clear.
- The paragraph field must be the shortest evidence passage from the paper that supports the triplet.
"""

EXTRACTION_RESPONSE_INSTRUCTIONS = """
Return only a JSON array. Do not include markdown fences or explanation.
Each array item must have exactly these keys:
- source_entity_instance
- source_entity_type
- relation
- target_entity_instance
- target_entity_type
- paragraph

If no valid triplets are present, return [].
"""


def now_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_environment() -> None:
    for path in (
        REPO_ROOT / ".env",
        EXTRACTION_DIR / ".env",
        SCRIPT_DIR / ".env",
        Path.cwd() / ".env",
    ):
        load_dotenv_file(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def iter_human_doc_ids(gold_dir: Path) -> list[str]:
    if not gold_dir.exists():
        raise FileNotFoundError(f"ClimateIE human gold directory not found: {gold_dir}")
    return sorted(path.stem for path in gold_dir.glob("*.json"))


def select_doc_ids(doc_ids: list[str], limit_docs: int | None, sample_seed: int) -> list[str]:
    if limit_docs is None or limit_docs >= len(doc_ids):
        return list(doc_ids)
    if limit_docs < 0:
        raise ValueError("--limit-docs must be non-negative")
    rng = random.Random(sample_seed)
    selected = set(rng.sample(doc_ids, limit_docs))
    return [doc_id for doc_id in doc_ids if doc_id in selected]


def filter_doc_ids(doc_ids: list[str], requested_doc_ids: list[str] | None) -> list[str]:
    if not requested_doc_ids:
        return doc_ids
    allowed = set(requested_doc_ids)
    return [doc_id for doc_id in doc_ids if doc_id in allowed]


def extract_json_text(raw_response: str) -> str:
    text = (raw_response or "").strip()
    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def parse_triplets(raw_response: str) -> tuple[list[dict[str, Any]], str | None]:
    json_text = extract_json_text(raw_response)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return [], f"json_decode_error: {exc}"
    if not isinstance(payload, list):
        return [], "response_json_is_not_list"

    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"item_{idx}_is_not_object")
            continue
        try:
            parsed.append(Kg_Extraction_Model.model_validate(item).model_dump())
        except Exception as exc:
            errors.append(f"item_{idx}_schema_error: {exc}")
    return parsed, "; ".join(errors) if errors else None


def build_prompt(doc_id: str) -> str:
    return (
        f"{CLIMATEIE_KG_LINKING_PROMPT}"
        f"{EXTRACTION_RESPONSE_INSTRUCTIONS}\n\n"
        f"Document id: {doc_id}\n"
        "Extract triplets from the attached PDF."
    )


def response_to_dict(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    for attr in ("model_dump", "to_json_dict"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return None


class GeminiPdfExtractor:
    def __init__(
        self,
        model: str,
        api_key_env: str,
        temperature: float,
        max_output_tokens: int,
        request_sleep_seconds: float,
    ) -> None:
        load_environment()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.request_sleep_seconds = request_sleep_seconds
        self.api_key = os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env}, GOOGLE_API_KEY, or GEMINI_API_KEY is required for Gemini extraction"
            )
        self._client: Any | None = None
        self._types: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def types(self) -> Any:
        if self._types is None:
            from google.genai import types

            self._types = types
        return self._types

    def extract(self, pdf_path: Path, prompt: str) -> tuple[str, dict[str, Any] | None]:
        pdf_bytes = pdf_path.read_bytes()
        types = self.types
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=config,
        )
        if self.request_sleep_seconds > 0:
            time.sleep(self.request_sleep_seconds)
        return (response.text or ""), response_to_dict(response)


def retry_delay_seconds(base_delay: float, attempt_index: int) -> float:
    if base_delay <= 0:
        return 0.0
    return base_delay * (2 ** attempt_index)


def iter_progress(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def run(args: argparse.Namespace) -> Path:
    gold_dir = args.gold_dir.expanduser().resolve()
    pdf_dir = args.pdf_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or now_run_id()
    run_dir = output_root / run_id
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"Run directory already exists; pass --force: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_doc_ids = iter_human_doc_ids(gold_dir)
    filtered_doc_ids = filter_doc_ids(all_doc_ids, args.doc_id)
    selected_doc_ids = select_doc_ids(filtered_doc_ids, args.limit_docs, args.sample_seed)
    config = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "gold_dir": str(gold_dir),
        "pdf_dir": str(pdf_dir),
        "output_dir": str(run_dir),
        "model": args.model,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "request_sleep_seconds": args.request_sleep_seconds,
        "max_retries": args.max_retries,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "doc_id_filter": args.doc_id,
        "limit_docs": args.limit_docs,
        "sample_seed": args.sample_seed,
        "ontology": {
            "entity_types": CLIMATEIE_ENTITY_TYPES,
            "relation_types": CLIMATEIE_RELATION_TYPES,
        },
        "selected_doc_ids": selected_doc_ids,
        "total_human_docs": len(all_doc_ids),
        "total_filtered_docs": len(filtered_doc_ids),
    }
    write_json(run_dir / "config.json", config)

    extractor = GeminiPdfExtractor(
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_sleep_seconds=args.request_sleep_seconds,
    )

    pdf_records_path = run_dir / "pdf_records.jsonl"
    prompts_path = run_dir / "prompts.jsonl"
    raw_responses_path = run_dir / "raw_responses.jsonl"
    triplets_path = run_dir / "triplets.jsonl"
    errors_path = run_dir / "errors.jsonl"

    stats = {
        "documents": len(selected_doc_ids),
        "documents_attempted": 0,
        "documents_succeeded": 0,
        "documents_failed": 0,
        "documents_parse_failed": 0,
        "api_call_attempts": 0,
        "triplets_extracted": 0,
    }

    with (
        pdf_records_path.open("w", encoding="utf-8") as pdf_handle,
        prompts_path.open("w", encoding="utf-8") as prompt_handle,
        raw_responses_path.open("w", encoding="utf-8") as raw_handle,
        triplets_path.open("w", encoding="utf-8") as triplet_handle,
        errors_path.open("w", encoding="utf-8") as error_handle,
    ):
        for doc_id in iter_progress(selected_doc_ids, desc="Extracting ClimateIE PDFs with Gemini"):
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                append_jsonl(
                    error_handle,
                    {"doc_id": doc_id, "error_type": "missing_pdf_file", "path": str(pdf_path)},
                )
                stats["documents_failed"] += 1
                continue

            prompt = build_prompt(doc_id)
            append_jsonl(
                pdf_handle,
                {
                    "doc_id": doc_id,
                    "pdf_path": str(pdf_path),
                    "pdf_bytes": pdf_path.stat().st_size,
                },
            )
            append_jsonl(prompt_handle, {"doc_id": doc_id, "prompt": prompt})
            stats["documents_attempted"] += 1

            doc_success = False
            last_parse_error: str | None = None
            for attempt_index in range(args.max_retries + 1):
                attempt_number = attempt_index + 1
                stats["api_call_attempts"] += 1
                try:
                    raw, response_payload = extractor.extract(pdf_path, prompt)
                    parsed, parse_error = parse_triplets(raw)
                    append_jsonl(
                        raw_handle,
                        {
                            "doc_id": doc_id,
                            "attempt": attempt_number,
                            "model": args.model,
                            "raw_response": raw,
                            "response_payload": response_payload,
                            "parse_error": parse_error,
                        },
                    )
                    if parse_error:
                        last_parse_error = parse_error
                        if attempt_index < args.max_retries:
                            delay = retry_delay_seconds(args.retry_sleep_seconds, attempt_index)
                            if delay > 0:
                                time.sleep(delay)
                            continue
                        stats["documents_parse_failed"] += 1
                        stats["documents_failed"] += 1
                        append_jsonl(
                            error_handle,
                            {
                                "doc_id": doc_id,
                                "attempt": attempt_number,
                                "error_type": "parse_error",
                                "error": parse_error,
                            },
                        )
                        break
                    for triplet_index, triplet in enumerate(parsed):
                        record = {
                            "record_id": f"{doc_id}:pdf:{triplet_index}",
                            "doc_id": doc_id,
                            "chunk_index": None,
                            "chunk_span": None,
                            "triplet_index": triplet_index,
                            **triplet,
                        }
                        append_jsonl(triplet_handle, record)
                    stats["documents_succeeded"] += 1
                    stats["triplets_extracted"] += len(parsed)
                    doc_success = True
                    break
                except Exception as exc:
                    if attempt_index < args.max_retries:
                        delay = retry_delay_seconds(args.retry_sleep_seconds, attempt_index)
                        append_jsonl(
                            error_handle,
                            {
                                "doc_id": doc_id,
                                "attempt": attempt_number,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "will_retry": True,
                                "retry_delay_seconds": delay,
                            },
                        )
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    stats["documents_failed"] += 1
                    append_jsonl(
                        error_handle,
                        {
                            "doc_id": doc_id,
                            "attempt": attempt_number,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "will_retry": False,
                        },
                    )
                    break

    write_json(run_dir / "metrics.json", stats)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ClimateIE-ontology KG extraction on ClimateIE human PDFs with Gemini."
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--doc-id", action="append", help="Restrict extraction to a document id. May be passed multiple times.")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run(args)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print("\nClimateIE PDF Gemini Extraction Summary")
    print(f"  Run dir: {run_dir}")
    print(f"  Documents: {metrics['documents']}")
    print(f"  Documents attempted: {metrics['documents_attempted']}")
    print(f"  Documents succeeded: {metrics['documents_succeeded']}")
    print(f"  Documents failed: {metrics['documents_failed']}")
    print(f"  Triplets extracted: {metrics['triplets_extracted']}")


if __name__ == "__main__":
    main()
