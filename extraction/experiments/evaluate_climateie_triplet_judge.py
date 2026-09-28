#!/usr/bin/env python3
"""Evaluate ClimateIE predicted text triplets with an LLM evidence judge.

This experiment uses ClimateIE chunk spans as evidence context. ClimateIE
predictions do not provide relation-level evidence spans.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXTRACTION_DIR.parent

DEFAULT_GOLD_DIR = SCRIPT_DIR / "data" / "ClimateIE" / "human_corpus"
# ClimateIE baseline predictions (3-shot Llama-3.3-70B-Instruct run of the ClimateIE system).
DEFAULT_PRED_DIR = SCRIPT_DIR / "data" / "climateie_predictions" / "3_shot_Llama-3.3-70B-Instruct"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "climateie_triplet_judge"
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 512
EVIDENCE_SOURCE = "climateie_chunk_substring"
OURS_EVIDENCE_SOURCE = "ours_paragraph"

CHUNK_SPAN_RE = re.compile(r"^\((\d+)\s*,\s*(\d+)\)$")
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
LABEL_RE = re.compile(r"\b(valid|invalid)\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
OUTER_PUNCT_RE = re.compile(r"^[\s\"'`.,;:()\[\]{}<>]+|[\s\"'`.,;:()\[\]{}<>]+$")

RELATION_DEFINITIONS = {
    "ComparedTo": "The source entity is compared to the target entity.",
    "Outputs": "A climate model, experiment, or project outputs data.",
    "RunBy": "An experiment or scenario is run by a climate model.",
    "ProvidedBy": "A dataset, instrument, or model is created or managed by an organization.",
    "ValidatedBy": "The accuracy or reliability of model simulations is confirmed by datasets or analyses.",
    "UsedIn": "An entity such as a model, simulation tool, experiment, or instrument is utilized within a project.",
    "MeasuredAt": "A variable or parameter is quantified or recorded at a geographic location.",
    "MountedOn": "An instrument or measurement device is physically attached or installed on a platform.",
    "TargetsLocation": "An experiment, project, model, weather event, natural hazard, teleconnection, or ocean circulation is designed to study, simulate, or focus on a geographic location.",
    "INCLUDES_EXPERIMENT": "An ExperimentFamily encompasses or includes a specific Experiment.",
    "HAS_SUB_EXPERIMENT": "An Experiment has a SubExperiment that refines, narrows, or extends it.",
    "IS_OF_TYPE": "A Source has a SourceType, or a Natural_Hazard has a NaturalHazardType.",
    "APPLIES_TO_REALM": "A Source, SourceComponent, or Experiment is associated with a specific Earth-system Realm.",
    "FOCUSES_ON_REALM": "An Activity is designed to study or emphasize a particular Realm.",
    "PARTICIPATED_IN": "An Institute participated in a Project.",
    "GENERATED_BY_ACTIVITY": "A Source was produced as part of a specific Activity.",
    "PRODUCED_BY_INSTITUTE": "A Source was produced by, developed by, or is the responsibility of an Institute.",
    "BELONGS_TO_MIP_ERA": "A Source or ExperimentFamily belongs to a MIP era such as CMIP5 or CMIP6.",
    "USED_IN_EXPERIMENT": "A Source or RCM was used in, run for, or provides simulations for a specific Experiment or scenario.",
    "PRODUCES_VARIABLE": "A Source or SourceComponent outputs, simulates, or produces a Variable.",
    "HAS_SUBSEQUENT_VERSION": "A Source or SourceComponent has a direct newer version.",
    "INHERITED_FROM": "A Source inherits from, derives from, or is based on another Source.",
    "HAS_SOURCE_COMPONENT": "A Source contains a SourceComponent, or a SourceComponent contains a subcomponent.",
    "USES_PHYSICAL_SCHEME": "A Source uses a PhysicalScheme, parameterization, closure, formulation, or algorithm.",
    "HAS_PHYSICAL_FEATURE": "A Source has a PhysicalFeature such as a grid, coordinate, dynamical core, resolution property, or configuration property.",
    "HAS_SIMULATION_TYPE": "A Source has a SimulationType such as coupled, atmosphere-only, control, transient, or initialized.",
    "HAS_METRIC": "A Source is associated with a quantitative Metric used to evaluate or diagnose it.",
    "PART_OF_PROJECT": "A Source, RCM, Experiment, or Activity belongs to a Project.",
    "ASSOCIATED_WITH_MEMBER": "A Source is associated with an ensemble Member.",
    "SAMPLED_AT_FREQUENCY": "A Source produces outputs at a Frequency.",
    "HAS_SPATIAL_RESOLUTION": "A Source, Variable, RCM, or Experiment has a spatial Resolution.",
    "PART_OF_ENSEMBLE": "A Source or SubExperiment belongs to an Ensemble.",
    "USES_FORCING": "An Experiment incorporates a specific Forcing.",
    "INCLUDES_ENSEMBLE_MEMBER": "An Experiment includes an Ensemble member.",
    "DRIVEN_BY_SOURCE": "An RCM is driven by a Source, typically a GCM.",
    "COVERS_DOMAIN": "An RCM or Project is defined over a Domain.",
    "MEASURES_VARIABLE": "An Instrument measures, observes, or records a Variable.",
    "MOUNTED_ON": "An Instrument is installed on, carried by, or mounted on a Platform.",
    "PERFORMED_BY_INSTITUTE": "An Experiment was carried out by an Institute.",
    "COVERS_REGION": "An RCM covers or spans a geographic region.",
    "IMPACTED_REGION": "A Weather_Event impacted a region.",
    "INFLUENCES_REGION": "A Teleconnection influences a region.",
    "OBSERVED_IN_REGION": "An Ocean_Circulation phenomenon is observed in a region.",
    "AFFECTED_REGION": "A Natural_Hazard affected a region.",
    "IN_CONTINENT": "A Country is located in a Continent.",
    "IN_COUNTRY": "A Country_Subdivision or City is located in a Country.",
    "HAS_CLIMATE_ZONE": "A No_Country_Region has a ClimateZone classification.",
}

JUDGE_PROMPT = """You are given a paragraph related to climate modeling or Earth system science, and a (source instance, relation, target instance) triplet extracted from it. Your task is to verify if the triplet supports the paragraph from the paper. Assume anaphoras are there in the paragraph. Use your own knowledge to insert the source or target to the context of the paragraph. Answer with 'valid' or 'invalid'. Do not provide an explanation.
Extracted Triplet:
Source: {source}
Relation: {relation}
Target: {target}
Paragraph context:
{evidence}
"""


def now_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


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


def append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = OUTER_PUNCT_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    return text.casefold().strip()


def normalize_relation(value: Any) -> str:
    relation = "" if value is None else str(value).strip()
    relation = relation.rstrip(">").strip()
    relation_lookup = {key.casefold(): key for key in RELATION_DEFINITIONS}
    return relation_lookup.get(relation.casefold(), relation)


def relation_definition(relation: str) -> str:
    return RELATION_DEFINITIONS.get(relation, "A relation asserted between the source and target entities.")


def canonical_text_triple(source: str, relation: str, target: str) -> tuple[str, str, str]:
    relation = normalize_relation(relation)
    left = normalize_text(source)
    right = normalize_text(target)
    if relation == "ComparedTo" and right < left:
        left, right = right, left
    return left, relation, right


def parse_chunk_span(value: str) -> tuple[int, int]:
    match = CHUNK_SPAN_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid ClimateIE chunk span: {value!r}")
    start = int(match.group(1))
    end = int(match.group(2))
    if start < 0 or end < start:
        raise ValueError(f"Invalid ClimateIE chunk span bounds: {value!r}")
    return start, end


def entity_label_lookup(entities: Iterable[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entity in entities:
        name = normalize_text(entity.get("name"))
        if name and name not in lookup:
            lookup[name] = str(entity.get("label") or "")
    return lookup


def load_gold_doc(gold_dir: Path, doc_id: str) -> dict[str, Any]:
    gold_path = gold_dir / f"{doc_id}.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"Missing gold ClimateIE document: {gold_path}")
    data = read_json(gold_path)
    if not isinstance(data.get("text"), str):
        raise ValueError(f"Gold ClimateIE document has no text field: {gold_path}")
    return data


def iter_prediction_files(pred_dir: Path, gold_dir: Path) -> list[Path]:
    if not pred_dir.exists():
        raise FileNotFoundError(f"ClimateIE prediction directory not found: {pred_dir}")
    if not gold_dir.exists():
        raise FileNotFoundError(f"ClimateIE gold directory not found: {gold_dir}")

    gold_doc_ids = {path.stem for path in gold_dir.glob("*.json")}
    return sorted(path for path in pred_dir.glob("*.json") if path.stem in gold_doc_ids)


def load_predicted_triplets(gold_dir: Path, pred_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    global_index = 0

    for pred_path in iter_prediction_files(pred_dir, gold_dir):
        doc_id = pred_path.stem
        gold_doc = load_gold_doc(gold_dir, doc_id)
        text = gold_doc["text"]
        pred_doc = read_json(pred_path)
        if not isinstance(pred_doc, dict):
            raise ValueError(f"ClimateIE prediction file is not a chunk dictionary: {pred_path}")

        for chunk_key in sorted(pred_doc.keys(), key=lambda value: parse_chunk_span(value)):
            chunk = pred_doc[chunk_key]
            if not isinstance(chunk, dict):
                raise ValueError(f"ClimateIE chunk is not an object: {pred_path}:{chunk_key}")
            start, end = parse_chunk_span(chunk_key)
            if end > len(text):
                raise ValueError(
                    f"Chunk span {chunk_key} exceeds gold text length {len(text)} for {doc_id}"
                )
            evidence = text[start:end]
            labels = entity_label_lookup(chunk.get("entities") or [])
            relationships = chunk.get("relationships") or []
            if not isinstance(relationships, list):
                raise ValueError(f"ClimateIE relationships is not a list: {pred_path}:{chunk_key}")

            for relation_index, relation_record in enumerate(relationships):
                if not isinstance(relation_record, dict):
                    raise ValueError(
                        f"ClimateIE relationship is not an object: {pred_path}:{chunk_key}:{relation_index}"
                    )
                source = str(relation_record.get("source") or "").strip()
                target = str(relation_record.get("target") or "").strip()
                relation = normalize_relation(relation_record.get("relation"))
                source_type = labels.get(normalize_text(source), "")
                target_type = labels.get(normalize_text(target), "")
                records.append(
                    {
                        "record_id": f"{doc_id}:{start}-{end}:{relation_index}",
                        "global_index": global_index,
                        "doc_id": doc_id,
                        "chunk_key": chunk_key,
                        "chunk_span": [start, end],
                        "relation_index": relation_index,
                        "source_text": source,
                        "source_type": source_type,
                        "relation": relation,
                        "target_text": target,
                        "target_type": target_type,
                        "evidence_source": EVIDENCE_SOURCE,
                        "evidence_start": start,
                        "evidence_end": end,
                        "evidence_text": evidence,
                        "normalized_text_triple": list(canonical_text_triple(source, relation, target)),
                    }
                )
                global_index += 1

    return records


def load_ours_v2_text_triplets(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        source = str(item.get("source_entity_instance") or "").strip()
        target = str(item.get("target_entity_instance") or "").strip()
        relation = normalize_relation(item.get("relation"))
        paragraph = str(item.get("paragraph") or "").strip()
        doc_id = str(item.get("doc_id") or "")
        chunk_span = item.get("chunk_span") or [None, None]
        record_id = item.get("record_id") or f"{path.stem}:{line_number}"
        records.append(
            {
                "record_id": record_id,
                "global_index": len(records),
                "doc_id": doc_id,
                "chunk_key": None,
                "chunk_span": chunk_span,
                "relation_index": item.get("triplet_index"),
                "source_text": source,
                "source_type": str(item.get("source_entity_type") or ""),
                "relation": relation,
                "target_text": target,
                "target_type": str(item.get("target_entity_type") or ""),
                "evidence_source": OURS_EVIDENCE_SOURCE,
                "evidence_start": chunk_span[0] if isinstance(chunk_span, list) and chunk_span else None,
                "evidence_end": chunk_span[1] if isinstance(chunk_span, list) and len(chunk_span) > 1 else None,
                "evidence_text": paragraph,
                "normalized_text_triple": list(canonical_text_triple(source, relation, target)),
            }
        )
    return records


def select_subset(
    records: list[dict[str, Any]],
    limit: int | None,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    rng = random.Random(sample_seed)
    selected_indices = set(rng.sample(range(len(records)), limit))
    return [record for idx, record in enumerate(records) if idx in selected_indices]


def filter_doc_ids(records: list[dict[str, Any]], doc_ids: list[str] | None) -> list[dict[str, Any]]:
    if not doc_ids:
        return records
    allowed = set(doc_ids)
    return [record for record in records if str(record.get("doc_id") or "") in allowed]


def load_gold_text_triples(gold_dir: Path, doc_ids: set[str]) -> Counter[tuple[str, str, str]]:
    triples: Counter[tuple[str, str, str]] = Counter()
    for doc_id in sorted(doc_ids):
        gold_doc = load_gold_doc(gold_dir, doc_id)
        entities = gold_doc.get("entities") or {}
        relations = gold_doc.get("relations") or {}
        if not isinstance(entities, dict) or not isinstance(relations, dict):
            raise ValueError(f"Unexpected gold ClimateIE schema for {doc_id}")

        entity_by_id = {str(entity_id): entity for entity_id, entity in entities.items()}
        for relation in relations.values():
            if not isinstance(relation, dict):
                continue
            source_entity = entity_by_id.get(str(relation.get("source")))
            target_entity = entity_by_id.get(str(relation.get("target")))
            if source_entity is None or target_entity is None:
                continue
            source = source_entity.get("substring")
            target = target_entity.get("substring")
            label = relation.get("label")
            triples[canonical_text_triple(source, label, target)] += 1
    return triples


def compute_exact_match_metrics(
    pred_records: list[dict[str, Any]],
    gold_dir: Path,
) -> dict[str, Any]:
    doc_ids = {str(record["doc_id"]) for record in pred_records}
    pred_counter = Counter(
        tuple(record["normalized_text_triple"]) for record in pred_records
    )
    gold_counter = load_gold_text_triples(gold_dir, doc_ids)
    correct = sum((pred_counter & gold_counter).values())
    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())
    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "scope": "selected_prediction_doc_ids",
        "doc_count": len(doc_ids),
        "predicted": pred_total,
        "gold": gold_total,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_prompt(record: dict[str, Any]) -> str:
    relation = str(record["relation"])
    return JUDGE_PROMPT.format(
        relation=relation,
        source=record["source_text"],
        target=record["target_text"],
        evidence=record["evidence_text"],
    )


def strip_deepseek_thinking(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def parse_judge_label(raw_response: str) -> str:
    stripped = strip_deepseek_thinking(raw_response)
    labels = LABEL_RE.findall(stripped)
    if not labels:
        return "parse_error"
    return labels[-1].lower()


def call_ollama_judge(
    record: dict[str, Any],
    model: str,
    base_url: str,
    timeout: float,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=timeout,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(record)}],
        temperature=0.0,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    parsed_label = parse_judge_label(content)
    return {
        **record,
        "judge_model": model,
        "judge_provider": "ollama_openai_compatible",
        "ollama_base_url": base_url,
        "raw_judge_response": content,
        "judge_response_without_thinking": strip_deepseek_thinking(content),
        "parsed_label": parsed_label,
        "parse_error": parsed_label == "parse_error",
    }


def call_gemini_judge(
    record: dict[str, Any],
    model: str,
    api_key_env: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    load_environment()
    api_key = os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"{api_key_env}, GOOGLE_API_KEY, or GEMINI_API_KEY is required")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(record),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        ),
    )
    content = response.text or ""
    parsed_label = parse_judge_label(content)
    payload = None
    for attr in ("model_dump", "to_json_dict"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                payload = method()
                break
            except Exception:
                pass
    return {
        **record,
        "judge_model": model,
        "judge_provider": "gemini",
        "raw_judge_response": content,
        "judge_response_without_thinking": content.strip(),
        "judge_response_payload": payload,
        "parsed_label": parsed_label,
        "parse_error": parsed_label == "parse_error",
    }


def dry_run_record(
    record: dict[str, Any],
    model: str,
    provider: str,
    base_url: str,
) -> dict[str, Any]:
    return {
        **record,
        "judge_model": model,
        "judge_provider": f"dry_run:{provider}",
        "ollama_base_url": base_url,
        "raw_judge_response": "",
        "judge_response_without_thinking": "",
        "parsed_label": "not_judged",
        "parse_error": False,
    }


def retry_delay_seconds(base_delay: float, attempt_index: int) -> float:
    if base_delay <= 0:
        return 0.0
    return base_delay * (2 ** attempt_index)


def call_judge_with_retries(
    record: dict[str, Any],
    judge_provider: str,
    model: str,
    api_key_env: str,
    gemini_max_output_tokens: int,
    ollama_base_url: str,
    timeout: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    last_parse_error_record: dict[str, Any] | None = None
    last_exception: Exception | None = None
    for attempt_index in range(max_retries + 1):
        attempt_number = attempt_index + 1
        try:
            if judge_provider == "gemini":
                judged = call_gemini_judge(record, model, api_key_env, gemini_max_output_tokens)
            else:
                judged = call_ollama_judge(record, model, ollama_base_url, timeout)
        except Exception as exc:
            last_exception = exc
            if attempt_index < max_retries:
                delay = retry_delay_seconds(retry_sleep_seconds, attempt_index)
                if delay > 0:
                    time.sleep(delay)
                continue
            raise

        judged["judge_attempts"] = attempt_number
        if not judged.get("parse_error"):
            return judged

        last_parse_error_record = judged
        if attempt_index < max_retries:
            delay = retry_delay_seconds(retry_sleep_seconds, attempt_index)
            if delay > 0:
                time.sleep(delay)

    if last_parse_error_record is None:
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Judge retry loop ended without a response")
    return last_parse_error_record


def iter_with_progress(futures, total: int, desc: str):
    iterator = concurrent.futures.as_completed(futures)
    if tqdm is None:
        return iterator
    return tqdm(iterator, total=total, desc=desc)


def write_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            append_jsonl(handle, record)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def compute_judge_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    label_counts = Counter(record.get("parsed_label", "parse_error") for record in records)
    total = len(records)
    valid = label_counts.get("valid", 0)
    invalid = label_counts.get("invalid", 0)
    parse_errors = label_counts.get("parse_error", 0)
    not_judged = label_counts.get("not_judged", 0)
    judged = valid + invalid
    metrics = {
        "total_records": total,
        "judged_triplets": judged,
        "valid_triplets": valid,
        "invalid_triplets": invalid,
        "parse_errors": parse_errors,
        "not_judged": not_judged,
        "valid_rate": valid / total if total else 0.0,
        "valid_rate_excluding_parse_errors": valid / judged if judged else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
    }

    by_relation: dict[str, dict[str, Any]] = {}
    relation_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        relation_records[str(record.get("relation") or "")].append(record)
    for relation, items in sorted(relation_records.items()):
        counts = Counter(item.get("parsed_label", "parse_error") for item in items)
        relation_total = len(items)
        relation_judged = counts.get("valid", 0) + counts.get("invalid", 0)
        by_relation[relation] = {
            "total_records": relation_total,
            "judged_triplets": relation_judged,
            "valid_triplets": counts.get("valid", 0),
            "invalid_triplets": counts.get("invalid", 0),
            "parse_errors": counts.get("parse_error", 0),
            "not_judged": counts.get("not_judged", 0),
            "valid_rate": counts.get("valid", 0) / relation_total if relation_total else 0.0,
            "valid_rate_excluding_parse_errors": (
                counts.get("valid", 0) / relation_judged if relation_judged else 0.0
            ),
            "label_counts": dict(sorted(counts.items())),
        }
    return metrics, by_relation


def recompute_metrics(run_dir: Path) -> None:
    judge_path = run_dir / "judge_records.jsonl"
    pred_path = run_dir / "pred_triplets.jsonl"
    config_path = run_dir / "config.json"
    if not judge_path.exists():
        raise FileNotFoundError(f"judge_records.jsonl not found in {run_dir}")
    if not pred_path.exists():
        raise FileNotFoundError(f"pred_triplets.jsonl not found in {run_dir}")
    config = read_json(config_path) if config_path.exists() else {}
    gold_dir = Path(config.get("gold_dir") or DEFAULT_GOLD_DIR)
    source_system = config.get("source_system") or "climateie_predictions"
    judge_records = read_jsonl(judge_path)
    pred_records = read_jsonl(pred_path)
    judge_metrics, by_relation = compute_judge_metrics(judge_records)
    if source_system in {"climateie_predictions", "ours_v2_text"}:
        exact_text_match = compute_exact_match_metrics(pred_records, gold_dir)
    else:
        exact_text_match = {
            "skipped": True,
            "reason": "Exact text match is skipped for this source system.",
        }
    metrics = {
        **judge_metrics,
        "exact_text_match": exact_text_match,
        "recomputed_from_records": True,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "metrics_by_relation.json", by_relation)


def run_experiment(args: argparse.Namespace) -> Path:
    gold_dir = args.gold_dir.expanduser().resolve()
    pred_dir = args.pred_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or now_run_id()
    run_dir = output_root / run_id
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"Run directory already exists; pass --force to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.source_system == "ours_v2_text":
        if args.input_triplets_jsonl is None:
            raise ValueError("--input-triplets-jsonl is required for --source-system ours_v2_text")
        all_records = load_ours_v2_text_triplets(args.input_triplets_jsonl.expanduser().resolve())
    else:
        all_records = load_predicted_triplets(gold_dir, pred_dir)
    all_records = filter_doc_ids(all_records, args.doc_id)
    selected_records = select_subset(all_records, args.limit, args.sample_seed)
    for sample_index, record in enumerate(selected_records):
        record["sample_index"] = sample_index

    config = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "gold_dir": str(gold_dir),
        "pred_dir": str(pred_dir),
        "output_dir": str(run_dir),
        "model": args.model,
        "judge_provider": args.judge_provider,
        "api_key_env": args.api_key_env,
        "gemini_max_output_tokens": args.gemini_max_output_tokens,
        "max_retries": args.max_retries,
        "retry_sleep_seconds": args.retry_sleep_seconds,
        "ollama_base_url": args.ollama_base_url,
        "timeout": args.timeout,
        "max_workers": args.max_workers,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "total_available_triplets": len(all_records),
        "doc_id_filter": args.doc_id,
        "selected_triplets": len(selected_records),
        "selection": "all" if args.limit is None else "deterministic_random_sample",
        "evidence_source": EVIDENCE_SOURCE,
        "dry_run": args.dry_run,
        "source_system": args.source_system,
        "input_triplets_jsonl": str(args.input_triplets_jsonl) if args.input_triplets_jsonl else None,
    }
    write_json(run_dir / "config.json", config)
    write_prediction_records(run_dir / "pred_triplets.jsonl", selected_records)

    errors_path = run_dir / "errors.jsonl"
    judge_records_path = run_dir / "judge_records.jsonl"
    with judge_records_path.open("w", encoding="utf-8") as judge_handle, errors_path.open(
        "w", encoding="utf-8"
    ) as error_handle:
        if args.dry_run:
            for record in selected_records:
                judged = dry_run_record(record, args.model, args.judge_provider, args.ollama_base_url)
                append_jsonl(judge_handle, judged)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_record = {
                    executor.submit(
                        call_judge_with_retries,
                        record,
                        args.judge_provider,
                        args.model,
                        args.api_key_env,
                        args.gemini_max_output_tokens,
                        args.ollama_base_url,
                        args.timeout,
                        args.max_retries,
                        args.retry_sleep_seconds,
                    ): record
                    for record in selected_records
                }
                for future in iter_with_progress(
                    future_to_record,
                    total=len(future_to_record),
                    desc="Judging ClimateIE triplets",
                ):
                    record = future_to_record[future]
                    try:
                        judged = future.result()
                        append_jsonl(judge_handle, judged)
                        if judged.get("parse_error"):
                            append_jsonl(
                                error_handle,
                                {
                                    "record_id": record.get("record_id"),
                                    "doc_id": record.get("doc_id"),
                                    "error_type": "parse_error",
                                    "raw_judge_response": judged.get("raw_judge_response"),
                                },
                            )
                    except Exception as exc:  # Keep the run resumable/inspectable.
                        append_jsonl(
                            error_handle,
                            {
                                "record_id": record.get("record_id"),
                                "doc_id": record.get("doc_id"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "record": record,
                            },
                        )

    recompute_metrics(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge ClimateIE predicted text triplets against evidence with Gemini or Ollama."
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument(
        "--source-system",
        choices=("climateie_predictions", "ours_v2_text"),
        default="climateie_predictions",
    )
    parser.add_argument(
        "--input-triplets-jsonl",
        type=Path,
        help="Triplets JSONL from extract_climateie_text_deepseek.py for --source-system ours_v2_text.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Optional explicit run id under --output-root.")
    parser.add_argument(
        "--doc-id",
        action="append",
        help="Restrict evaluation to a document id. May be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, help="Judge only N predicted triplets.")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--judge-provider", choices=("gemini", "ollama"), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--gemini-max-output-tokens", type=int, default=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files in an existing explicit --run-id directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create records and metrics without calling the LLM judge.",
    )
    parser.add_argument(
        "--recompute-metrics-from",
        type=Path,
        help="Recompute metrics from an existing run directory without model calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    if args.recompute_metrics_from is not None:
        recompute_metrics(args.recompute_metrics_from.expanduser().resolve())
        print(f"Recomputed metrics in {args.recompute_metrics_from}")
        return

    run_dir = run_experiment(args)
    metrics = read_json(run_dir / "metrics.json")
    print("\nClimateIE Triplet Judge Summary")
    print(f"  Run dir: {run_dir}")
    print(f"  Total records: {metrics['total_records']}")
    print(f"  Judged triplets: {metrics['judged_triplets']}")
    print(f"  Valid: {metrics['valid_triplets']}")
    print(f"  Invalid: {metrics['invalid_triplets']}")
    print(f"  Parse errors: {metrics['parse_errors']}")
    print(f"  Valid rate: {metrics['valid_rate']:.4f}")
    exact = metrics["exact_text_match"]
    if exact.get("skipped"):
        print("  Exact text F1: skipped")
    else:
        print(f"  Exact text F1: {exact['f1']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
