#!/usr/bin/env python3
"""
Entity linking for v2 extraction outputs.

This combines the old entity_linking.py pass and the subsequent
embedding_unmatched_postprocessing.py LLM fallback into one corpus-level
script. By default it reads every date partition under output/evaluated_papers.
The output keeps the Neo4j loader-facing keys:
create_entities, create_relations, mentions, matching_remarks, and
matched_index_relations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Locks for the fine-grained parallel mode (--max-workers > 1). The paper loop is
# network(LLM)-bound with the GPU idle, so we overlap papers' Gemini fallback
# calls while serialising the short shared-state critical sections:
#   _EMB_LOCK   — the in-memory embedding index (numpy read/matmul + append/vstack)
#                 and the GPU embed model (single instance).
#   _STATE_LOCK — the cumulative saturation tracker + run-level output aggregation.
# LLM verify() calls sit OUTSIDE these locks, so they run concurrently.
_EMB_LOCK = threading.RLock()
_STATE_LOCK = threading.RLock()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from consts import (  # noqa: E402
    ALL_RELATIONSHIP_SOURCE_TO_TARGET,
    EMBEDDING_MATCH_ENTITIES,
    ENTITY_TYPES_TO_ADD_INSTANCES,
)
from prompts_deprecated import EMBEDDING_FALLBACK_LLM_PROMPT  # noqa: E402


SOURCE_COMPONENT_MATCH_CONFIRMATION_PROMPT = """Given a SourceComponent extracted from a climate-modeling paper and an existing SourceComponent candidate from the knowledge graph, decide whether they refer to the same concrete model component, subsystem, or component version.

Answer with only 'yes' or 'no'. Do not provide an explanation.

Extracted SourceComponent: {extracted_instance}
Existing SourceComponent candidate: {candidate_instance}
Embedding similarity score: {score}
Paragraph context: {paragraph}
"""

SOURCE_EMBEDDING_BELOW_THRESHOLD = "SOURCE_EMBEDDING_BELOW_THRESHOLD"
TARGET_EMBEDDING_BELOW_THRESHOLD = "TARGET_EMBEDDING_BELOW_THRESHOLD"
SOURCE_STRING_NO_MATCH = "SOURCE_STRING_NO_MATCH"
TARGET_STRING_NO_MATCH = "TARGET_STRING_NO_MATCH"
SOURCE_GRAPH_NO_MATCH = "SOURCE_GRAPH_NO_MATCH"
TARGET_GRAPH_NO_MATCH = "TARGET_GRAPH_NO_MATCH"
VALIDATION_REJECTED = "VALIDATION_REJECTED"
EVIDENCE_VALIDATION_REJECTED = "EVIDENCE_VALIDATION_REJECTED"
ONTOLOGY_CONFORMANCE_REJECTED = "ONTOLOGY_CONFORMANCE_REJECTED"
EVALUATION_REJECTED = "EVALUATION_REJECTED"

STRICT_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NO_MATCH = "no match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link evaluated v2 extraction triplets/entities to the KG and run "
            "LLM fallback for embedding misses."
        )
    )
    parser.add_argument(
        "--date",
        help="Optional date partition under output/evaluated_papers. Defaults to all dates.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing evaluated paper JSON files. Bypasses date discovery.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Final Neo4j-ready entity-linked JSON path.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Entity-linking stats JSON path.",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        help=(
            "Directory containing <EntityType>.json embedding seed files. "
            "Defaults to v2 embedding_files if present, then v2 output/embeddings, "
            "then info_extraction/output/embeddings."
        ),
    )
    parser.add_argument(
        "--variable-long-name-path",
        type=Path,
        help=(
            "Path to variable_name_to_long_name.json. Defaults to v2 output file "
            "if present, otherwise info_extraction/output/variable_name_to_long_name.json."
        ),
    )
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:8687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--embedding-model", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-threshold", type=float, default=0.8)
    parser.add_argument("--fallback-threshold", type=float, default=0.55)
    parser.add_argument("--fallback-top-k", type=int, default=10)
    parser.add_argument("--llm-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--skip-llm-fallback",
        action="store_true",
        help="Run only threshold-based entity linking and skip Gemini fallback verification.",
    )
    parser.add_argument(
        "--no-require-embedding-graph-check",
        action="store_true",
        help=(
            "Record the graph check for embedding matches but do not reject a match "
            "when the Neo4j node lookup misses."
        ),
    )
    parser.add_argument("--max-papers", type=int, help="Process at most this many unprocessed papers.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any existing output file and reprocess papers from scratch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write output and stats after every N processed papers.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Raise on the first paper-level processing error instead of recording and continuing.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Papers processed concurrently. The loop is Gemini-fallback (network) bound with the "
        "GPU idle, so concurrency overlaps those calls; shared state is lock-guarded. 1 = sequential.",
    )
    return parser.parse_args()


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for env_path in (
        SCRIPT_DIR / ".env",
        REPO_ROOT / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def date_partitions(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Date-partitioned input directory does not exist: {base_dir}")
    date_dirs = sorted(path.name for path in base_dir.iterdir() if path.is_dir())
    if not date_dirs:
        raise FileNotFoundError(f"No date partitions found under {base_dir}")
    return date_dirs


def collect_paper_paths(
    evaluated_base: Path,
    input_dir: Path | None,
    date: str | None,
) -> tuple[list[tuple[str, Path]], list[Path], list[str], list[dict[str, str]]]:
    if input_dir is not None:
        resolved_input_dir = input_dir.expanduser().resolve()
        input_dirs = [resolved_input_dir]
        dates = [resolved_input_dir.name]
    elif date is not None:
        input_dirs = [evaluated_base / date]
        dates = [date]
    else:
        dates = date_partitions(evaluated_base)
        input_dirs = [evaluated_base / each_date for each_date in dates]

    candidates: list[tuple[str, Path]] = []
    for source_date, each_input_dir in zip(dates, input_dirs):
        if not each_input_dir.exists():
            raise FileNotFoundError(f"Evaluated paper directory does not exist: {each_input_dir}")
        candidates.extend((source_date, path) for path in sorted(each_input_dir.glob("*.json")))

    selected_by_paper_id: dict[str, tuple[str, Path]] = {}
    duplicate_input_files: list[dict[str, str]] = []
    for source_date, paper_path in candidates:
        paper_id = paper_path.stem
        if paper_id in selected_by_paper_id:
            previous_date, previous_path = selected_by_paper_id[paper_id]
            duplicate_input_files.append(
                {
                    "paper_id": paper_id,
                    "dropped_date": previous_date,
                    "dropped_path": str(previous_path),
                    "kept_date": source_date,
                    "kept_path": str(paper_path),
                }
            )
        selected_by_paper_id[paper_id] = (source_date, paper_path)

    selected = sorted(
        selected_by_paper_id.values(),
        key=lambda item: (item[0], item[1].name),
    )
    return selected, input_dirs, dates, duplicate_input_files


def default_embedding_root() -> Path:
    v2_embedding_files = SCRIPT_DIR / "embedding_files"
    if v2_embedding_files.exists():
        return v2_embedding_files
    v2_embeddings = SCRIPT_DIR / "output" / "embeddings"
    if v2_embeddings.exists():
        return v2_embeddings
    return v2_embedding_files


def default_variable_long_name_path() -> Path:
    return SCRIPT_DIR / "output" / "variable_name_to_long_name.json"


def resolve_path(path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    return path.expanduser().resolve()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)


def read_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def all_entity_types() -> set[str]:
    entity_types: set[str] = set()
    for source_type, target_type, _ in ALL_RELATIONSHIP_SOURCE_TO_TARGET:
        entity_types.add(source_type)
        entity_types.add(target_type)
    return entity_types


def normalise_instance(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def triplet_instance(triplet: dict[str, Any], side: str) -> str:
    value = triplet.get(f"{side}_entity_instance")
    if value is None:
        value = triplet.get(f"{side}_entity_instance_anaphora_resolved")
    return normalise_instance(value)


def set_triplet_instance(triplet: dict[str, Any], side: str, value: str) -> None:
    triplet[f"{side}_entity_instance"] = value
    triplet.pop(f"{side}_entity_instance_anaphora_resolved", None)


def flag_is_accepted(triplet: dict[str, Any], key: str) -> bool:
    if key not in triplet:
        return True
    return bool(triplet.get(key))


def load_embedding_files(embedding_root: Path) -> dict[str, dict[str, Any]]:
    embedding_files: dict[str, dict[str, Any]] = {}
    for entity_type in EMBEDDING_MATCH_ENTITIES:
        embedding_path = embedding_root / f"{entity_type}.json"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file for {entity_type}: {embedding_path}")
        embedding_data = read_json(embedding_path)
        keys: list[str] = []
        embeddings: list[np.ndarray] = []
        for key, vector in embedding_data.items():
            keys.append(key)
            embeddings.append(np.asarray(vector, dtype=float))
        embedding_files[entity_type] = {
            "keys": keys,
            "embeddings": np.asarray(embeddings, dtype=float),
        }
    return embedding_files


def load_embedding_model(model_name: str) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
    )


def embedding_scores(embed_model: Any, entity_type: str, instance: str, embedding_files: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    # Held under _EMB_LOCK so the matmul reads a consistent index snapshot and the
    # shared GPU embed model is used by one thread at a time; keys is copied so a
    # concurrent add_embedding_seed can't desync it from the returned scores.
    with _EMB_LOCK:
        if entity_type not in embedding_files:
            return [], np.asarray([], dtype=float)
        keys = embedding_files[entity_type]["keys"]
        embeddings = embedding_files[entity_type]["embeddings"]
        if len(keys) == 0:
            return [], np.asarray([], dtype=float)
        query_embedding = np.asarray(embed_model.get_text_embedding(instance), dtype=float)
        scores = embeddings @ query_embedding
        return list(keys), scores


def embedding_match(
    embed_model: Any,
    entity_type: str,
    instance: str,
    embedding_files: dict[str, Any],
    threshold: float,
) -> tuple[str | None, float | None]:
    keys, scores = embedding_scores(embed_model, entity_type, instance, embedding_files)
    if len(keys) == 0:
        return None, None
    index = int(np.argmax(scores))
    score = float(scores[index])
    if score >= threshold:
        return keys[index], score
    return None, score


def embedding_match_with_tolerance(
    embed_model: Any,
    entity_type: str,
    instance: str,
    embedding_files: dict[str, Any],
    tolerance_threshold: float,
    top_k: int,
) -> list[tuple[str, float]]:
    keys, scores = embedding_scores(embed_model, entity_type, instance, embedding_files)
    if len(keys) == 0:
        return []
    indices = np.where(scores >= tolerance_threshold)[0]
    ranked_indices = sorted(indices, key=lambda i: float(scores[i]), reverse=True)[:top_k]
    return [(keys[int(index)], float(scores[int(index)])) for index in ranked_indices]


def add_embedding_seed(embedding_files: dict[str, Any], entity_type: str, instance: str, embedding: np.ndarray) -> None:
    with _EMB_LOCK:
        if entity_type not in embedding_files:
            embedding_files[entity_type] = {
                "keys": [instance],
                "embeddings": np.asarray([embedding], dtype=float),
            }
            return
        if instance in embedding_files[entity_type]["keys"]:
            return
        embedding_files[entity_type]["keys"].append(instance)
        embedding_files[entity_type]["embeddings"] = np.vstack(
            [embedding_files[entity_type]["embeddings"], embedding]
        )


def normalise_graph_name(value: str) -> str:
    return (
        value.replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace(" ", "")
    )


def entity_exists(session: Any, entity_type: str, entity_name: str) -> str | None:
    if not entity_type or not entity_name or not STRICT_LABEL_RE.match(entity_type):
        return None
    if entity_type in {"Source", "RCM"}:
        normalised_name = normalise_graph_name(entity_name)
        query = f"""
            MATCH (n:{entity_type})
            WHERE $name = REPLACE(REPLACE(REPLACE(REPLACE(n.name, "-", ""), "_", ""), ".", ""), " ", "")
            RETURN n.name AS name
        """
        result = session.run(query, name=normalised_name)
    else:
        query = f"""
            MATCH (n:{entity_type})
            WHERE n.name = $name
            RETURN n.name AS name
        """
        result = session.run(query, name=entity_name)
    output = result.single()
    return None if output is None else output["name"]


def add_unique(mapping: dict[str, list[str]], key: str, value: str) -> bool:
    if not value:
        return False
    values = set(mapping.get(key, []))
    before = len(values)
    values.add(value)
    mapping[key] = sorted(values)
    return len(values) > before


def add_create_entity(paper: dict[str, Any], stats: dict[str, Any], entity_type: str, instance: str) -> bool:
    added = add_unique(paper["create_entities"], entity_type, instance)
    if added:
        stats["created_entities_added"] += 1
        stats["created_entities_by_type"][entity_type] += 1
    return added


def add_mention(paper: dict[str, Any], stats: dict[str, Any], entity_type: str, instance: str) -> bool:
    added = add_unique(paper["mentions"], entity_type, instance)
    if added:
        stats["mentions_added"] += 1
        stats["mentions_by_type"][entity_type] += 1
    return added


def add_relation(
    paper: dict[str, Any],
    stats: dict[str, Any],
    triplet_index: int,
    relation_name: str,
    source_entity: str,
    source_entity_type: str,
    target_entity: str,
    target_entity_type: str,
    supporting_paragraph: str,
    source_strategy: str | None = None,
    target_strategy: str | None = None,
) -> bool:
    relation_triplets = paper["create_relations"].get(relation_name, [])
    entry = {
        "source_entity": source_entity,
        "source_entity_type": source_entity_type,
        "target_entity": target_entity,
        "target_entity_type": target_entity_type,
        "supporting_paragraph": supporting_paragraph,
        "triplet_index": triplet_index,
    }
    if source_strategy:
        entry["source_linking_strategy"] = source_strategy
    if target_strategy:
        entry["target_linking_strategy"] = target_strategy

    relation_key = (
        triplet_index,
        relation_name,
        source_entity_type,
        source_entity,
        target_entity_type,
        target_entity,
        supporting_paragraph,
    )
    existing_keys = {
        (
            item.get("triplet_index"),
            relation_name,
            item.get("source_entity_type"),
            item.get("source_entity"),
            item.get("target_entity_type"),
            item.get("target_entity"),
            item.get("supporting_paragraph"),
        )
        for item in relation_triplets
    }
    if relation_key in existing_keys:
        return False

    relation_triplets.append(entry)
    paper["create_relations"][relation_name] = relation_triplets
    if triplet_index not in paper["matched_index_relations"]:
        paper["matched_index_relations"].append(triplet_index)
        paper["matched_index_relations"].sort()
    stats["relations_created"] += 1
    stats["relations_by_type"][relation_name] += 1
    return True


def increment_nested(counter_dict: dict[str, Any], entity_type: str, key: str) -> None:
    by_type = counter_dict.setdefault(entity_type, defaultdict(int))
    by_type[key] += 1


def llm_confirmed_yes(output: str | None) -> bool:
    if output is None:
        return False
    choice = normalise_llm_choice(output).casefold()
    return choice in {"yes", "y", "true", "same", "match"} or choice.startswith("yes")


def verify_source_component_match(
    verifier: "GeminiFallbackVerifier | None",
    extracted_instance: str,
    candidate_instance: str,
    score: float | None,
    paragraph: str,
) -> dict[str, Any]:
    prompt = SOURCE_COMPONENT_MATCH_CONFIRMATION_PROMPT.format(
        extracted_instance=extracted_instance,
        candidate_instance=candidate_instance,
        score="" if score is None else f"{score:.6f}",
        paragraph=paragraph,
    )
    output = verifier.verify(prompt) if verifier is not None else None
    return {
        "required": True,
        "prompt": prompt,
        "llm_response": output,
        "confirmed_match": llm_confirmed_yes(output),
        "skipped_no_verifier": verifier is None,
    }


def consolidate_addable_entity(
    embed_model: Any,
    embedding_files: dict[str, Any],
    entity_type: str,
    instance: str,
    threshold: float,
    stats: dict[str, Any],
    verifier: "GeminiFallbackVerifier | None" = None,
    paragraph: str = "",
) -> tuple[str, dict[str, Any]]:
    stats["consolidation"]["calls"] += 1
    increment_nested(stats["consolidation"]["by_type"], entity_type, "calls")

    existing_match, score = embedding_match(embed_model, entity_type, instance, embedding_files, threshold)
    if existing_match is not None:
        if entity_type == "SourceComponent":
            confirmation = verify_source_component_match(
                verifier,
                instance,
                existing_match,
                score,
                paragraph,
            )
            stats["consolidation"]["source_component_llm_checks"] += 1
            if confirmation["confirmed_match"]:
                stats["consolidation"]["source_component_llm_confirmed"] += 1
                stats["consolidation"]["reused_existing"] += 1
                increment_nested(stats["consolidation"]["by_type"], entity_type, "reused_existing")
                return existing_match, {
                    "status": "reused_existing_llm_confirmed",
                    "matched_instance": existing_match,
                    "score": score,
                    "llm_confirmation": confirmation,
                }

            stats["consolidation"]["source_component_llm_rejected"] += 1
            with _EMB_LOCK:
                embedding = np.asarray(embed_model.get_text_embedding(instance), dtype=float)
            add_embedding_seed(embedding_files, entity_type, instance, embedding)
            stats["consolidation"]["created_new"] += 1
            increment_nested(stats["consolidation"]["by_type"], entity_type, "created_new")
            return instance, {
                "status": "created_new_llm_rejected_match",
                "matched_instance": existing_match,
                "score": score,
                "llm_confirmation": confirmation,
            }

        stats["consolidation"]["reused_existing"] += 1
        increment_nested(stats["consolidation"]["by_type"], entity_type, "reused_existing")
        return existing_match, {
            "status": "reused_existing",
            "matched_instance": existing_match,
            "score": score,
        }

    with _EMB_LOCK:
        embedding = np.asarray(embed_model.get_text_embedding(instance), dtype=float)
    add_embedding_seed(embedding_files, entity_type, instance, embedding)
    stats["consolidation"]["created_new"] += 1
    increment_nested(stats["consolidation"]["by_type"], entity_type, "created_new")
    return instance, {
        "status": "created_new",
        "matched_instance": None,
        "score": score,
    }


def graph_checked_embedding_match(
    session: Any,
    entity_type: str,
    matched_instance: str,
    require_graph_check: bool,
) -> tuple[str | None, dict[str, Any]]:
    graph_name = entity_exists(session, entity_type, matched_instance)
    check_data = {
        "checked": True,
        "graph_matched_instance": graph_name,
        "required": require_graph_check,
    }
    if graph_name is not None:
        return graph_name, check_data
    if require_graph_check:
        return None, check_data
    return matched_instance, check_data


def link_entity_for_triplet(
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    paper: dict[str, Any],
    stats: dict[str, Any],
    side: str,
    entity_type: str,
    entity_instance: str,
    threshold: float,
    require_embedding_graph_check: bool,
    verifier: GeminiFallbackVerifier | None,
    paragraph: str,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    side_upper = side.upper()
    decision = {
        "side": side,
        "entity_type": entity_type,
        "original_instance": entity_instance,
        "resolved_instance": entity_instance,
        "strict_scientific_convention": False,
        "strategy": None,
        "matched": True,
        "llm_fallback": False,
        "score": None,
        "graph_database_check": None,
        "consolidation": None,
    }

    if entity_type in ENTITY_TYPES_TO_ADD_INSTANCES:
        decision["strategy"] = "add_to_kg_with_consolidation"
        resolved_instance, consolidation = consolidate_addable_entity(
            embed_model,
            embedding_files,
            entity_type,
            entity_instance,
            threshold,
            stats,
            verifier,
            paragraph,
        )
        decision["resolved_instance"] = resolved_instance
        decision["consolidation"] = consolidation
        add_create_entity(paper, stats, entity_type, resolved_instance)
        return True, resolved_instance, None, decision

    if entity_type in EMBEDDING_MATCH_ENTITIES:
        decision["strategy"] = "embedding_similarity_with_graph_check"
        matched_instance, score = embedding_match(
            embed_model,
            entity_type,
            entity_instance,
            embedding_files,
            threshold,
        )
        decision["score"] = score
        if matched_instance is None:
            reason = f"{side_upper}_EMBEDDING_BELOW_THRESHOLD"
            decision["matched"] = False
            stats["embedding_below_threshold"] += 1
            stats["embedding_below_threshold_by_type"][entity_type] += 1
            return False, entity_instance, reason, decision

        resolved_instance, graph_check = graph_checked_embedding_match(
            session,
            entity_type,
            matched_instance,
            require_embedding_graph_check,
        )
        decision["graph_database_check"] = graph_check
        if resolved_instance is None:
            reason = f"{side_upper}_GRAPH_NO_MATCH"
            decision["matched"] = False
            stats["embedding_graph_no_match"] += 1
            stats["embedding_graph_no_match_by_type"][entity_type] += 1
            return False, entity_instance, reason, decision

        decision["resolved_instance"] = resolved_instance
        stats["embedding_matches"] += 1
        stats["embedding_matches_by_type"][entity_type] += 1
        return True, resolved_instance, None, decision

    decision["strict_scientific_convention"] = True
    decision["strategy"] = "string_matching"
    matched_db_name = entity_exists(session, entity_type, entity_instance)
    if matched_db_name is None:
        reason = f"{side_upper}_STRING_NO_MATCH"
        decision["matched"] = False
        stats["string_no_matches"] += 1
        stats["string_no_matches_by_type"][entity_type] += 1
        return False, entity_instance, reason, decision

    decision["resolved_instance"] = matched_db_name
    stats["string_matches"] += 1
    stats["string_matches_by_type"][entity_type] += 1
    return True, matched_db_name, None, decision


def link_entity_for_mention(
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    paper: dict[str, Any],
    stats: dict[str, Any],
    entity_type: str,
    entity_instance: str,
    threshold: float,
    require_embedding_graph_check: bool,
    verifier: GeminiFallbackVerifier | None,
) -> None:
    if entity_type in ENTITY_TYPES_TO_ADD_INSTANCES:
        resolved_instance, _ = consolidate_addable_entity(
            embed_model,
            embedding_files,
            entity_type,
            entity_instance,
            threshold,
            stats,
            verifier,
        )
        add_create_entity(paper, stats, entity_type, resolved_instance)
        return

    if entity_type in EMBEDDING_MATCH_ENTITIES:
        matched_instance, _ = embedding_match(
            embed_model,
            entity_type,
            entity_instance,
            embedding_files,
            threshold,
        )
        if matched_instance is None:
            return
        resolved_instance, _ = graph_checked_embedding_match(
            session,
            entity_type,
            matched_instance,
            require_embedding_graph_check,
        )
        if resolved_instance is not None:
            add_mention(paper, stats, entity_type, resolved_instance)
        return

    matched_db_name = entity_exists(session, entity_type, entity_instance)
    if matched_db_name is not None:
        add_mention(paper, stats, entity_type, matched_db_name)


def make_empty_paper_stats(paper_id: str, source_date: str | None = None) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "source_date": source_date,
        "triplets_total": 0,
        "triplets_considered": 0,
        "triplets_matched": 0,
        "triplets_unmatched": 0,
        "rejected_ontology_conformance": 0,
        "rejected_evidence_validation": 0,
        "rejected_validation": 0,
        "rejected_evaluation": 0,
        "relations_created": 0,
        "relations_by_type": Counter(),
        "created_entities_added": 0,
        "created_entities_by_type": Counter(),
        "mentions_added": 0,
        "mentions_by_type": Counter(),
        "embedding_matches": 0,
        "embedding_matches_by_type": Counter(),
        "embedding_below_threshold": 0,
        "embedding_below_threshold_by_type": Counter(),
        "embedding_graph_no_match": 0,
        "embedding_graph_no_match_by_type": Counter(),
        "string_matches": 0,
        "string_matches_by_type": Counter(),
        "string_no_matches": 0,
        "string_no_matches_by_type": Counter(),
        "model_source_matches": 0,
        "model_source_no_matches": 0,
        "llm_fallback_attempts": 0,
        "llm_fallback_matches": 0,
        "llm_fallback_no_matches": 0,
        "llm_fallback_no_candidates": 0,
        "consolidation": {
            "calls": 0,
            "reused_existing": 0,
            "created_new": 0,
            "source_component_llm_checks": 0,
            "source_component_llm_confirmed": 0,
            "source_component_llm_rejected": 0,
            "by_type": {},
        },
        "graph_delta": {},
        "saturation": {},
    }


def serialise_stats(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: serialise_stats(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: serialise_stats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialise_stats(item) for item in value]
    return value


def normalise_llm_choice(text: str) -> str:
    cleaned = text.strip().strip("`").strip()
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = cleaned.lstrip("-*0123456789. )(").strip()
    return cleaned.strip("\"'").rstrip(".").strip()


def load_variable_long_name_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return read_json(path)


def variable_from_long_name(long_name: str, lookup_dict: dict[str, str]) -> str:
    reverse_lookup = {value: key for key, value in lookup_dict.items()}
    return reverse_lookup.get(long_name, long_name)


def choose_llm_match(
    llm_output: str,
    matches: list[tuple[str, float]],
    variable_long_names: dict[str, str],
) -> str | None:
    choice = normalise_llm_choice(llm_output)
    if choice.casefold() == NO_MATCH:
        return None

    candidates: dict[str, str] = {}
    for instance, _ in matches:
        candidates[instance.casefold()] = instance
        long_name = variable_long_names.get(instance, instance)
        candidates[long_name.casefold()] = instance

    transformed_choice = variable_from_long_name(choice, variable_long_names)
    return candidates.get(choice.casefold()) or candidates.get(transformed_choice.casefold())


class GeminiFallbackVerifier:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            with _STATE_LOCK:
                if self._client is None:
                    from google.genai import Client

                    self._client = Client()
        return self._client

    def verify(self, prompt: str) -> str | None:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        if response is None or response.text is None:
            return None
        return response.text.strip()


def run_llm_fallback_for_reason(
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    verifier: GeminiFallbackVerifier,
    variable_long_names: dict[str, str],
    triplet: dict[str, Any],
    remark: dict[str, Any],
    stats: dict[str, Any],
    side: str,
    reason: str,
    fallback_threshold: float,
    fallback_top_k: int,
    require_embedding_graph_check: bool,
) -> bool:
    if reason not in remark["unmatch_reasons"]:
        return False

    entity_type = triplet.get(f"{side}_entity_type")
    entity_instance = triplet_instance(triplet, side)
    paragraph = triplet.get("paragraph", "")
    stats["llm_fallback_attempts"] += 1
    matches = embedding_match_with_tolerance(
        embed_model,
        entity_type,
        entity_instance,
        embedding_files,
        fallback_threshold,
        fallback_top_k,
    )

    fallback_data = {
        "side": side,
        "candidate_entity_type": entity_type,
        "candidate_entity_instance": entity_instance,
        "candidate_matches": [
            {"instance": instance, "score": score} for instance, score in matches
        ],
        "matched_instance": None,
        "llm_response": None,
        "prompt": None,
    }
    fallback_container = remark.setdefault("llm_fallback_data", {"attempts": []})
    if isinstance(fallback_container, list):
        fallback_container = {"attempts": fallback_container}
        remark["llm_fallback_data"] = fallback_container
    fallback_container.setdefault("attempts", []).append(fallback_data)
    fallback_container.update(
        {
            "candidate_entity_type": entity_type,
            "candidate_entity_instance": entity_instance,
            "side": side,
        }
    )

    if not matches:
        stats["llm_fallback_no_candidates"] += 1
        stats["llm_fallback_no_matches"] += 1
        return False

    display_instances = [
        variable_long_names.get(instance, instance)
        for instance, _ in matches
    ]
    prompt = EMBEDDING_FALLBACK_LLM_PROMPT.format(
        key_word_phrase=entity_instance,
        entity_type=entity_type,
        instances="\n".join(f"- {instance}" for instance in display_instances),
        paragraph=paragraph,
    )
    fallback_data["prompt"] = prompt
    fallback_container["prompt"] = prompt
    llm_output = verifier.verify(prompt)
    fallback_data["llm_response"] = llm_output
    fallback_container["llm_response"] = llm_output
    if llm_output is None:
        stats["llm_fallback_no_matches"] += 1
        return False

    matched_instance = choose_llm_match(llm_output, matches, variable_long_names)
    if matched_instance is None:
        stats["llm_fallback_no_matches"] += 1
        return False

    resolved_instance, graph_check = graph_checked_embedding_match(
        session,
        entity_type,
        matched_instance,
        require_embedding_graph_check,
    )
    fallback_data["graph_database_check"] = graph_check
    fallback_container["graph_database_check"] = graph_check
    if resolved_instance is None:
        stats["llm_fallback_no_matches"] += 1
        return False

    set_triplet_instance(triplet, side, resolved_instance)
    fallback_data["matched_instance"] = resolved_instance
    fallback_container["matched_instance"] = resolved_instance
    remark["unmatch_reasons"] = [
        item for item in remark["unmatch_reasons"] if item != reason
    ]
    remark[f"{side}_entity"] = resolved_instance
    remark.setdefault("remarks", []).append(f"{side.upper()}_LLM_FALLBACK_MATCH")
    # The decision dict was written by the embedding stage and still says the
    # mention was refused. Bring it in line with what we just wrote to the
    # triple, so anything reading the per-mention record sees the same identity
    # the graph gets: an analysis that counts decision["matched"] is otherwise
    # counting a kept triple as a NIL and the raw surface string as an entity.
    decision = remark.get(f"{side}_decision")
    if isinstance(decision, dict):
        decision["matched"] = True
        decision["resolved_instance"] = resolved_instance
        decision["graph_database_check"] = graph_check
        decision["llm_fallback"] = True
    stats["llm_fallback_matches"] += 1
    return True


def relation_set(paper: dict[str, Any]) -> set[tuple[str, str, str, str, str]]:
    result = set()
    for relation_name, relation_items in paper.get("create_relations", {}).items():
        for item in relation_items:
            result.add(
                (
                    relation_name,
                    item.get("source_entity_type", ""),
                    item.get("source_entity", ""),
                    item.get("target_entity_type", ""),
                    item.get("target_entity", ""),
                )
            )
    return result


def entity_set(paper: dict[str, Any], key: str) -> set[tuple[str, str]]:
    result = set()
    for entity_type, instances in paper.get(key, {}).items():
        for instance in instances:
            result.add((entity_type, instance))
    return result


def add_graph_delta_and_saturation(
    paper: dict[str, Any],
    paper_stats: dict[str, Any],
    cumulative: dict[str, Any],
) -> None:
    # Serialised under _STATE_LOCK: the whole per-paper delta must be computed and
    # the cumulative "seen" sets updated atomically so concurrent papers don't
    # corrupt them or double-count. (The stored per-paper delta becomes order-
    # sensitive, but E1/E2/E6 re-derive curves via permutation replay, so this
    # does not affect the saturation analysis.)
    with _STATE_LOCK:
        _add_graph_delta_and_saturation_locked(paper, paper_stats, cumulative)


def _add_graph_delta_and_saturation_locked(
    paper: dict[str, Any],
    paper_stats: dict[str, Any],
    cumulative: dict[str, Any],
) -> None:
    paper_create_entities = entity_set(paper, "create_entities")
    paper_mentions = entity_set(paper, "mentions")
    paper_relations = relation_set(paper)

    seen_create_entities = cumulative.setdefault("seen_create_entities", set())
    seen_mentions = cumulative.setdefault("seen_mentions", set())
    seen_relations = cumulative.setdefault("seen_relations", set())

    new_create_entities = paper_create_entities - seen_create_entities
    new_mentions = paper_mentions - seen_mentions
    new_relations = paper_relations - seen_relations

    seen_create_entities.update(paper_create_entities)
    seen_mentions.update(paper_mentions)
    seen_relations.update(paper_relations)

    cumulative["papers_processed"] = cumulative.get("papers_processed", 0) + 1
    cumulative["triplets_seen"] = cumulative.get("triplets_seen", 0) + paper_stats["triplets_total"]
    cumulative["triplets_considered"] = cumulative.get("triplets_considered", 0) + paper_stats["triplets_considered"]
    cumulative["relations_created"] = cumulative.get("relations_created", 0) + paper_stats["relations_created"]
    cumulative["created_entities_added"] = cumulative.get("created_entities_added", 0) + paper_stats["created_entities_added"]

    relations_created = max(1, paper_stats["relations_created"])
    created_entities_added = max(1, paper_stats["created_entities_added"])
    triplets_considered = max(1, cumulative["triplets_considered"])

    paper_stats["graph_delta"] = {
        "new_unique_create_entities": len(new_create_entities),
        "new_unique_mentions": len(new_mentions),
        "new_unique_relations": len(new_relations),
        "paper_unique_create_entities": len(paper_create_entities),
        "paper_unique_mentions": len(paper_mentions),
        "paper_unique_relations": len(paper_relations),
    }
    paper_stats["saturation"] = {
        "paper_new_entity_rate": len(new_create_entities) / created_entities_added,
        "paper_new_relation_rate": len(new_relations) / relations_created,
        "cumulative_unique_create_entities": len(seen_create_entities),
        "cumulative_unique_mentions": len(seen_mentions),
        "cumulative_unique_relations": len(seen_relations),
        "cumulative_relation_yield_per_considered_triplet": len(seen_relations) / triplets_considered,
        "cumulative_created_entity_yield_per_considered_triplet": len(seen_create_entities) / triplets_considered,
    }


def process_models(session: Any, paper: dict[str, Any], stats: dict[str, Any]) -> None:
    models = paper.get("model")
    if not models:
        return
    if isinstance(models, str):
        models = [models]

    aligned_models = []
    for model in models:
        matched_db_name = entity_exists(session, "Source", normalise_instance(model))
        if matched_db_name is not None:
            aligned_models.append(matched_db_name)
            stats["model_source_matches"] += 1
        else:
            stats["model_source_no_matches"] += 1
    paper["model"] = aligned_models


def reject_triplet(remark: dict[str, Any], reason: str, stats: dict[str, Any]) -> None:
    remark["matched"] = False
    remark["unmatch_reasons"].append(reason)
    if reason == ONTOLOGY_CONFORMANCE_REJECTED:
        stats["rejected_ontology_conformance"] += 1
    elif reason == EVIDENCE_VALIDATION_REJECTED:
        stats["rejected_evidence_validation"] += 1
    elif reason == VALIDATION_REJECTED:
        stats["rejected_validation"] += 1
    elif reason == EVALUATION_REJECTED:
        stats["rejected_evaluation"] += 1


def process_triplets(
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    paper: dict[str, Any],
    stats: dict[str, Any],
    args: argparse.Namespace,
    verifier: GeminiFallbackVerifier | None,
    variable_long_names: dict[str, str],
) -> None:
    triplets = paper.get("kg_extraction_triplets", []) or []
    stats["triplets_total"] = len(triplets)
    paper["matching_remarks"] = []

    require_embedding_graph_check = not args.no_require_embedding_graph_check

    for idx, triplet in enumerate(triplets):
        source_instance = triplet_instance(triplet, "source")
        target_instance = triplet_instance(triplet, "target")
        set_triplet_instance(triplet, "source", source_instance)
        set_triplet_instance(triplet, "target", target_instance)

        source_type = triplet.get("source_entity_type", "")
        target_type = triplet.get("target_entity_type", "")
        relation_name = triplet.get("relation", "")
        paragraph = triplet.get("paragraph", "")
        remark = {
            "triplet_index": idx,
            "matched": True,
            "unmatch_reasons": [],
            "remarks": [],
            "source_entity_type": source_type,
            "source_entity": source_instance,
            "target_entity_type": target_type,
            "target_entity": target_instance,
            "relation": relation_name,
            "graph_action": None,
            "source_decision": None,
            "target_decision": None,
            "llm_fallback_used": False,
            "llm_fallback_data": {"attempts": []},
        }

        if not flag_is_accepted(triplet, "ontology_conformed"):
            reject_triplet(remark, ONTOLOGY_CONFORMANCE_REJECTED, stats)
        if not flag_is_accepted(triplet, "evidence_validated"):
            reject_triplet(remark, EVIDENCE_VALIDATION_REJECTED, stats)
        if not flag_is_accepted(triplet, "valid"):
            reject_triplet(remark, VALIDATION_REJECTED, stats)
        if not flag_is_accepted(triplet, "eval"):
            reject_triplet(remark, EVALUATION_REJECTED, stats)

        if remark["unmatch_reasons"]:
            remark["graph_action"] = "discard_relation"
            paper["matching_remarks"].append(remark)
            stats["triplets_unmatched"] += 1
            continue

        stats["triplets_considered"] += 1
        source_matched, resolved_source, source_reason, source_decision = link_entity_for_triplet(
            session,
            embed_model,
            embedding_files,
            paper,
            stats,
            "source",
            source_type,
            source_instance,
            args.embedding_threshold,
            require_embedding_graph_check,
            verifier,
            paragraph,
        )
        target_matched, resolved_target, target_reason, target_decision = link_entity_for_triplet(
            session,
            embed_model,
            embedding_files,
            paper,
            stats,
            "target",
            target_type,
            target_instance,
            args.embedding_threshold,
            require_embedding_graph_check,
            verifier,
            paragraph,
        )

        set_triplet_instance(triplet, "source", resolved_source)
        set_triplet_instance(triplet, "target", resolved_target)
        remark["source_entity"] = resolved_source
        remark["target_entity"] = resolved_target
        remark["source_decision"] = source_decision
        remark["target_decision"] = target_decision

        if not source_matched and source_reason is not None:
            remark["unmatch_reasons"].append(source_reason)
        if not target_matched and target_reason is not None:
            remark["unmatch_reasons"].append(target_reason)

        fallback_reasons = (
            ("source", SOURCE_EMBEDDING_BELOW_THRESHOLD),
            ("source", SOURCE_GRAPH_NO_MATCH),
            ("target", TARGET_EMBEDDING_BELOW_THRESHOLD),
            ("target", TARGET_GRAPH_NO_MATCH),
        )
        if (
            verifier is not None
            and not args.skip_llm_fallback
            and any(reason in remark["unmatch_reasons"] for _, reason in fallback_reasons)
        ):
            fallback_used = False
            for side, reason in fallback_reasons:
                fallback_used = (
                    run_llm_fallback_for_reason(
                        session,
                        embed_model,
                        embedding_files,
                        verifier,
                        variable_long_names,
                        triplet,
                        remark,
                        stats,
                        side,
                        reason,
                        args.fallback_threshold,
                        args.fallback_top_k,
                        require_embedding_graph_check,
                    )
                    or fallback_used
                )
            remark["llm_fallback_used"] = fallback_used

        if not remark["unmatch_reasons"]:
            final_source = triplet_instance(triplet, "source")
            final_target = triplet_instance(triplet, "target")
            remark["matched"] = True
            remark["source_entity"] = final_source
            remark["target_entity"] = final_target
            remark["graph_action"] = "create_relation"
            add_relation(
                paper,
                stats,
                idx,
                relation_name,
                final_source,
                source_type,
                final_target,
                target_type,
                paragraph,
                source_decision.get("strategy") if source_decision else None,
                target_decision.get("strategy") if target_decision else None,
            )
            stats["triplets_matched"] += 1
        else:
            remark["matched"] = False
            remark["graph_action"] = "discard_relation"
            stats["triplets_unmatched"] += 1

        paper["matching_remarks"].append(remark)


def process_mentions(
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    paper: dict[str, Any],
    stats: dict[str, Any],
    args: argparse.Namespace,
    valid_entity_types: set[str],
    verifier: GeminiFallbackVerifier | None,
) -> None:
    require_embedding_graph_check = not args.no_require_embedding_graph_check
    for entity_data in paper.get("entities_extracted", []) or []:
        entity_type = entity_data.get("entity_type")
        entity_instance = normalise_instance(entity_data.get("entity_instance"))
        if not entity_type or not entity_instance or entity_type not in valid_entity_types:
            continue
        link_entity_for_mention(
            session,
            embed_model,
            embedding_files,
            paper,
            stats,
            entity_type,
            entity_instance,
            args.embedding_threshold,
            require_embedding_graph_check,
            verifier,
        )

    for entity_type, instances in list(paper.get("create_entities", {}).items()):
        for instance in instances:
            add_mention(paper, stats, entity_type, instance)


def process_paper(
    paper_id: str,
    source_date: str | None,
    paper_data: dict[str, Any],
    session: Any,
    embed_model: Any,
    embedding_files: dict[str, Any],
    args: argparse.Namespace,
    verifier: GeminiFallbackVerifier | None,
    variable_long_names: dict[str, str],
    valid_entity_types: set[str],
    cumulative: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paper = copy.deepcopy(paper_data)
    paper["create_entities"] = {}
    paper["create_relations"] = {}
    paper["mentions"] = {}
    paper["matched_index_relations"] = []
    if source_date is not None:
        paper["entity_linking_source_date"] = source_date

    stats = make_empty_paper_stats(paper_id, source_date)
    process_models(session, paper, stats)
    process_triplets(
        session,
        embed_model,
        embedding_files,
        paper,
        stats,
        args,
        verifier,
        variable_long_names,
    )
    process_mentions(
        session,
        embed_model,
        embedding_files,
        paper,
        stats,
        args,
        valid_entity_types,
        verifier,
    )
    add_graph_delta_and_saturation(paper, stats, cumulative)
    paper["entity_linking_stats"] = serialise_stats(stats)
    return paper, serialise_stats(stats)


def rebuild_cumulative_from_existing(processed: dict[str, Any]) -> dict[str, Any]:
    cumulative: dict[str, Any] = {
        "seen_create_entities": set(),
        "seen_mentions": set(),
        "seen_relations": set(),
        "papers_processed": 0,
        "triplets_seen": 0,
        "triplets_considered": 0,
        "relations_created": 0,
        "created_entities_added": 0,
    }
    for paper in processed.values():
        cumulative["seen_create_entities"].update(entity_set(paper, "create_entities"))
        cumulative["seen_mentions"].update(entity_set(paper, "mentions"))
        cumulative["seen_relations"].update(relation_set(paper))
        stats = paper.get("entity_linking_stats", {})
        cumulative["papers_processed"] += 1
        cumulative["triplets_seen"] += stats.get("triplets_total", len(paper.get("kg_extraction_triplets", []) or []))
        cumulative["triplets_considered"] += stats.get("triplets_considered", 0)
        cumulative["relations_created"] += stats.get("relations_created", 0)
        cumulative["created_entities_added"] += stats.get("created_entities_added", 0)
    return cumulative


def serialise_cumulative(cumulative: dict[str, Any]) -> dict[str, Any]:
    return {
        "papers_processed": cumulative.get("papers_processed", 0),
        "triplets_seen": cumulative.get("triplets_seen", 0),
        "triplets_considered": cumulative.get("triplets_considered", 0),
        "relations_created": cumulative.get("relations_created", 0),
        "created_entities_added": cumulative.get("created_entities_added", 0),
        "unique_create_entities": len(cumulative.get("seen_create_entities", set())),
        "unique_mentions": len(cumulative.get("seen_mentions", set())),
        "unique_relations": len(cumulative.get("seen_relations", set())),
    }


def main() -> None:
    args = parse_args()
    load_environment()

    evaluated_base = SCRIPT_DIR / "output" / "evaluated_papers"
    input_dir_arg = args.input_dir.expanduser().resolve() if args.input_dir is not None else None
    paper_paths, input_dirs, dates, duplicate_input_files = collect_paper_paths(
        evaluated_base,
        input_dir_arg,
        args.date,
    )
    deduplicated_input_file_count = len(paper_paths)
    output_dir = SCRIPT_DIR / "output" / "entity_linked_papers"
    output_path = resolve_path(
        args.output_path,
        output_dir / "v2_processed_papers_entity_linking_post_llm.json",
    )
    stats_path = resolve_path(args.stats_path, output_dir / "entity_linking_stats.json")
    embedding_root = resolve_path(args.embedding_root, default_embedding_root())
    variable_long_name_path = resolve_path(
        args.variable_long_name_path,
        default_variable_long_name_path(),
    )

    if args.max_papers is not None:
        paper_paths = paper_paths[: args.max_papers]
    if not paper_paths:
        raise FileNotFoundError(f"No evaluated paper JSON files found in {input_dirs}")

    if output_path.exists() and not args.force:
        processed_papers = read_json(output_path)
    else:
        processed_papers = {}

    cumulative = rebuild_cumulative_from_existing(processed_papers)
    stats_output: dict[str, Any] = {
        "run": {
            "date": args.date,
            "dates": dates,
            "input_dirs": [str(path) for path in input_dirs],
            "output_path": str(output_path),
            "stats_path": str(stats_path),
            "embedding_root": str(embedding_root),
            "variable_long_name_path": str(variable_long_name_path),
            "embedding_model": args.embedding_model,
            "embedding_threshold": args.embedding_threshold,
            "fallback_threshold": args.fallback_threshold,
            "fallback_top_k": args.fallback_top_k,
            "llm_model": args.llm_model,
            "skip_llm_fallback": args.skip_llm_fallback,
            "source_component_llm_confirmation": True,
            "require_embedding_graph_check": not args.no_require_embedding_graph_check,
            "input_files": deduplicated_input_file_count + len(duplicate_input_files),
            "deduplicated_input_files": deduplicated_input_file_count,
            "selected_input_files": len(paper_paths),
            "max_papers": args.max_papers,
            "duplicate_input_files": duplicate_input_files,
            "processed_files": 0,
            "skipped_existing": 0,
            "failed_files": [],
        },
        "papers": [],
        "cumulative": serialise_cumulative(cumulative),
    }

    from neo4j import GraphDatabase

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda value, **_: value  # type: ignore

    embed_model = load_embedding_model(args.embedding_model)
    embedding_files = load_embedding_files(embedding_root)
    variable_long_names = load_variable_long_name_lookup(variable_long_name_path)
    verifier = GeminiFallbackVerifier(args.llm_model)
    valid_entity_types = all_entity_types()

    with GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    ) as driver:
        driver.verify_connectivity()

        # Only submit papers not already linked (snapshot the skip set up front).
        pending: list[tuple[str | None, Path]] = []
        for source_date, paper_path in paper_paths:
            if paper_path.stem in processed_papers and not args.force:
                stats_output["run"]["skipped_existing"] += 1
            else:
                pending.append((source_date, paper_path))

        # Parallel mode: papers run concurrently so their network-bound Gemini
        # fallback calls overlap; shared state (embedding index, cumulative
        # tracker, output aggregation) is guarded by the module locks. Each task
        # uses its own Neo4j session (a session is not thread-safe; the driver is).
        def link_one(item: tuple[str | None, Path]):
            source_date, paper_path = item
            paper_id = paper_path.stem
            try:
                paper_data = read_json(paper_path)
                with driver.session() as session:
                    processed_paper, paper_stats = process_paper(
                        paper_id, source_date, paper_data, session,
                        embed_model, embedding_files, args, verifier,
                        variable_long_names, valid_entity_types, cumulative,
                    )
                return ("ok", paper_id, source_date, str(paper_path), processed_paper, paper_stats)
            except Exception as exc:  # noqa: BLE001
                return ("err", paper_id, source_date, str(paper_path), None, exc)

        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [executor.submit(link_one, item) for item in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Entity linking v2 papers"):
                status, paper_id, source_date, paper_path_str, processed_paper, payload = future.result()
                with _STATE_LOCK:
                    if status == "ok":
                        processed_papers[paper_id] = processed_paper
                        stats_output["papers"].append(payload)
                        stats_output["run"]["processed_files"] += 1
                        stats_output["cumulative"] = serialise_cumulative(cumulative)
                    else:
                        stats_output["run"]["failed_files"].append({
                            "paper_id": paper_id,
                            "source_date": source_date,
                            "path": paper_path_str,
                            "error": repr(payload),
                        })
                        if args.stop_on_error:
                            raise payload
                    completed += 1
                    if args.save_every > 0 and completed % args.save_every == 0:
                        write_json(output_path, processed_papers)
                        write_json(stats_path, serialise_stats(stats_output))

    write_json(output_path, processed_papers)
    write_json(stats_path, serialise_stats(stats_output))
    print(f"Wrote entity-linked papers to {output_path}")
    print(f"Wrote entity-linking stats to {stats_path}")


if __name__ == "__main__":
    main()
