#!/usr/bin/env python3
"""Incrementally create v2 entity-linking embedding files from Neo4j."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from consts import EMBEDDING_MATCH_ENTITIES
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_ROOT = SCRIPT_DIR / "embedding_files"
DEFAULT_MODEL = "nvidia/NV-Embed-v2"
STRICT_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the current Neo4j graph and embed only entity names that are "
            "missing from v2/extraction/embedding_files."
        )
    )
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:8687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Directory containing <EntityType>.json files.",
    )
    parser.add_argument(
        "--entity-type",
        action="append",
        choices=sorted(EMBEDDING_MATCH_ENTITIES),
        help=(
            "Optional entity type to update. Can be passed more than once. "
            "Defaults to all EMBEDDING_MATCH_ENTITIES."
        ),
    )
    parser.add_argument(
        "--variable-long-name-output",
        type=Path,
        default=SCRIPT_DIR / "output" / "variable_name_to_long_name.json",
        help="Output path for Variable name -> long-name lookup.",
    )
    parser.add_argument(
        "--skip-variable-long-name-lookup",
        action="store_true",
        help="Do not write the Variable name -> long-name lookup file.",
    )
    return parser.parse_args()


def load_embedding_model(model_name: str) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
    )


def load_embedding_file(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    tmp_path.replace(path)


def clean_embed_text(value: Any, fallback: str) -> str:
    if value is None or value == "":
        value = fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    value = str(value)
    return value.replace("[", "").replace("]", "").strip() or fallback


def retrieve_entities(session: Any, entity_type: str, entity_key: str) -> list[dict[str, str]]:
    if not STRICT_LABEL_RE.match(entity_type) or not STRICT_LABEL_RE.match(entity_key):
        raise ValueError(f"Unsafe entity type or key: {entity_type}.{entity_key}")

    query = f"""
    MATCH (n:{entity_type})
    WHERE n.name IS NOT NULL
    RETURN n.name AS name, n.{entity_key} AS embed_value
    ORDER BY n.name
    """
    records = []
    for record in session.run(query):
        name = str(record["name"])
        records.append(
            {
                "name": name,
                "embed_text": clean_embed_text(record.get("embed_value"), name),
            }
        )
    return records


def update_embeddings_for_entity_type(
    embed_model: Any,
    records: list[dict[str, str]],
    entity_type: str,
    output_path: Path,
) -> dict[str, int]:
    embedding_dict = load_embedding_file(output_path)
    before_count = len(embedding_dict)

    for record in tqdm(records, desc=f"Embedding {entity_type}"):
        name = record["name"]
        if name in embedding_dict:
            continue
        embedding_dict[name] = embed_model.get_text_embedding(record["embed_text"])

    write_json(output_path, embedding_dict)
    return {
        "graph_nodes": len(records),
        "existing_embeddings": before_count,
        "created_embeddings": len(embedding_dict) - before_count,
        "total_embeddings": len(embedding_dict),
    }


def write_variable_long_name_lookup(
    records: list[dict[str, str]],
    output_path: Path,
) -> None:
    lookup = {
        record["name"]: record["embed_text"]
        for record in records
    }
    write_json(output_path, lookup)


def main() -> None:
    args = parse_args()
    entity_types = args.entity_type or sorted(EMBEDDING_MATCH_ENTITIES)
    embedding_root = args.embedding_root.expanduser().resolve()
    embedding_root.mkdir(parents=True, exist_ok=True)

    from neo4j import GraphDatabase

    embed_model = load_embedding_model(args.embedding_model)
    summary: dict[str, dict[str, int]] = {}

    with GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    ) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            for entity_type in entity_types:
                entity_key = EMBEDDING_MATCH_ENTITIES[entity_type]
                records = retrieve_entities(session, entity_type, entity_key)
                output_path = embedding_root / f"{entity_type}.json"
                summary[entity_type] = update_embeddings_for_entity_type(
                    embed_model,
                    records,
                    entity_type,
                    output_path,
                )
                if entity_type == "Variable" and not args.skip_variable_long_name_lookup:
                    write_variable_long_name_lookup(
                        records,
                        args.variable_long_name_output.expanduser().resolve(),
                    )

    print("\nEmbedding update summary")
    print(f"  embedding_root: {embedding_root}")
    for entity_type, stats in summary.items():
        print(
            f"  {entity_type}: graph_nodes={stats['graph_nodes']} "
            f"existing={stats['existing_embeddings']} "
            f"created={stats['created_embeddings']} "
            f"total={stats['total_embeddings']}"
        )


if __name__ == "__main__":
    main()
