#!/usr/bin/env bash
# Unpack the released 2,000-publication data from data/ into the directories the
# pipeline scripts read from and write to.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
EXT="$ROOT/extraction"

echo "Paper metadata -> extraction/papers/metadata/"
mkdir -p "$EXT/papers/metadata"
cp "$DATA"/papers/*.json "$EXT/papers/metadata/"

echo "Stage outputs -> extraction/output/<stage>/<date>/"
for archive in "$DATA"/pipeline_outputs/*_papers_*.tar.gz; do
  name="$(basename "$archive" .tar.gz)"
  stage="${name%_*}"
  mkdir -p "$EXT/output/$stage"
  tar -xzf "$archive" -C "$EXT/output/$stage"
done

echo "Entity-linked corpus -> extraction/output/entity_linked_papers/"
mkdir -p "$EXT/output/entity_linked_papers"
gunzip -c "$DATA/pipeline_outputs/entity_linked_papers_2000.json.gz" \
  > "$EXT/output/entity_linked_papers/v2_processed_papers_entity_linking_post_llm.json"

echo "KG dump -> kg_create/neo4j/{cypher,json}/"
mkdir -p "$ROOT/kg_create/neo4j/cypher" "$ROOT/kg_create/neo4j/json"
for f in "$DATA"/kg/*.cypher.gz; do gunzip -c "$f" > "$ROOT/kg_create/neo4j/cypher/$(basename "$f" .gz)"; done
for f in "$DATA"/kg/*.json.gz; do gunzip -c "$f" > "$ROOT/kg_create/neo4j/json/$(basename "$f" .gz)"; done

echo "Done."
