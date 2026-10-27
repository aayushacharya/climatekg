"""Build the Hugging Face release of ClimateKG as Parquet node and edge tables.

Two versions go into one dataset repo, one folder each:

  v1/  the 2,000-publication KG of the AACL 2026 paper, read from data/kg/*_20260622.json.gz
  v2/  the expanded 10,910-publication KG, exported from a running Neo4j instance

Each folder holds nodes.parquet and edges.parquet. The dataset card in
huggingface/README.md declares them as the configs v1-nodes, v1-edges, v2-nodes and
v2-edges. Properties differ by node label, so every table keeps the common columns
and stores the remaining properties as a JSON string.

Usage (from the repo root):
  python scripts/build_hf_dataset.py --out hf_release                 # both versions
  python scripts/build_hf_dataset.py --out hf_release --skip-v2       # v1 only, no Neo4j needed
"""
import argparse
import gzip
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("label", pa.string()),
    ("name", pa.string()),
    ("properties", pa.string()),
])
EDGE_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("source_label", pa.string()),
    ("relation", pa.string()),
    ("target", pa.string()),
    ("target_label", pa.string()),
    ("properties", pa.string()),
])
BATCH_SIZE = 100_000
# Set by v2/site's paper submission form on papers that skipped the extraction pipeline
SUBMISSION_PROVENANCE = "user_submission"


def node_row(label, props):
    props = dict(props)
    uuid = props.pop("uuid")
    name = props.get("name", props.get("title"))
    return {"uuid": uuid, "label": label, "name": None if name is None else str(name),
            "properties": json.dumps(props, ensure_ascii=False, default=str)}


def edge_row(source, source_label, relation, target, target_label, props):
    return {"source": source, "source_label": source_label, "relation": relation,
            "target": target, "target_label": target_label,
            "properties": json.dumps(dict(props or {}), ensure_ascii=False, default=str)}


class BatchedWriter:
    """Write rows to a Parquet file in batches, so the 2M-edge export never sits in memory."""

    def __init__(self, path, schema):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.rows = []
        self.count = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= BATCH_SIZE:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows)
            self.rows = []

    def close(self):
        self.flush()
        self.writer.close()
        return self.count


def build_v1(kg_dir, date, out_dir):
    """Convert the released JSON dump. Edges to ESGF Source nodes missing from the
    metadata snapshot are dropped, as Neo4j drops them on load."""
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(kg_dir, f"nodes_{date}.json.gz"), "rt") as f:
        nodes = json.load(f)
    label_of = {}
    writer = BatchedWriter(os.path.join(out_dir, "nodes.parquet"), NODE_SCHEMA)
    for label, by_uuid in tqdm(nodes.items(), desc="v1 nodes"):
        for uuid, props in by_uuid.items():
            label_of[uuid] = label
            writer.add(node_row(label, props))
    n_nodes = writer.close()
    del nodes

    with gzip.open(os.path.join(kg_dir, f"edges_{date}.json.gz"), "rt") as f:
        edges = json.load(f)
    writer = BatchedWriter(os.path.join(out_dir, "edges.parquet"), EDGE_SCHEMA)
    dangling = 0
    for source, targets in tqdm(edges.items(), desc="v1 edges"):
        for target, relations in targets.items():
            for relation, props in relations.items():
                if source not in label_of or target not in label_of:
                    dangling += 1
                    continue
                writer.add(edge_row(source, label_of[source], relation,
                                    target, label_of[target], props))
    n_edges = writer.close()
    print(f"v1: {n_nodes:,} nodes, {n_edges:,} edges ({dangling:,} dangling edges dropped)")


def submitted_uuids(session):
    """Uuids of papers added by hand through the site's submission form, plus the
    Author, Keyword and Venue nodes that only those papers reference. The form marks
    the Paper node with provenance but MERGEs the others without a marker."""
    papers = {r["u"] for r in session.run(
        "MATCH (p:Paper {provenance: $prov}) RETURN p.uuid AS u", prov=SUBMISSION_PROVENANCE)}
    orphans = {r["u"] for r in session.run(
        "MATCH (p:Paper {provenance: $prov})--(n) "
        "WHERE (n:Author OR n:Keyword OR n:Venue) "
        "AND NOT EXISTS { MATCH (n)--(q) WHERE coalesce(q.provenance, '') <> $prov } "
        "RETURN DISTINCT n.uuid AS u", prov=SUBMISSION_PROVENANCE)}
    return papers, orphans


def build_v2(uri, auth, database, out_dir):
    """Export the loaded graph. Labels starting with '_' are neosemantics (n10s)
    bookkeeping left by export_rdf.py and are not part of the KG. Hand-submitted
    papers are left out, so v2 holds only what the extraction pipeline produced."""
    from neo4j import GraphDatabase

    os.makedirs(out_dir, exist_ok=True)
    kg_node = "NOT any(l IN labels(n) WHERE l STARTS WITH '_')"
    with GraphDatabase.driver(uri, auth=auth) as driver, driver.session(database=database) as session:
        papers, orphans = submitted_uuids(session)
        excluded = papers | orphans
        print(f"v2: leaving out {len(papers):,} submitted papers and {len(orphans):,} nodes only they reference")

        total = session.run(f"MATCH (n) WHERE {kg_node} RETURN count(n) AS c").single()["c"]
        writer = BatchedWriter(os.path.join(out_dir, "nodes.parquet"), NODE_SCHEMA)
        result = session.run(f"MATCH (n) WHERE {kg_node} RETURN labels(n)[0] AS label, properties(n) AS props")
        for record in tqdm(result, total=total, desc="v2 nodes"):
            if record["props"]["uuid"] not in excluded:
                writer.add(node_row(record["label"], record["props"]))
        n_nodes = writer.close()

        kg_edge = kg_node.replace("(n)", "(a)") + " AND " + kg_node.replace("(n)", "(b)")
        total = session.run(f"MATCH (a)-[r]->(b) WHERE {kg_edge} RETURN count(r) AS c").single()["c"]
        writer = BatchedWriter(os.path.join(out_dir, "edges.parquet"), EDGE_SCHEMA)
        result = session.run(
            f"MATCH (a)-[r]->(b) WHERE {kg_edge} "
            "RETURN a.uuid AS source, labels(a)[0] AS source_label, type(r) AS relation, "
            "b.uuid AS target, labels(b)[0] AS target_label, properties(r) AS props")
        for record in tqdm(result, total=total, desc="v2 edges"):
            if record["source"] in excluded or record["target"] in excluded:
                continue
            writer.add(edge_row(record["source"], record["source_label"], record["relation"],
                                record["target"], record["target_label"], record["props"]))
        n_edges = writer.close()
    print(f"v2: {n_nodes:,} nodes, {n_edges:,} edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Hugging Face release (Parquet + dataset card).")
    parser.add_argument("--out", default=os.path.join(ROOT, "hf_release"), help="Output folder (default: hf_release/)")
    parser.add_argument("--v1-date", default="20260622", help="Suffix of the data/kg/ JSON dump for v1")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "neo4j://localhost:8687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.skip_v1:
        build_v1(os.path.join(ROOT, "data", "kg"), args.v1_date, os.path.join(args.out, "v1"))
    if not args.skip_v2:
        build_v2(args.neo4j_uri, (args.neo4j_user, args.neo4j_password), args.neo4j_database,
                 os.path.join(args.out, "v2"))
    shutil.copy(os.path.join(ROOT, "huggingface", "README.md"), os.path.join(args.out, "README.md"))
    print(f"Wrote {args.out}")
