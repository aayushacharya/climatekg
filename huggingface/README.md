---
license: cc-by-4.0
pretty_name: ClimateKG
language:
  - en
tags:
  - knowledge-graph
  - climate
  - cmip6
  - esgf
  - scientific-literature
task_categories:
  - graph-ml
  - question-answering
size_categories:
  - 1M<n<10M
configs:
  - config_name: v2-nodes
    default: true
    data_files:
      - split: train
        path: v2/nodes.parquet
  - config_name: v2-edges
    data_files:
      - split: train
        path: v2/edges.parquet
  - config_name: v1-nodes
    data_files:
      - split: train
        path: v1/nodes.parquet
  - config_name: v1-edges
    data_files:
      - split: train
        path: v1/edges.parquet
---

# ClimateKG

A climate-science knowledge graph built from ESGF dataset metadata, CMIP6 controlled vocabularies, GCMD keywords and GeoNames, then enriched with facts that LLM agents extracted from scientific publications. Each extracted triplet had to conform to the ontology, be grounded in an evidence span of the paper, pass an LLM validator and link to canonical KG entities before it entered the graph.

Paper: [Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications](https://arxiv.org/abs/2610.32725) (AACL 2026). Code: [github.com/aayushacharya/climatekg](https://github.com/aayushacharya/climatekg).

## Versions

| Version | Publications | Nodes | Edges | Node labels | Relation types |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v1` (AACL 2026 paper) | 2,000 | 133,990 | 755,310 | 49 | 59 |
| `v2` (expanded corpus) | 10,910 | 373,684 | 2,009,392 | 49 | 65 |

`v1` is the graph evaluated in the paper. Use it to reproduce the paper's results. `v2` runs the same pipeline over a larger corpus: the AACL papers plus publications cited in the IPCC AR6 Working Group I, II and III reports, papers reached by citation expansion, and the WCRP-CMIP Zotero library. The two versions were built separately, so node uuids are not shared between them. Match nodes across versions by name, or by the `id` property for papers (an OpenAlex id).

The `aacl-2026` tag freezes this repository as it was for the paper.

## Loading

Each version has a node table and an edge table:

```python
from datasets import load_dataset

nodes = load_dataset("aayushacharya/climatekg", "v2-nodes", split="train")
edges = load_dataset("aayushacharya/climatekg", "v2-edges", split="train")

# The paper's graph
nodes_v1 = load_dataset("aayushacharya/climatekg", "v1-nodes", split="train")
```

## Schema

Nodes:

| Column | Description |
| --- | --- |
| `uuid` | Node id, `<Label>_<uuid4>`, referenced by the edge table |
| `label` | Entity type, e.g. `Source` (a climate model), `Experiment`, `Variable`, `Paper`, `Method`, `Result`, `Country` |
| `name` | Display name (the title for papers) |
| `properties` | All other node properties as a JSON string. They depend on the label, e.g. `doi` and `publication_year` for papers, `geonameid` for places |

Edges:

| Column | Description |
| --- | --- |
| `source`, `target` | Node uuids |
| `source_label`, `target_label` | Labels of the two endpoints |
| `relation` | Relation type, e.g. `PAPER_USES_MODEL`, `USED_IN_EXPERIMENT`, `COVERS_REGION` |
| `properties` | Edge properties as a JSON string, e.g. `dataset_count` on ESGF metadata edges |

The allowed (source type, relation, target type) combinations are defined by the ontology in the GitHub repository (`kg_create/output/climatepub4kg.ontology.ttl`).

## Notes

- The `v1` edge table is the released `data/kg/edges_20260622` dump without its 70,829 edges to ESGF `Source` nodes that are missing from the metadata snapshot. The paper reports 755,308 relations, counted in Neo4j after loading this dump.
- `v2` is an export of the loaded Neo4j graph. Bookkeeping nodes from the RDF export (labels starting with `_`) are left out, and so are papers added by hand through the project website, which did not go through the extraction pipeline.
- The per-paper pipeline outputs of `v1` (every candidate triplet with its evidence span and the verdict of each verification stage) are in the GitHub repository under `data/pipeline_outputs/`. PDFs and parsed full texts are not redistributed.

## License

CC BY 4.0. The graph builds on third-party sources that keep their own terms: OpenAlex (CC0), ESGF search metadata (ESGF / CMIP6 terms of use), WCRP CMIP6_CVs (CC BY 4.0), GeoNames (CC BY 4.0) and NASA GCMD Keywords (public domain). Each extracted triplet's evidence is a short paragraph quoted from the source publication.

## Citation

```bibtex
@inproceedings{acharya2026climatekg,
  title     = {Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications},
  author    = {Acharya, Aayush and Zhang, Qi and Dragut, Eduard and Latecki, Longin Jan},
  booktitle = {Proceedings of AACL 2026},
  year      = {2026}
}
```
