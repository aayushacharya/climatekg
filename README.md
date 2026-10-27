# ClimateKG

Code and data for **Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications** (AACL 2026).

The pipeline builds a climate-science knowledge graph (KG) in two steps. It first initializes a trusted backbone from ESGF dataset metadata, CMIP6 controlled vocabularies, GCMD keywords and GeoNames. It then enriches that backbone with facts extracted from scientific publications by LLM agents. Each extracted triplet must conform to the ontology, be grounded in an evidence span of the paper, pass an LLM validator and link to canonical KG entities before it enters the graph.

This release covers the 2,000 publications used in the paper:

| | |
| --- | --- |
| KG nodes | 133,990 |
| KG relations | 755,308 |
| Candidate triplets extracted | 39,964 |
| Triplets kept after verification | 23,327 (98.96% judged correct) |

## Repository layout

```
extraction/            LLM extraction and verification pipeline (one script per stage)
  experiments/         scripts behind the paper's ablation, recall and baseline comparisons
kg_create/             KG backbone construction and Neo4j loading
  gcmd/ geolocation/   GCMD keyword and GeoNames inputs
  other_sources/       CMIP6 source_id controlled vocabulary
  output/              ontology (Turtle) and CMIP6 model components
data/                  released data for the 2,000 publications (compressed)
scripts/unpack_data.sh unpacks data/ into the directories the scripts use
```

## Data

| File | Contents |
| --- | --- |
| `data/kg/nodes_20260622.cypher.gz`, `edges_20260622.cypher.gz` | The KG as Cypher `CREATE` statements, ready for `kg_create/upload_neo4j.py` |
| `data/kg/nodes_20260622.json.gz`, `edges_20260622.json.gz` | The same KG as JSON node and edge lists |
| `data/pipeline_outputs/<stage>_<date>.tar.gz` | Per-paper JSON output of each pipeline stage (see below) |
| `data/pipeline_outputs/entity_linked_papers_2000.json.gz` | Entity-linking output, keyed by OpenAlex paper id: the input to the KG build |
| `data/papers/candidate_papers_<date>.json`, `paper_metadata_v2.json` | OpenAlex metadata of the candidate papers |

The corpus is split into two partitions, `2026-02-01` (1,007 papers) and `2026-04-24` (993 papers). The stage outputs are cumulative, and every triplet keeps the verdict of every stage even after an earlier stage rejected it:

| Stage | Directory | Flag added to each triplet |
| --- | --- | --- |
| Triplet and evidence extraction | `extracted_papers` | |
| Ontology conformance (O) | `ontology_conformed_papers` | `ontology_conformed` |
| Evidence validation (E) | `evidence_validated_papers` | `evidence_validated` |
| Validator agent (V) | `validated_papers` | `valid` |
| C_triplet judge (evaluation only) | `evaluated_papers` | `eval` |

The Cypher dump has 826,139 edge statements. 70,829 of them reference ESGF `Source` nodes that are not in the metadata snapshot, so Neo4j skips them on load, which leaves the 755k relations reported in the paper. PDFs and parsed full texts are not redistributed. The 1,000 human-annotated triplets used in the agreement study are also not part of this release.

The KG is also on Hugging Face as Parquet node and edge tables, in two versions: `v1` is the 2,000-publication graph of this release and `v2` is an expanded graph of 10,910 publications built with the same pipeline:

```python
from datasets import load_dataset
nodes = load_dataset("aayushacharya/climatekg", "v1-nodes", split="train")
edges = load_dataset("aayushacharya/climatekg", "v1-edges", split="train")
```

`scripts/build_hf_dataset.py` builds that release from `data/kg/` (v1) and a running Neo4j instance (v2). The dataset card is `huggingface/README.md`.

Unpack everything with

```bash
./scripts/unpack_data.sh
```

## Setup

```bash
conda create -n climatekg python=3.10
conda activate climatekg
pip install -r requirements.txt
cp extraction/.env.example extraction/.env   # add GOOGLE_API_KEY, OPENAI_API_KEY, HF_TOKEN, Neo4j credentials
```

Neo4j 5 is required to load the KG. The scripts default to `neo4j://localhost:8687`. Set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD`, or pass the matching `--neo4j-*` flags, to point them elsewhere. Run every script from its own directory, because paths are resolved relative to it.

## Load the KG into Neo4j

`upload_neo4j.py` **deletes everything in the target database** before loading, so point it at an empty database.

```bash
./scripts/unpack_data.sh
cd kg_create
python upload_neo4j.py --date 20260622
```

## Reproduce the paper's results

The stage-attrition and ablation tables need no API calls, because they are read from the stored stage flags:

```bash
cd extraction
python experiments/stage_ablation.py
```

| Result in the paper | Script |
| --- | --- |
| Per-stage attrition and stage ablation | `extraction/experiments/stage_ablation.py` |
| Per-partition pipeline statistics | `extraction/stats.py --date <date>` |
| Entity-level recall against CMIP Publication Hub | `extraction/experiments/extract_cmip_publication_moreinfo_entities.py`, `evaluate_cmip_moreinfo_entity_extraction.py`, `analyze_cmip_missing_entities_in_parsed_papers.py`, `create_cmip_recall_report.py` |
| Comparison with ClimateIE | `extraction/experiments/run_climateie_full_gemini_analysis.py` (with `extract_climateie_pdf_gemini.py`, `evaluate_climateie_triplet_judge.py`) |
| Comparison with KARMA | `extraction/experiments/karma/validate_1k_karma.py`, `analyze_1k_karma.py` |
| Evidence location and hallucination analysis | `extraction/experiments/evidence_section_analysis.py` |

The baseline comparisons need inputs that are not bundled here. The ClimateIE comparison needs the ClimateIE human corpus and its PDFs under `extraction/experiments/data/`, plus the ClimateIE system's predictions (see [ClimateIE](https://aclanthology.org/2025.climatenlp-1.6/)). The KARMA comparison needs the output of [KARMA](https://github.com/YuxingLu613/KARMA) run on the first 1,007 publications. The CMIP recall and evidence-location scripts need the parsed paper text, which you can regenerate with `parse_pdfs_olmocr.py`.

## Run the pipeline on new papers

Run each stage from `extraction/`. Each one reads the previous stage's `output/<stage>/<date>/` partition and skips papers that already have output, so an interrupted run can simply be restarted.

1. `find_papers.py` finds candidate papers through OpenAlex, screens them with Llama-3.3-70B (Ollama), and downloads the PDFs.
2. `parse_pdfs_olmocr.py` parses the PDFs to Markdown with [olmOCR](https://github.com/allenai/olmocr).
3. `extraction_agent.py` extracts triplets with evidence, plus document-level fields (Gemini-2.5-pro).
4. `ontology_conformance_agent.py` checks each (source type, relation, target type) against the ontology in `consts.py`.
5. `evidence_validation_agent.py` fuzzy-matches each evidence span against the parsed paper.
6. `validation_agent.py` is the LLM validator (Gemini-2.5-flash).
7. `evaluation_agent.py` is the C_triplet judge, used for evaluation only.
8. `create_embeddings.py` embeds KG entity names with NV-Embed-v2. Run it before step 9 whenever the KG has gained nodes.
9. `entity_linking.py` links triplet entities to KG nodes, with an LLM fallback.

Then build and load the KG from `kg_create/`:

```bash
python download_esgf.py --help     # ESGF search metadata into kg_create/data/ (about 15 GB, not bundled)
python neo4j_v2.py                 # writes neo4j/cypher/{nodes,edges}_<today>.cypher
python upload_neo4j.py             # wipes the database and loads today's files
```

`neo4j_v2.py` reads the ESGF snapshot as `data/esgf_noncmip6.json` plus the CMIP6 offset ranges `data/esgf_cmip6_<start>_<end>.json` listed at the top of the script (our snapshot was taken April to May 2025). `output/new_model_components.json` is bundled, and `extract_model_components.py` regenerates it from the CMIP6 CV.

`export_rdf.py` and `rdf_utils.py` export the loaded graph to RDF/Turtle. The ontology alone is in `kg_create/output/climatepub4kg.ontology.ttl`.

A few modules keep a `_deprecated` suffix from earlier versions of the code (`prompts_deprecated.py`, `utils_deprecated.py`). The current agents still import them, and they hold the prompts listed in the paper's appendix.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE.md`, which also lists the licenses of the third-party sources).

## Citation

```bibtex
@inproceedings{acharya2026climatekg,
  title     = {Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications},
  author    = {Acharya, Aayush and Zhang, Qi and Dragut, Eduard and Latecki, Longin Jan},
  booktitle = {Proceedings of AACL 2026},
  year      = {2026}
}
```
