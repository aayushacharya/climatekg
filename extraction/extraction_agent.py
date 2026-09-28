"""
Information Extraction for Scientific Papers (cost-optimized).

Differences vs extraction_agent_deprecated.py (the original):
  1. Model is configurable instead of a hardcoded gemini-2.5-pro. By default the
     structured KG/observation/entity extractions use gemini-2.5-pro (for
     quality) while the merged free-text metadata call uses the cheaper
     gemini-2.5-flash (--metadata-model). The model name(s) + paper_screening
     are kept on the heavier --model because they were the metadata fields most
     sensitive to a Flash downgrade. Override either with --model /
     --metadata-model.
  2. The seven plain-text sub-extractions are collapsed from seven calls into
     two: the five free-text fields (field, problem, method, task, innovation)
     in one structured call on --metadata-model, and model + paper_screening in
     one structured call on --model. Total per paper is ~5 calls (prose +
     model/screening + obs + entity + kg) versus ~10 originally. Each call
     re-reads the cached PDF, so fewer calls means proportionally fewer
     cached-token reads.
  3. Context caches are given a short TTL and are explicitly deleted after each
     paper, so a paper's PDF does not sit in cache storage (billed per
     token-hour) long after it is used.

The output JSON schema is identical to the original, so downstream stages
(ontology_conformance_agent.py, etc.) consume it unchanged.
"""

import argparse
import concurrent.futures
import datetime as dt
import io
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

from consts import ALL_RELATIONSHIP_SOURCE_TO_TARGET, GEMINI_MIN_CACHE_TOKEN_LIMIT
from models import (Entity_Extraction_Model, Kg_Extraction_Model,
                    Observation_Metric_Model)
from prompts_deprecated import (ENTITY_EXTRACTION_PROMPT, FIELD_EXTRACTION,
                                INNOVATION_EXTRACTION, KG_LINKING_PROMPT,
                                METHOD_EXTRACTION, MODEL_EXTRACTION,
                                OBSERVATION_AND_METRIC_EXTRACTION,
                                PAPER_SCREENING_PROMPT, PROBLEM_EXTRACTION,
                                TASK_EXTRACTION)
from utils_deprecated import (construct_openalex_metadata_url, prompt_llm,
                              validate_entity_extraction_response)

# Please add relevant keys from .env.example document
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

# Heavy structured extractions (KG/observation/entity) default to Pro for
# quality; the merged metadata call defaults to the cheaper Flash.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_METADATA_MODEL = "gemini-2.5-flash"
# Cache TTL is a safety net so caches expire on their own if the process dies
# before the explicit delete in process_paper runs.
CACHE_TTL = "600s"

SYSTEM_INSTRUCTION = (
    "You are an expert in extracting information from scientific papers. Your job "
    "is to answer the user's query based on the pdf file you have access to."
)

# All entity types (used to filter the entity-extraction response).
all_entity_types = set()
for source, target, _ in ALL_RELATIONSHIP_SOURCE_TO_TARGET:
    all_entity_types.add(source)
    all_entity_types.add(target)


class Paper_Prose_Model(BaseModel):
    """Merged schema for the free-text sub-extractions (run on metadata_model)."""
    field: str
    problem: str
    method: str
    task: str
    innovation: str


class Model_Screening_Model(BaseModel):
    """Climate model(s) + screening flag (run on the heavier extraction_model)."""
    model: list[str]
    paper_screening: str


# Single structured prompt for the free-text fields, replacing five separate
# text prompts. Wording is reused verbatim from prompts_deprecated.py.
COMBINED_METADATA_PROMPT = f"""You are extracting high-level metadata from a scientific paper related to climate modeling or Earth system science. Populate every field of the response schema using the instructions below.

field: {FIELD_EXTRACTION}

problem: {PROBLEM_EXTRACTION}

method: {METHOD_EXTRACTION}

task: {TASK_EXTRACTION}

innovation: {INNOVATION_EXTRACTION}
"""

# model + paper_screening are kept together (screening is gated by model) and run
# on the heavier extraction_model for accuracy.
MODEL_SCREENING_PROMPT = f"""You are analyzing a scientific paper related to climate modeling or Earth system science. Populate every field of the response schema using the instructions below.

model: {MODEL_EXTRACTION} Return the model name(s) as a list of strings; if nothing is proposed, return ["no"].

paper_screening: {PAPER_SCREENING_PROMPT}
"""


def today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def latest_pdf_date() -> str:
    pdf_root = SCRIPT_DIR / "papers" / "pdfs"
    if not pdf_root.exists():
        return today()
    date_dirs = sorted(path.name for path in pdf_root.iterdir() if path.is_dir())
    if not date_dirs:
        return today()
    return date_dirs[-1]


def default_paper_dir(date: str) -> Path:
    return SCRIPT_DIR / "papers" / "pdfs" / date


def default_metadata_path(date: str) -> Path:
    return SCRIPT_DIR / "papers" / "metadata" / f"candidate_papers_{date}.json"


def default_output_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "extracted_papers" / date


def load_paper_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {"papers": {}}

    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "papers" in data:
        return data
    return {"papers": data}


def process_paper(
    each_paper_id: str,
    paper_dir: Path,
    output_dir: Path,
    paper_metadata: dict,
    metadata_model: str,
    extraction_model: str,
):
    client = genai.Client()
    created_caches: list = []
    uploaded_file = None  # the PDF we upload to the Gemini File API (must be deleted)
    # One cache per model (a Gemini context cache is tied to a single model), so
    # when metadata_model == extraction_model only a single cache is created.
    cache_by_model: dict = {}

    def get_cache(model_name: str, document, token_count: int):
        if model_name in cache_by_model:
            return cache_by_model[model_name]
        cache = None
        if token_count > GEMINI_MIN_CACHE_TOKEN_LIMIT:
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    contents=[document],
                    ttl=CACHE_TTL,
                ),
            )
            created_caches.append(cache)
        cache_by_model[model_name] = cache
        return cache

    try:
        pdf_path = paper_dir / f"{each_paper_id}.pdf"
        with pdf_path.open("rb") as f:
            doc_io = io.BytesIO(f.read())

        metadata = paper_metadata["papers"].get(
            construct_openalex_metadata_url(each_paper_id), {}
        )

        document = client.files.upload(
            file=doc_io,
            config=dict(mime_type="application/pdf"),
        )
        uploaded_file = document

        token_count = client.models.count_tokens(
            model=metadata_model,
            contents=[SYSTEM_INSTRUCTION, document],
        ).total_tokens

        # ── Merged free-text metadata call (replaces 5 separate text calls) ─
        metadata_cache = get_cache(metadata_model, document, token_count)
        prose_config = types.GenerateContentConfig(
            cached_content=metadata_cache.name if metadata_cache is not None else None,
            response_mime_type="application/json",
            response_schema=Paper_Prose_Model,
        )
        prose: Paper_Prose_Model = prompt_llm(
            client,
            COMBINED_METADATA_PROMPT,
            metadata_cache,
            document,
            model_name=metadata_model,
            config=prose_config,
        )
        if prose is None:
            raise ValueError("metadata extraction returned no parsed response")

        field = prose.field
        problem = prose.problem
        method = prose.method
        task = prose.task
        innovation = prose.innovation

        # ── model + paper_screening on the heavier extraction_model ─────────
        extraction_cache = get_cache(extraction_model, document, token_count)
        model_screening_config = types.GenerateContentConfig(
            cached_content=extraction_cache.name if extraction_cache is not None else None,
            response_mime_type="application/json",
            response_schema=Model_Screening_Model,
        )
        model_screening: Model_Screening_Model = prompt_llm(
            client,
            MODEL_SCREENING_PROMPT,
            extraction_cache,
            document,
            model_name=extraction_model,
            config=model_screening_config,
        )
        if model_screening is None:
            raise ValueError("model/screening extraction returned no parsed response")

        model = model_screening.model

        # Preserve the original screening gate: if no climate model is proposed,
        # the paper cannot have model-produced/evaluated results.
        if (len(model) == 1 and model[0].lower() == "no") or (len(model) == 0):
            paper_screening = "no"
        else:
            paper_screening = (model_screening.paper_screening or "no").strip()

        # ── Observations and metrics extraction ───────────────────────────
        obs_and_metrics_resp = None
        if paper_screening.lower() != "no":
            obs_and_metrics_config = types.GenerateContentConfig(
                cached_content=extraction_cache.name if extraction_cache is not None else None,
                response_schema=list[Observation_Metric_Model],
                response_mime_type="application/json",
            )
            obs_and_metrics_resp: list[Observation_Metric_Model] = prompt_llm(
                client, OBSERVATION_AND_METRIC_EXTRACTION, extraction_cache,
                document, model_name=extraction_model, config=obs_and_metrics_config,
            )

        # ── Entity extraction (for the "mentions" relation) ────────────────
        entity_extraction_config = types.GenerateContentConfig(
            cached_content=extraction_cache.name if extraction_cache is not None else None,
            response_schema=list[Entity_Extraction_Model],
            response_mime_type="application/json",
        )
        entity_extraction_resp: list[Entity_Extraction_Model] = prompt_llm(
            client, ENTITY_EXTRACTION_PROMPT, extraction_cache, document,
            model_name=extraction_model, config=entity_extraction_config,
        )
        entities_extracted = validate_entity_extraction_response(
            entity_extraction_resp, all_entity_types
        )

        # ── KG extraction ──────────────────────────────────────────────────
        kg_extraction_config = types.GenerateContentConfig(
            cached_content=extraction_cache.name if extraction_cache is not None else None,
            response_schema=list[Kg_Extraction_Model],
            response_mime_type="application/json",
        )
        kg_extraction_resp: list[Kg_Extraction_Model] = prompt_llm(
            client, KG_LINKING_PROMPT, extraction_cache, document,
            model_name=extraction_model, config=kg_extraction_config,
        )

        venue = {
            "name": metadata.get("venue_name", None),
            "id": metadata.get("venue_id", None),
            "type": metadata.get("venue_type", None),
        }

        output_dict = {
            "uri": metadata.get("id", None),
            "title": metadata.get("title", None),
            "venue": venue,
            "publication_year": metadata.get("publication_year", None),
            "authors": metadata.get("authors", []),
            "doi": metadata.get("doi", None),
            "abstract": metadata.get("abstract", None),
            "field": field,
            "problem": problem,
            "method": method,
            "task": task,
            "model": model,
            "paper_screening": paper_screening,
            "keywords": metadata.get("keywords", []),
            "concepts": metadata.get("concepts", []),
            "obs_and_metrics": [m.model_dump() for m in obs_and_metrics_resp] if obs_and_metrics_resp is not None else [],
            "innovation": innovation,
            "kg_extraction_triplets": [t.model_dump() for t in kg_extraction_resp] if kg_extraction_resp is not None else [],
            "entities_extracted": [e.model_dump() for e in entities_extracted] if entities_extracted is not None else [],
            "referenced_works": metadata.get("referenced_works", []),
        }

        output_path = output_dir / f"{each_paper_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
    except Exception as e:
        print(f"Error processing paper {each_paper_id}: {str(e)}")
    finally:
        # Don't let a paper's PDF sit in cache storage after we're done with it.
        for cache in created_caches:
            try:
                client.caches.delete(name=cache.name)
            except Exception as exc:
                print(f"  [cache cleanup] failed to delete {cache.name}: {exc}")
        # Delete the uploaded PDF from the Gemini File API too. Uploads persist
        # ~48h and count against a project-wide 20GB file_storage_bytes quota;
        # leaving them makes large runs fail with 429 RESOURCE_EXHAUSTED once the
        # cap fills. (Deleting caches alone does NOT free the underlying file.)
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as exc:
                print(f"  [file cleanup] failed to delete {uploaded_file.name}: {exc}")


def parse_args() -> argparse.Namespace:
    default_date = latest_pdf_date()
    parser = argparse.ArgumentParser(
        description="Extract paper information from PDFs (cost-optimized v2)."
    )
    parser.add_argument(
        "--date",
        default=default_date,
        help="PDF date partition to process. Defaults to the latest papers/pdfs/<date> directory.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        help="Directory containing PDFs. Defaults to papers/pdfs/<date>.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata JSON path. Defaults to papers/metadata/candidate_papers_<date>.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to output/extracted_papers/<date>.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model for the structured KG/observation/entity extractions. "
            f"Defaults to {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--metadata-model",
        default=DEFAULT_METADATA_MODEL,
        help=(
            "Model for the merged metadata call. Defaults to "
            f"{DEFAULT_METADATA_MODEL}. Set this equal to --model to use one "
            "model (and a single cache) for everything."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of concurrent papers to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir or default_paper_dir(args.date)
    metadata_path = args.metadata or default_metadata_path(args.date)
    output_dir = args.output_dir or default_output_dir(args.date)
    extraction_model = args.model
    metadata_model = args.metadata_model

    if not paper_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {paper_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_metadata = load_paper_metadata(metadata_path)
    paper_ids = sorted(paper.stem for paper in paper_dir.glob("*.pdf"))

    processed_ids = {output_file.stem for output_file in output_dir.glob("*.json")}
    paper_ids = [paper_id for paper_id in paper_ids if paper_id not in processed_ids]

    print(f"Metadata model: {metadata_model} | Extraction model: {extraction_model}")
    print(f"Papers to process: {len(paper_ids)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = executor.map(
            lambda paper_id: process_paper(
                paper_id, paper_dir, output_dir, paper_metadata,
                metadata_model, extraction_model,
            ),
            paper_ids,
        )
        list(tqdm(futures, total=len(paper_ids), desc="Processing Papers"))


if __name__ == "__main__":
    main()
