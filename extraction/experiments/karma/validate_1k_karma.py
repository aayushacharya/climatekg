"""Judge KARMA's output triplets with the same C_triplet judge used for our pipeline.

KARMA (https://github.com/YuxingLu613/KARMA) is run separately on the first 1,007
publications with prompts adapted to the climate domain; this script reads its
`<paper>.intermediate.json` outputs and labels every integrated triple valid/invalid
with gpt-5-mini and GPT_VALIDATION_PROMPT (the evaluator prompt of evaluation_agent.py).
"""
import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

EXTRACTION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXTRACTION_DIR))
from prompts_deprecated import GPT_VALIDATION_PROMPT  # noqa: E402

MODEL = "gpt-5-mini-2025-08-07"

load_dotenv(EXTRACTION_DIR / ".env")
client = OpenAI()


def validate_triplets(source: str, relation: str, target: str, evidence: str) -> str:
    prompt = GPT_VALIDATION_PROMPT.format(source=source, relation=relation, target=target, paragraph=evidence)
    response = client.responses.create(model=MODEL, input=prompt)
    return response.output_text.strip()


def process_paper(paper_id: str, input_dir: str, output_dir: str) -> None:
    with open(os.path.join(input_dir, f"{paper_id}.intermediate.json"), "r") as f:
        paper_data = json.load(f)

    karma_data = []
    for triple in paper_data["integrated_triples"]:
        item = {
            "source_instance": triple["head"],
            "relation": triple["relation"],
            "target_instance": triple["tail"],
            "paragraph": paper_data["summaries"][0],
        }
        response = validate_triplets(item["source_instance"], item["relation"], item["target_instance"], item["paragraph"])
        item["gpt_validated"] = response == "valid"
        karma_data.append(item)

    with open(os.path.join(output_dir, f"{paper_id}.json"), "w") as f:
        json.dump({"model": MODEL, "karma_validated_triplets": karma_data}, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="KARMA output directory with <paper>.intermediate.json files")
    parser.add_argument("--output-dir", required=True, help="Where to write the judged triplets")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paper_ids = [name.split(".")[0] for name in os.listdir(args.input_dir) if "intermediate" in name]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(
            executor.map(lambda pid: process_paper(pid, args.input_dir, args.output_dir), paper_ids),
            total=len(paper_ids),
            desc="Processing KARMA Papers",
        ))


if __name__ == "__main__":
    main()
