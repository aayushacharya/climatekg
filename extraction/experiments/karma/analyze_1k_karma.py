"""Report C_triplet over the KARMA triplets judged by validate_1k_karma.py."""
import argparse
import json
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validated-dir", required=True, help="Output directory of validate_1k_karma.py")
args = parser.parse_args()

total_triples = 0
valid_triples = 0

paper_ids = [name.split(".")[0] for name in os.listdir(args.validated_dir) if name.endswith(".json")]
for paper_id in tqdm(paper_ids, desc="Analyzing KARMA Validated Papers"):
    with open(os.path.join(args.validated_dir, f"{paper_id}.json"), "r") as f:
        paper_data = json.load(f)

    for triplet in paper_data.get("karma_validated_triplets", []):
        total_triples += 1
        if triplet.get("gpt_validated", False):
            valid_triples += 1

print(f"Total Triples Processed: {total_triples}")
print(f"Total Valid Triples: {valid_triples}")
if total_triples > 0:
    print(f"Percentage of Valid Triples: {(valid_triples / total_triples) * 100:.2f}%")
