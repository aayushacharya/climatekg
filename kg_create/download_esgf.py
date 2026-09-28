"""
Download ESGF data as JSON from LLNL SOLR API.
Error happens. Just retry.
1589000/14117317
python download_esgf.py --output esgf_cmip6 --offset_start 2927000 --offset_end 5000000
python download_esgf.py --output esgf_cmip6 --offset_start 5487000 --offset_end 10000000
python download_esgf.py --output esgf_cmip6 --offset_start 10000000 --offset_end 14117317
"""

import requests
import json
import time
import argparse
from tqdm import tqdm
import random

BASE_URL = "https://aims2.llnl.gov/proxy/search"
# BASE_URL = "http://esgf-node.llnl.gov/esg-search/search"

# Default facets for CMIP6
CMIP6_FACETS = (
    "activity_id, data_node, source_id, institution_id, source_type, experiment_id, "
    "sub_experiment_id, nominal_resolution, variant_label, grid_label, table_id, "
    "frequency, realm, variable_id, cf_standard_name"
)

# Facets for non-CMIP6 (e.g., CORDEX, obs4MIPs, etc.)
NOT_CMIP6_FACETS = (
    "project, product, institute, model, data_node, source_id, experiment, experiment_family, "
    "time_frequency, realm, cmor_table, ensemble, variable, variable_long_name, "
    "cf_standard_name, driving_model, domain, rcm_name, rcm_version"
)


def fetch_all_documents_from_query_template(
    base_query_url: str,
    output_path: str,
    max_docs: int = None,
    batch_size: int = 500,
    offset_start: int = 0,
    offset_end: int = None,
):
    offset = offset_start
    all_docs = []
    print(f"📁 Output path: {output_path}")

    if offset_end is None:
        initial_url = base_query_url.replace("offset=0", f"offset={offset}").replace(
            "limit=10", f"limit={batch_size}"
        )
        print("📡 Fetching metadata...")
        response = requests.get(initial_url)
        response.raise_for_status()
        num_found = response.json()["response"]["numFound"]
        if max_docs:
            num_found = min(num_found, max_docs)
        print(f"📦 Total datasets available: {num_found:,}")
        offset_end = num_found

    pbar = tqdm(
        total=offset_end - offset_start,
        desc=f"Downloading from {offset_start:,} to {offset_end:,}",
    )
    retries = 0
    while offset < offset_end:
        paged_url = base_query_url.replace("offset=0", f"offset={offset}").replace(
            "limit=10", f"limit={batch_size}"
        )

        try:
            response = requests.get(paged_url)
            response.raise_for_status()
            docs = response.json()["response"]["docs"]
            all_docs.extend(docs)
            offset += batch_size
            pbar.update(batch_size)
            retries = 0
        except Exception as e:
            print(f"⚠️ Error at offset {offset}: {e}. ")
            retries += 1
            sleep_time = random.randint(10, 20)
            print(f"Sleeping {sleep_time}s and retrying for {retries}th times...")
            time.sleep(10)
            continue
        time.sleep(
            round(random.uniform(0.5, 2), 1)
        )  # polite delay from 0.5 to 2 seconds

        with open(output_path, "w") as f:
            json.dump(all_docs, f, indent=2)

    pbar.close()
    print(f"\n✅ Downloaded {len(all_docs):,} documents.")

    with open(output_path, "w") as f:
        json.dump(all_docs, f, indent=2)
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ESGF datasets as JSON from LLNL SOLR API."
    )
    parser.add_argument(
        "--project", type=str, default="CMIP6", help="Project name (default: CMIP6)"
    )
    parser.add_argument(
        "--not_project", action="store_true", help="Exclude this project from results"
    )
    parser.add_argument(
        "--output", type=str, default="datasets.json", help="Output JSON file"
    )
    parser.add_argument(
        "--max_docs", type=int, default=None, help="Maximum number of docs to download"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Documents per page"
    )

    parser.add_argument(
        "--offset_start", type=int, default=0, help="Offset to start from"
    )
    parser.add_argument("--offset_end", type=int, default=None, help="Offset to end at")

    args = parser.parse_args()

    if args.not_project:
        base_url = f"{BASE_URL}?project%21={args.project}&offset=0&limit=10&type=Dataset&format=application%2Fsolr%2Bjson&facets=project%2C+product%2C+institute%2C+model%2C+data_node%2C+source_id%2C+experiment%2C+experiment_family%2C+time_frequency%2C+realm%2C+cmor_table%2C+ensemble%2C+variable%2C+variable_long_name%2C+cf_standard_name%2C+driving_model%2C+domain%2C+rcm_name%2C+rcm_version&latest=true&query=*"
    else:
        base_url = f"{BASE_URL}?project={args.project}&offset=0&limit=10&type=Dataset&format=application%2Fsolr%2Bjson&facets=activity_id%2C+data_node%2C+source_id%2C+institution_id%2C+source_type%2C+experiment_id%2C+sub_experiment_id%2C+nominal_resolution%2C+variant_label%2C+grid_label%2C+table_id%2C+frequency%2C+realm%2C+variable_id%2C+cf_standard_name&latest=true&query=*"

    fetch_all_documents_from_query_template(
        base_query_url=base_url,
        output_path="data/"+args.output + f"_{args.offset_start}_{args.offset_end}.json",
        max_docs=args.max_docs,
        batch_size=args.batch_size,
        offset_start=args.offset_start,
        offset_end=args.offset_end,
    )
