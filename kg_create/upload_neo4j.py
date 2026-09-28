import argparse
import os
from neo4j import GraphDatabase
from tqdm import tqdm
import datetime
import concurrent.futures

parser = argparse.ArgumentParser(description="Wipe the Neo4j database and load neo4j/cypher/nodes_<date>.cypher and edges_<date>.cypher.")
parser.add_argument("--date", default=datetime.datetime.now().strftime("%Y%m%d"),
                    help="Suffix of the cypher files to load, e.g. 20260622 for the released 2,000-paper KG (default: today)")
current_date = parser.parse_args().date

URI = os.getenv("NEO4J_URI", "neo4j://localhost:8687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

####### DANGEROUS: This will delete all nodes and edges in the database
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # delete all
        session.run("MATCH (n) DETACH DELETE n")

        # delete all edges
        # session.run("MATCH ()-[r]->() DELETE r")

def run_query(driver, query):
    query = query.strip()
    if not query:
        return
    if query.endswith(";"):
        query = query[:-1].strip()
    try:
        with driver.session() as session:
            # .consume() forces the result to be pulled before the session closes. Without
            # it the auto-commit query is only queued, and an occasional statement was
            # discarded on session close -- silently, since errors surface on consumption.
            session.run(query).consume()
    except Exception as e:
        print(f"Error running query: {e}")


# Load nodes.cypher file and execute line by line.
# newline='\n' splits only on newline. With newline='' (or the default) Python also
# treats a bare \r as a line terminator, which would cut a single CREATE into fragments
# and produce syntax errors. neo4j_v2.py strips control characters from property values,
# so this is belt-and-braces.
with open(f'neo4j/cypher/nodes_{current_date}.cypher', 'r', newline='\n') as file:
    node_queries = file.readlines()

# Run node creation in parallel using ThreadPoolExecutor
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(
            executor.map(lambda query: run_query(driver, query), node_queries),
            total=len(node_queries),
            desc="Creating Nodes"
        ))

# Create a per-label uuid index for every label now present, so the labeled edge
# MATCHes (see neo4j_v2.py) load via index seeks (~minutes) instead of full scans (~13h).
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
        for lab in labels:
            session.run(f"CREATE INDEX uuid_{lab} IF NOT EXISTS FOR (n:`{lab}`) ON (n.uuid)")
        session.run("CALL db.awaitIndexes(600)")
    print(f"Created/verified uuid indexes for {len(labels)} labels")

# Load edges.cypher file (statements split by ';\n')
with open(f'neo4j/cypher/edges_{current_date}.cypher', 'r', newline='\n') as file:
    edge_queries = file.read()
    edge_queries = edge_queries.split(';\n')[:-1]  # Remove the last empty query

# Run in parallel using ThreadPoolExecutor
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(
            executor.map(lambda query: run_query(driver, query), edge_queries),
            total=len(edge_queries),
            desc="Creating Edges"
        ))

# Report final graph size
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = session.run("MATCH ()-[e]->() RETURN count(e) AS c").single()["c"]
        p = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
    print(f"DONE. graph now: {n:,} nodes, {r:,} relationships, {p:,} Paper nodes")
