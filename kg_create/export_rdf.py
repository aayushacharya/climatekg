import os
import datetime as dt
from neo4j import GraphDatabase, basic_auth
import requests
import base64
from rdf_utils import add_custom_prefix_ontology, export_unified_ttl

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:8687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

ONTO_NS = os.getenv("ONTO_NS", "https://climatepub4kg.github.io/ontology#")
ID_NS   = os.getenv("ID_NS",   "https://climatepub4kg.github.io/id/")

AUTH_STRING = f"{NEO4J_USER}:{NEO4J_PASSWORD}"
ENCODED_AUTH = base64.b64encode(AUTH_STRING.encode("utf-8")).decode("utf-8")


PREFIXES = [
    ("clm", ONTO_NS),
    ("clmid",   ID_NS),
]

def get_node_labels():
    driver= GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        result = session.run("CALL db.labels() YIELD label WHERE NOT label STARTS WITH '_' RETURN label;")
        return [record["label"] for record in result]

def get_rel_types():
    driver= GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        result = session.run("CALL db.relationshipTypes()")
        return [record["relationshipType"] for record in result]

# Get node labels from neo4j query
NODE_LABELS= get_node_labels()
REL_TYPES  = get_rel_types()

# ESGF-side node types: their `name` is the ESGF/CMIP controlled-vocabulary term, so it
# is a better instance identifier than an opaque uuid (clmid:Source/CESM2 rather than
# clmid:Source/Source_<uuid>). Publication nodes (Paper, Author, Venue, Keyword) and
# GeoNames nodes keep uuid-based IRIs — they carry their own external identifiers in
# `uri` / `geonameid` instead.
NAME_AS_ID = [
    "Activity", "Domain", "Ensemble", "Experiment", "ExperimentFamily", "Forcing",
    "Frequency", "GridLabel", "Institute", "MIPEra", "Member", "Metric", "Model",
    "PhysicalFeature", "PhysicalScheme", "Project", "RCM", "Realm", "Resolution",
    "SimulationType", "Source", "SourceComponent", "SourceType", "SubExperiment",
    "Variable",
]

def run(session, cypher: str, params: dict = None):
    return session.run(cypher, params or {}).consume()

def init_graphconfig(session):
    try:
        run(session, "CALL n10s.graphconfig.drop()")
    except Exception:
        pass
    run(session, """
        CALL n10s.graphconfig.init({
            handleVocabUris: 'MAP',
            handleMultival: 'ARRAY',
            keepLangTag: true,
            keepCustomDataTypes: true
        })
    """)

def remove_all_prefixes(session):
    run(session, "CALL n10s.nsprefixes.removeAll()")

def add_prefixes(session):
    for p, ns in PREFIXES:
        run(session, "CALL n10s.nsprefixes.add($p, $ns)", {"p": p, "ns": ns})

def ensure_uri_constraints(session, label: str):
    cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.uri IS UNIQUE"
    run(session, cypher)

def preserve_source_uri(session, label: str):
    """Keep the identifier the node was built from before mint_uris overwrites n.uri.

    Author/Keyword/Venue/Paper arrive from neo4j_v2.py with n.uri holding their OpenAlex
    URL, which is also the natural key build_retained_uuid_map() uses to reuse existing
    uuids on a rebuild. Minting the canonical IRI into the same property destroyed that
    key, so every rebuild after an export re-minted ~47k uuids (and with them the
    name-based IRI suffixes this module assumes are stable). Copy it aside first.
    """
    run(session, f'''
    MATCH (n:`{label}`)
    WHERE n.uri IS NOT NULL AND n.source_uri IS NULL AND NOT n.uri STARTS WITH $base
    SET n.source_uri = n.uri
    ''', {"base": ID_NS})


def mint_uris(session, label: str):
    # Every node gets a uri, whether or not it already had one (values are overwritten,
    # which is why preserve_source_uri must run first).
    if label not in NAME_AS_ID:
        run(session, f'''
        MATCH (n:`{label}`)
        SET n.uri = $base + "{label}/" + n.uuid
        ''', {"base": ID_NS})
        return

    # Name-based IRIs. Two guards, because n.uri carries a uniqueness constraint and a
    # clashing or null slug would abort the whole export:
    #   - nodes without a usable name fall back to their uuid;
    #   - names that are not unique within the label (a handful in PhysicalScheme,
    #     Metric, SourceComponent, …) get the node's uuid appended. The suffix is
    #     stable across rebuilds because uuids are retained (see neo4j_v2.py).
    run(session, f'''
    MATCH (n:`{label}`)
    WHERE n.name IS NULL OR trim(toString(n.name)) = ""
    SET n.uri = $base + "{label}/" + n.uuid
    ''', {"base": ID_NS})

    run(session, f'''
    MATCH (n:`{label}`)
    WHERE n.name IS NOT NULL AND trim(toString(n.name)) <> ""
    WITH apoc.text.urlencode(toString(n.name)) AS slug, collect(n) AS nodes
    WITH slug, nodes, size(nodes) AS shared
    UNWIND nodes AS n
    SET n.uri = $base + "{label}/" + slug +
        CASE WHEN shared = 1 THEN "" ELSE "-" + last(split(n.uuid, "_")) END
    ''', {"base": ID_NS})

def add_mapping_for_label(session, label: str):
    iri = f"{ONTO_NS}{label}"
    run(session, "CALL n10s.mapping.add($iri, $local)", {"iri": iri, "local": label})

def add_mapping_for_rel(session, rel: str):
    iri = f"{ONTO_NS}{rel}"
    run(session, "CALL n10s.mapping.add($iri, $local)", {"iri": iri, "local": rel})

    

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        init_graphconfig(session)
        remove_all_prefixes(session)
        add_prefixes(session)
        for label in NODE_LABELS:
            label = label.strip()
            ensure_uri_constraints(session, label)
            preserve_source_uri(session, label)
            mint_uris(session, label)
            add_mapping_for_label(session, label)
        for rel in REL_TYPES:
            add_mapping_for_rel(session, rel.strip())

    # call http api to get ttl export
    HTTP_URI = NEO4J_URI.replace("neo4j://", "http://").replace("bolt://", "http://").replace(":8687", ":8474")
    # Headers
    headers = {
        "Authorization": f"Basic {ENCODED_AUTH}",
        "Content-Type": "application/json"
    }

    data = {
    "cypher": "MATCH (n)-[r]-(m) RETURN n,r,m",
    "format": "Turtle"
    }

    response = requests.post(f"{HTTP_URI}/rdf/neo4j/cypher", auth=(NEO4J_USER, NEO4J_PASSWORD), headers=headers, json=data)
    response.raise_for_status()
    ttl_data = response.text
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join("output", current_date)
    os.makedirs(out_dir, exist_ok=True)
    instance_filename="climatepub4kg.instances.ttl"
    instance_file_path=os.path.join(out_dir,instance_filename)
    with open(instance_file_path, "w", encoding="utf-8") as f:
        f.write(ttl_data)
    
    # call http api to get ontology export
    response = requests.get(f"{HTTP_URI}/rdf/neo4j/onto?format=Turtle", auth=(NEO4J_USER, NEO4J_PASSWORD), headers={"Accept": "text/turtle"})
    response.raise_for_status()
    onto_data = response.text
    ont_filename="climatepub4kg.ontology.ttl"
    ont_file_path=os.path.join(out_dir,ont_filename)
    with open(ont_file_path, "w", encoding="utf-8") as f:
        f.write(onto_data)

    unified_ttl_path=os.path.join(out_dir,"climatepub4kg.ttl")
    # version=current_date stamps owl:versionInfo / dcterms:issued on the ontology header
    export_unified_ttl(ont_file_path, instance_file_path, unified_ttl_path, version=current_date)


    driver.close()
    print(f"Successfully exported RDF data. Refer to {out_dir}/ directory.")
