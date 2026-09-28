"""
Script to create Neo4j database files from ESGF metadata, Geonames location data and CMIP6 CVs.
After completion the files created are stored in data/ and neo4j/ directories. To upload the files
to a neo4j database, run the `upload_neo4j.py` script.
"""


import json
import argparse
import re
import unicodedata
from collections import defaultdict, Counter
from tqdm import tqdm
import uuid
import datetime
import pandas as pd
import os
from geolocation import get_intersecting_geonameids
os.makedirs('data', exist_ok=True)
os.makedirs('neo4j/cypher', exist_ok=True)
os.makedirs('neo4j/json', exist_ok=True)


parser = argparse.ArgumentParser(
    description="Build Neo4j node/edge cypher from ESGF + geo + entity-linked papers.")
parser.add_argument("--neo4j-uri", default="neo4j://localhost:8687")
parser.add_argument("--neo4j-user", default="neo4j")
parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
parser.add_argument("--neo4j-database", default="neo4j")
parser.add_argument("--no-retain-uuids", action="store_true",
                    help="Mint all-new uuids instead of reusing existing-node uuids "
                         "from the live graph (default: retain).")
parser.add_argument("--citation-source",
                    default="../extraction/output/entity_linked_papers/v2_processed_papers_entity_linking_post_llm.json",
                    help="Entity-linked JSON providing per-paper referenced_works, "
                         "used to build Paper-Paper CITED_BY edges.")
args = parser.parse_args()

# Label -> node property that was used as the natural key when the uuid was minted.
# get_entity_id lower-cases its key; geo nodes are keyed by geonameid. Any label not
# listed here is keyed by "name". Keep in sync with the minting sites (get_entity_id,
# add_geonodes_from_dict) below.
RETAIN_KEY_PROP = {
    "Paper": "id",
    "Author": "uri",
    "Keyword": "uri",
    "Venue": "uri",   # falls back to name for uri-less venues (see below)
    "Continent": "geonameid",
    "Country": "geonameid",
    "City": "geonameid",
    "No_Country_Region": "geonameid",
    "Country_Subdivision": "geonameid",
}

# natural_key(lower) -> existing uuid, per label. Populated from the live graph so a
# full rebuild reuses prior uuids (stable references) instead of minting new ones.
RETAINED_UUID = defaultdict(dict)


def build_retained_uuid_map(uri, auth, database):
    """Map each live node's natural key -> its existing uuid, so get_entity_id and
    geo minting can reuse it on a full rebuild. Returns an empty map (all-new) if the
    graph is unreachable or empty (e.g. the first-ever build)."""
    from neo4j import GraphDatabase
    retained = defaultdict(dict)
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
    except Exception as e:
        print(f"[retain] Neo4j unreachable ({e}); minting all-new uuids.")
        return retained
    try:
        with driver.session(database=database) as s:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
            total = 0
            for lab in labels:
                if lab == "Venue":
                    # mirror effective_venueid in populate_paper_nodes: uri, else name
                    # (source_uri first, see the coalesce note below)
                    for rec in s.run("MATCH (n:Venue) RETURN coalesce(n.source_uri, n.uri) AS uri, "
                                     "n.name AS name, n.uuid AS u"):
                        key = rec["uri"] if rec["uri"] else rec["name"]
                        if key and rec["u"]:
                            retained[lab].setdefault(str(key).lower(), rec["u"])
                            total += 1
                    continue
                prop = RETAIN_KEY_PROP.get(lab, "name")
                # export_rdf.py overwrites n.uri with the canonical ClimatePub4KG IRI and
                # stashes the original identifier in n.source_uri, so prefer that when the
                # natural key is the uri; otherwise retention breaks after every export.
                key_expr = "coalesce(n.source_uri, n.uri)" if prop == "uri" else f"n.`{prop}`"
                q = (f"MATCH (n:`{lab}`) WHERE {key_expr} IS NOT NULL AND n.uuid IS NOT NULL "
                     f"RETURN {key_expr} AS k, n.uuid AS u")
                for rec in s.run(q):
                    retained[lab].setdefault(str(rec["k"]).lower(), rec["u"])
                    total += 1
        print(f"[retain] loaded {total:,} existing uuids across {len(labels)} labels from {uri}")
    finally:
        driver.close()
    return retained



with open('data/esgf_noncmip6.json', 'r') as f:
    datasets_n = json.load(f)
print(f"Number of non-CMIP6 datasets: {len(datasets_n):,}")


with open('data/esgf_cmip6_0_1589000.json', 'r') as f:
    datasets_c0 = json.load(f)
with open('data/esgf_cmip6_1589000_2927000.json', 'r') as f:
    datasets_c1 = json.load(f)
with open('data/esgf_cmip6_2927000_3421000.json', 'r') as f:
    datasets_c2 = json.load(f)
with open('data/esgf_cmip6_5000000_5487000.json', 'r') as f:
    datasets_c3 = json.load(f)
# print(f"Number of CMIP6 datasets: {len(datasets_c0):,} {len(datasets_c1):,} {len(datasets_c2):,} {len(datasets_c3):,} ")
# print(f"Total number of CMIP6 datasets: {len(datasets_c0) + len(datasets_c1) + len(datasets_c2) + len(datasets_c3):,}")



combined = datasets_c0 + datasets_c1 + datasets_c2 + datasets_c3+datasets_n 
# combined = datasets_n[:100000]
# combined= datasets_n[90000:170000]+datasets_c0[:4000]+datasets_c1[:4000]+datasets_c2[:4000]+datasets_c3[:4000]
print(f"Total number of datasets: {len(combined):,}")



# load geolocation lookup
with open('geolocation/geonameid_lookup.json', 'r') as f:
    geolocation_lookup = json.load(f)

continents_df=pd.read_csv('geolocation/continents.csv',keep_default_na=False)

with open('geolocation/domain_to_geonameid_lookup.json', 'r') as f:
    domain_to_geoname_ids = json.load(f)


# CMIP model metadata
with open('output/new_model_components.json', 'r') as f:
    cmip6_model_metadata = json.load(f)

NATURAL_HAZARD_TYPE_SEED=["Astronomical Low Tide","Avalanche","Blizzard","Coastal Flood","Cold/Wind Chill","Debris Flow","Dense Fog","Dense Smoke","Drought","Dust Devil","Dust Storm","Excessive Heat","Extreme Cold/Wind Chill","Flash Flood","Flood","Freezing Fog","Frost/Freeze","Funnel Cloud","Hail","Heat","Heavy Rain","Heavy Snow","High Surf","High Wind","Hurricane (Typhoon)","Ice Storm","Lake-Effect Snow","Lakeshore Flood","Lightning C","Marine Hail","Marine High Wind","Marine Strong Wind","Marine Thunderstorm Wind","Rip Current","Seiche","Sleet","Storm Surge/Tide","Strong Wind","Thunderstorm Wind","Tornado","Tropical Depression","Tropical Storm","Tsunami","Volcanic Ash","Waterspout","Wildfire","Winter Storm","Winter Weather"]
CLIMATE_ZONE_SEED=["Tropical", "Dry", "Temperate", "Continental", "Polar"]

GELOCATION_ENTITY_LABELS=["Continent","Country","City","No_Country_Region","Country_Subdivision"]

KEY_TO_ATTRIBUTES = {
    "Model": ["model_cohort"],
    "Experiment": [ "experiment_title"],
    "Variable": ["variable_long_name", "variable_units", "cf_standard_name"],
    "RCM": ['rcm_version']
}

# Define main entity attributes (these will become nodes)
ENTITY_KEY_TO_LABEL = {
    # two keys for the same label
    "experiment_id": "Experiment",
    "experiment": "Experiment",
    "member_id": "Member",
    "variant_label": "Member",
    "institution_id": "Institute",
    "institute": "Institute",
    "variable_id": "Variable",
    "variable": "Variable",
    "source_id": "Source",
    "driving_model": "Source",
    
    # 1 key for each label
    "activity_id": "Activity",
    "experiment_family": "ExperimentFamily",
    "forcing": "Forcing",
    "frequency": "Frequency",
    "grid_label": "GridLabel",
    "mip_era": "MIPEra",
    "nominal_resolution": "Resolution",
    "project": "Project",
    "realm": "Realm",
    "source_type": "SourceType",
    "sub_experiment_id": "SubExperiment",
    "ensemble": "Ensemble",
    # "driving_model": "DrivingModel",
    "domain": "Domain",
    "rcm_name": "RCM",
}

RELATIONSHIP_SOURCE_TO_TARGET = [('ExperimentFamily', 'Experiment', 'INCLUDES_EXPERIMENT'),
                                 ('Experiment', 'SubExperiment', 'HAS_SUB_EXPERIMENT'),
                                 ('Source', 'SourceType', 'IS_OF_TYPE'),
                                 ('Experiment', 'Realm', 'APPLIES_TO_REALM'),
                                 ('Activity', 'Realm', 'FOCUSES_ON_REALM'),
                                #  ('Project', 'Realm', 'HAS_REALM'),
                                #  ('Project', 'Institute', 'HAS_INSTITUTE'),
                                 ('Institute', 'Project', 'PARTICIPATED_IN'),
                                 ('Source', 'Activity', 'GENERATED_BY_ACTIVITY'),
                                 ('Source', 'Institute', 'PRODUCED_BY_INSTITUTE'),
                                 ('Source', 'MIPEra', 'BELONGS_TO_MIP_ERA'),
                                 ('Source', 'Experiment', 'USED_IN_EXPERIMENT'),
                                 ('Source', 'Variable', 'PRODUCES_VARIABLE'),
                                 #('Source', 'GridLabel', 'HAS_GRID_LABEL'), #exclude because trivial
                                 ('Source', 'Project', 'PART_OF_PROJECT'),
                                 ('Source', 'Member', 'ASSOCIATED_WITH_MEMBER'),
                                 ('Source', 'Frequency', 'SAMPLED_AT_FREQUENCY'),
                                 ('Source', 'Resolution', 'HAS_SPATIAL_RESOLUTION'),
                                 ('Source', 'Ensemble', 'PART_OF_ENSEMBLE'),
                                 ('Source', 'Realm', 'APPLIES_TO_REALM'),
                                 ('Experiment', 'Forcing', 'USES_FORCING'), 
                                 ('Experiment', 'Ensemble', 'INCLUDES_ENSEMBLE_MEMBER'),
                                #('Project', 'Variable', 'HAS_VARIABLE'), #exclude to prevent confusion
                                 ('RCM','Source','DRIVEN_BY_SOURCE'),   
                                #  ('RCM', 'DrivingModel', 'HAS_DRIVING_MODEL'),
                                 ('RCM', 'Domain', 'COVERS_DOMAIN'),
                                 ('RCM', 'Variable', 'PRODUCES_VARIABLE'),
                                # New relations added from LLM
                                 ('Project', 'Domain', 'COVERS_DOMAIN'),
                                 ('RCM','Project','PART_OF_PROJECT'),
                                 ('RCM', 'Experiment', 'USED_IN_EXPERIMENT'),
                                 ('RCM', 'Resolution', 'HAS_SPATIAL_RESOLUTION'),
                                 ('Activity','Project','PART_OF_PROJECT'),
                                 ('SubExperiment','Ensemble','PART_OF_ENSEMBLE'),
                                 ('ExperimentFamily','MIPEra','BELONGS_TO_MIP_ERA'),
                                 ('Experiment', 'Project', 'PART_OF_PROJECT'),
                                 ('Experiment', 'Resolution', 'HAS_SPATIAL_RESOLUTION'),
                                 ('Experiment', 'Institute', 'PERFORMED_BY_INSTITUTE'),
                                 ('Variable','Resolution','HAS_SPATIAL_RESOLUTION'),
                                 ]

# Geographic nodes are keyed by geonameid rather than by name, so they are absent from
# ENTITY_ID_TO_UUID and cannot be looked up with get_entity_id(). These two maps resolve a
# geographic name to every node carrying it -- one name can belong to several nodes
# (e.g. Florida is both a US state and a Uruguayan department), and to several labels.
NAME_TO_GEONODES_MAPPER={}
# Same, keyed by a punctuation/accent-insensitive form of the name, used as a fallback:
# extracted names regularly lose apostrophes ("Hawkes Bay Region" vs "Hawke's Bay Region").
NORMALISED_NAME_TO_GEONODES_MAPPER={}


def normalise_geo_name(name):
    """Casefolded, accent- and punctuation-free form of a geographic name."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    return re.sub(r"[^a-z0-9]", "", decomposed.lower())


def resolve_entity_uuids(entity_name, entity_label, nodes, restrict_to_label=True):
    """Return every node uuid that `entity_name` refers to.

    Non-geographic labels are keyed by name and resolve to at most one uuid via
    get_entity_id(). Geographic labels are keyed by geonameid, so they are resolved
    through the name maps instead, and may legitimately match more than one node.

    `restrict_to_label` keeps only geographic nodes carrying `entity_label`. Relations
    need that, because the relation is only meaningful for the endpoint type it was
    extracted for. PAPER_MENTIONS deliberately passes False so that a mention still
    reaches every same-named place regardless of which geographic label it was linked
    to (its historical behaviour).
    """
    if entity_label not in GELOCATION_ENTITY_LABELS:
        entity_uuid = get_entity_id(entity_name, entity_label)
        return [entity_uuid] if entity_uuid in nodes[entity_label] else []

    candidates = NAME_TO_GEONODES_MAPPER.get(entity_name)
    if not candidates:
        candidates = NORMALISED_NAME_TO_GEONODES_MAPPER.get(normalise_geo_name(entity_name), [])
    if not restrict_to_label:
        return list(candidates)
    # a name map spans all geographic labels, so keep only nodes of the wanted label
    return [u for u in candidates if u in nodes[entity_label]]

# RELATIONSHIP_SOURCE_TO_TARGET = [('ExperimentFamily', 'Experiment', 'HAS_EXPERIMENT_MEMBER'),
#                                  ('Experiment', 'SubExperiment', 'HAS_SUB_EXPERIMENT'),
#                                  ('Source', 'SourceType', 'HAS_SOURCE_TYPE'),
#                                  ('Experiment', 'Realm', 'HAS_REALM'),
#                                  ('Activity', 'Realm', 'HAS_REALM'),
#                                  ('Project', 'Realm', 'HAS_REALM'),
#                                  ('Project', 'Institute', 'HAS_INSTITUTE'),
#                                  ('Source', 'Activity', 'HAS_ACTIVITY'),
#                                  ('Source', 'Institute', 'HAS_INSTITUTE'),
#                                  ('Source', 'MIPEra', 'HAS_MIP_ERA'),
#                                  ('Source', 'Experiment', 'HAS_EXPERIMENT'),
#                                  ('Source', 'Variable', 'HAS_VARIABLE'),
#                                  #('Source', 'GridLabel', 'HAS_GRID_LABEL'), #exclude because trivial
#                                  ('Source', 'Project', 'HAS_PROJECT'),
#                                  ('Source', 'Member', 'HAS_MEMBER'),
#                                  ('Source', 'Frequency', 'HAS_FREQUENCY'),
#                                  ('Source', 'Resolution', 'HAS_RESOLUTION'),
#                                  ('Source', 'Ensemble', 'HAS_ENSEMBLE'),
#                                  ('Experiment', 'Forcing', 'HAS_FORCING'), 
#                                  ('Experiment', 'Ensemble', 'HAS_ENSEMBLE'),
#                                 #('Project', 'Variable', 'HAS_VARIABLE'), #exclude to prevent confusion
#                                  ('Source','RCM','HAS_RCM'),   
#                                 #  ('RCM', 'DrivingModel', 'HAS_DRIVING_MODEL'),
#                                  ('RCM', 'Domain', 'HAS_DOMAIN'),
#                                  ]



len(set([i[2] for i in RELATIONSHIP_SOURCE_TO_TARGET]))



def populate_geographical_data(region_entity_mapper, geolocation_lookup,continents_df,domain_to_geoname_ids,entity_label, entity_uuid, record):
    if entity_label != "RCM" or ("rcm_name" not in record and "domain" not in record):
        return

    # required_keys = ["north_degrees", "south_degrees", "east_degrees", "west_degrees"]
    # if not all(k in record and record[k] is not None for k in required_keys):
    #     return

    # bbox = {
    #     "north": record["north_degrees"],
    #     "south": record["south_degrees"],
    #     "east": record["east_degrees"],
    #     "west": record["west_degrees"]
    # }

    # geonameids = get_intersecting_geonameids(bbox,geolocation_lookup,continents_df)
    geonameids=set()
    for each_domain in record["domain"]:
        if each_domain in domain_to_geoname_ids:
            geonameids.update(domain_to_geoname_ids[each_domain])

    if entity_uuid not in region_entity_mapper:
        region_entity_mapper[entity_uuid] = {"geonameids": list(geonameids)}
    else:
        existing = set(region_entity_mapper[entity_uuid]["geonameids"])
        updated = existing.union(geonameids)
        region_entity_mapper[entity_uuid]["geonameids"] = list(updated)




def add_geonodes_from_dict(nodes,entity_label,region_dict,geonameid_to_uuid_mapper):
    for geonameid, data in region_dict.items():
        geonameid = str(geonameid)
        if geonameid not in nodes[entity_label]:
            tmp_uuid = RETAINED_UUID.get(entity_label, {}).get(geonameid)
            if tmp_uuid is None:
                tmp_uuid = entity_label + '_' + str(uuid.uuid4())
            nodes[entity_label][tmp_uuid]=data
            nodes[entity_label][tmp_uuid]['geonameid']=geonameid
            nodes[entity_label][tmp_uuid]['uuid']=tmp_uuid
            geonameid_to_uuid_mapper[geonameid]=tmp_uuid
            if data['name'] not in NAME_TO_GEONODES_MAPPER:
                NAME_TO_GEONODES_MAPPER[data['name']]=[tmp_uuid]
            else:
                NAME_TO_GEONODES_MAPPER[data['name']].append(tmp_uuid)
            normalised_name = normalise_geo_name(data['name'])
            NORMALISED_NAME_TO_GEONODES_MAPPER.setdefault(normalised_name, []).append(tmp_uuid)



def populate_geographical_nodes(nodes,geonameid_to_uuid_mapper):
    continents_df= pd.read_csv("geolocation/continents.csv",keep_default_na=False)
    countries_df= pd.read_csv("geolocation/countries.csv",keep_default_na=False)
    # water_df= pd.read_csv("geolocation/water.csv")
    cities_df= pd.read_csv("geolocation/cities.csv")
    # no_country_df= pd.read_csv("geolocation/no-country.csv")
    no_country_df= pd.read_csv("geolocation/no-country-water-merged.csv")
    csd_df=pd.read_csv("geolocation/country_subdivision.csv")
    
    # Country does not have a name column so we use the country column as the name
    countries_df['name'] = countries_df['country']


    continents_dict=continents_df.set_index('geonameid').to_dict(orient='index')
    countries_dict=countries_df.set_index('geonameid').to_dict(orient='index')
    # water_dict=water_df.set_index('geonameid').to_dict(orient='index')
    cities_dict=cities_df.set_index('geonameid').to_dict(orient='index')
    no_country_dict=no_country_df.set_index('geonameid').to_dict(orient='index')
    csd_dict=csd_df.set_index('geonameid').to_dict(orient='index')

    add_geonodes_from_dict(nodes,"Continent",continents_dict,geonameid_to_uuid_mapper)
    add_geonodes_from_dict(nodes,"Country",countries_dict,geonameid_to_uuid_mapper)
    # add_geonodes_from_dict(nodes,"Water_Bodies",water_dict,geonameid_to_uuid_mapper)
    add_geonodes_from_dict(nodes,"City",cities_dict,geonameid_to_uuid_mapper)
    add_geonodes_from_dict(nodes,"No_Country_Region",no_country_dict,geonameid_to_uuid_mapper)
    add_geonodes_from_dict(nodes,"Country_Subdivision",csd_dict,geonameid_to_uuid_mapper)

# Observation triplets carry a free-text `model` string. Only ESGF/CMIP source models are
# Source nodes (Source is not in ENTITY_TYPES_TO_ADD_INSTANCES, i.e. papers never mint one),
# so a name that is not in the controlled vocabulary has no node to attach to. Resolving it
# with get_entity_id() minted a uuid that matched nothing, and the MODEL_* edge statements
# written against it silently no-opped at load time. Resolve against existing Sources
# instead, and count what cannot be resolved so the loss is reported rather than hidden.
_SOURCE_NORMALISED_NAMES = {}
UNRESOLVED_OBS_MODELS = Counter()


def resolve_existing_source(model_name, nodes):
    """uuid of the Source node this model string names, or None if there is no such node."""
    if not model_name:
        return None
    if not _SOURCE_NORMALISED_NAMES:
        for uid, data in nodes["Source"].items():
            name = data.get("name")
            if name:
                _SOURCE_NORMALISED_NAMES.setdefault(normalise_geo_name(name), uid)
    candidate = ENTITY_ID_TO_UUID.get("Source", {}).get(str(model_name).lower())
    if candidate is None:
        candidate = _SOURCE_NORMALISED_NAMES.get(normalise_geo_name(model_name))
    # only accept a uuid that belongs to a Source node that actually exists
    return candidate if candidate in nodes["Source"] else None


def add_paper_nodes_from_dict(nodes,entity_label,paper_id,paper_data,key):
    if key not in paper_data:
        return None
    if paper_data[key] is None or paper_data[key] == "":
        return None
    node_uuid = get_entity_id(paper_data[key], entity_label)
    nodes[entity_label][node_uuid] = {
        "uuid": node_uuid,
        "paper_id": paper_id,
        "name": paper_data.get(key, ""),
    }
    return node_uuid

def populate_climate_zone_nodes(nodes):
    for each_climate_zone in CLIMATE_ZONE_SEED:
        node_uuid = get_entity_id(each_climate_zone, "ClimateZone")
        nodes["ClimateZone"][node_uuid] = {
            "uuid": node_uuid,
            "name": each_climate_zone,
        }

def populate_natural_hazard_type_nodes(nodes):
    for each_natural_hazard_type in NATURAL_HAZARD_TYPE_SEED:
        node_uuid = get_entity_id(each_natural_hazard_type, "NaturalHazardType")
        nodes["NaturalHazardType"][node_uuid] = {
            "uuid": node_uuid,
            "name": each_natural_hazard_type,
        }

GCMD_SEED_FILES = {
    "Instrument": "gcmd/instruments.json",
    "Platform": "gcmd/platforms.json",
    "Ocean_Circulation": "gcmd/ocean_circulations.json",
    "Teleconnection": "gcmd/teleconnections.json",
}


def iter_gcmd_leaf_concepts(concept_data):
    """
    Yield only the narrowest GCMD concepts.

    Some GCMD exports are search-result pages with a top-level "concepts" list,
    while others are a broader concept tree with "narrower" children. For trees,
    only leaf children are emitted so broad parent concepts are not seeded.
    """
    if isinstance(concept_data, list):
        for each_item in concept_data:
            yield from iter_gcmd_leaf_concepts(each_item)
        return

    if not isinstance(concept_data, dict):
        return

    if isinstance(concept_data.get("concepts"), list):
        for each_item in concept_data["concepts"]:
            yield from iter_gcmd_leaf_concepts(each_item)
        return

    narrower = concept_data.get("narrower")
    if isinstance(narrower, list) and len(narrower) > 0:
        for each_item in narrower:
            yield from iter_gcmd_leaf_concepts(each_item)
        return

    if concept_data.get("uuid") and concept_data.get("prefLabel"):
        yield concept_data


def append_unique_property(node, key, value):
    if value is None or value == "":
        return
    if key not in node:
        node[key] = value
        return
    existing_value = node[key]
    if isinstance(existing_value, list):
        if value not in existing_value:
            existing_value.append(value)
        return
    if existing_value != value:
        node[key] = [existing_value, value]


def seed_gcmd_nodes(nodes):
    """
    Seed GCMD Instruments, Platforms, Ocean_Circulation, and Teleconnection nodes.

    GCMD prefLabel is mapped to the graph "name" property. The GCMD source uuid
    is stored as "gcmd_uuid", and "gcmd_uri" points at the GCMD KeywordViewer.
    """
    for entity_label, relative_path in GCMD_SEED_FILES.items():
        with open(relative_path, "r") as f:
            gcmd_data = json.load(f)

        for concept in iter_gcmd_leaf_concepts(gcmd_data):
            concept_name = concept["prefLabel"]
            gcmd_uuid = concept["uuid"]
            node_uuid = get_entity_id(concept_name, entity_label)
            node = nodes[entity_label].setdefault(
                node_uuid,
                {
                    "uuid": node_uuid,
                    "name": concept_name,
                },
            )
            node["name"] = node.get("name", concept_name)
            append_unique_property(node, "gcmd_uuid", gcmd_uuid)
            append_unique_property(
                node,
                "gcmd_uri",
                f"https://gcmd.earthdata.nasa.gov/KeywordViewer/scheme/all/{gcmd_uuid}",
            )

def populate_paper_nodes(nodes):
    """
    Populate paper nodes from the processed papers.
    """
    with open("../extraction/output/entity_linked_papers/v2_processed_papers_entity_linking_post_llm.json", "r") as f:
        processed_papers_entity_linked = json.load(f)

    paper_id_to_uuid_mapper ={}
    paper_nodes_uuids={}
    for paper_id, paper_data in processed_papers_entity_linked.items():
        paper_uuid = get_entity_id(paper_id, "Paper")
        nodes["Paper"][paper_uuid] = {
            "uuid": paper_uuid,
            "id": paper_id,
            "uri": paper_data.get("uri", ""),
            "title": paper_data.get("title", ""),
            "name": paper_data.get("title", ""),
            "abstract": paper_data.get("abstract", ""),
            "doi": paper_data.get("doi", ""),
            "publication_year": paper_data.get("publication_year", ""),
        }
        paper_id_to_uuid_mapper[paper_id] = paper_uuid
        paper_nodes_uuids[paper_id] = {}
        
        field_uuid=add_paper_nodes_from_dict(nodes,"Field",paper_id,paper_data,"field")
        problem_uuid=add_paper_nodes_from_dict(nodes,"Problem",paper_id,paper_data,"problem")
        method_uuid=add_paper_nodes_from_dict(nodes,"Method",paper_id,paper_data,"method")
        task_uuid=add_paper_nodes_from_dict(nodes,"Task",paper_id,paper_data,"task")
        innovation_uuid=add_paper_nodes_from_dict(nodes,"Innovation",paper_id,paper_data,"innovation")
        if "venue" in paper_data and paper_data["venue"] is not None and "id" in paper_data["venue"]:
            effective_venueid=paper_data["venue"]["id"]
            if paper_data["venue"]["id"] is None:
                if "name" in paper_data["venue"] and paper_data["venue"]["name"] is not None:
                    effective_venueid=paper_data["venue"]["name"]
                else:
                    effective_venueid="No Venue"
            venue_uuid=get_entity_id(effective_venueid,"Venue")
            nodes["Venue"][venue_uuid]={
            "uuid": venue_uuid,
            "uri": paper_data["venue"]["id"],
            "name": "No Venue" if effective_venueid == "No Venue" else paper_data["venue"]["name"],
            "type": paper_data["venue"].get("type",""),
            }
            paper_nodes_uuids[paper_id]["Venue"]=venue_uuid 

        paper_nodes_uuids[paper_id]["Field"]=field_uuid
        paper_nodes_uuids[paper_id]["Problem"]=problem_uuid
        paper_nodes_uuids[paper_id]["Method"]=method_uuid
        paper_nodes_uuids[paper_id]["Task"]=task_uuid
        paper_nodes_uuids[paper_id]["Innovation"]=innovation_uuid

        model_data=paper_data["model"]
        paper_nodes_uuids[paper_id]["Source"]=[]
        if model_data is not None and len(model_data) > 0:
            for each_model in model_data:
                model_uuid = get_entity_id(each_model, "Source")
                paper_nodes_uuids[paper_id]["Source"].append(model_uuid)
        paper_nodes_uuids[paper_id]["Keyword"]=[]
        for each_keyword in paper_data.get("keywords", []):
            if each_keyword is None or "id" not in each_keyword or each_keyword["id"] is None:
                continue
            keyword_uuid = get_entity_id(each_keyword["id"], "Keyword")
            nodes["Keyword"][keyword_uuid] = {
                "uuid": keyword_uuid,
                "uri": each_keyword["id"],
                "name": each_keyword["display_name"],
            }
            paper_nodes_uuids[paper_id]["Keyword"].append(keyword_uuid)

        # Add check for duplicates
        paper_nodes_uuids[paper_id]["Author"]=[]
        for each_author in paper_data.get("authors", []):
            if "id" not in each_author or each_author["id"] is None:
                continue
            author_uuid = get_entity_id(each_author["id"], "Author")
            nodes["Author"][author_uuid] = {
                "uuid": author_uuid,
                "uri": each_author["id"],
                "name": each_author["display_name"],
            }
            paper_nodes_uuids[paper_id]["Author"].append(author_uuid)
        
        # Check for duplicates
        paper_nodes_uuids[paper_id]["ResultTriplet"]=[]
        if paper_data["obs_and_metrics"] is not None:
            for each_result_triplet in paper_data["obs_and_metrics"]:
                # None when the model is not in the Source controlled vocabulary; the
                # MODEL_* edges are then skipped instead of being written against a
                # uuid that no node carries (add_edges_source_to_target ignores None).
                obs_model_uuid = resolve_existing_source(each_result_triplet["model"], nodes)
                if obs_model_uuid is None and each_result_triplet["model"]:
                    UNRESOLVED_OBS_MODELS[str(each_result_triplet["model"]).strip()] += 1
                observational_dataset_uuid = get_entity_id(each_result_triplet["observational_dataset"], "ObservationalDataset")
                nodes["ObservationalDataset"][observational_dataset_uuid] = {
                    "uuid": observational_dataset_uuid,
                    "name": each_result_triplet["observational_dataset"],
                }

                metric_uuid = get_entity_id(each_result_triplet["evaluation_metric"], "EvaluationMetric")
                nodes["EvaluationMetric"][metric_uuid] = {
                    "uuid": metric_uuid,
                    "name": each_result_triplet["evaluation_metric"],
                }

                result_uuid = get_entity_id(each_result_triplet["result"], "Result")
                nodes["Result"][result_uuid] = {
                    "uuid": result_uuid,
                    "name": each_result_triplet["result"],
                }
                paper_nodes_uuids[paper_id]["ResultTriplet"].append((obs_model_uuid,observational_dataset_uuid, metric_uuid, result_uuid))

        # Check for duplicates.
        for entity_label,entity_instances in paper_data.get("create_entities", {}).items():
            for each_instance in entity_instances:
                entity_uuid = get_entity_id(each_instance, entity_label)
                if entity_uuid in nodes[entity_label]:
                    if "added_from_paper_id" in nodes[entity_label][entity_uuid]:
                        nodes[entity_label][entity_uuid]["added_from_paper_id"].append(paper_uuid)
                    else:
                        nodes[entity_label][entity_uuid]["added_from_paper_id"]=[paper_uuid]
                else:
                    nodes[entity_label][entity_uuid] = {
                        "uuid": entity_uuid,
                        "name": each_instance,
                        "added_from_paper_id": [paper_uuid],
                    }

    return paper_id_to_uuid_mapper,paper_nodes_uuids

def add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids,edge_label,entity_label):
    if entity_label in paper_nodes_uuids[paper_id] and paper_nodes_uuids[paper_id][entity_label] is not None:
            entity_uuid = paper_nodes_uuids[paper_id][entity_label]
            if entity_uuid not in edges[paper_uuid]:
                edges[paper_uuid][entity_uuid]={}
            if edge_label not in edges[paper_uuid][entity_uuid]:
                edges[paper_uuid][entity_uuid][edge_label]={
                    "added_from_paper_id": [paper_uuid],
                }
            else:
                edges[paper_uuid][entity_uuid][edge_label]["added_from_paper_id"].append(paper_uuid)


# Anything that would terminate a line or is not printable. OCR'd papers regularly leave
# stray CR / FF / BS bytes in extracted prose (degree signs and math markup in particular).
# upload_neo4j.py reads the nodes file line by line and Python treats a bare CR as a line
# terminator, so an unstripped CR splits one CREATE across several "lines", each an
# invalid statement with an unterminated string.
_CYPHER_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


def _cypher_value(v):
    """The value as Neo4j will store it: control characters and quotes removed.

    This is the form that comes back when the graph is read again, so uuid retention
    must key on it (see get_entity_id).
    """
    s = _CYPHER_CONTROL_CHARS.sub(" ", str(v))
    return s.replace('"', '').replace("'", '')


def _clean_cypher_str(v):
    """Make a value safe to embed in a double-quoted Cypher string literal."""
    # A backslash starts an escape sequence in Cypher, and text such as "^\circ C" or the
    # "\d{4}" in GeoNames postal_code_regex is not a valid one, so escape it. Cypher undoes
    # this on load, so the stored value is _cypher_value(v).
    return _cypher_value(v).replace("\\", "\\\\")


def format_cypher_property(k, v):
    """Render one property as a Cypher `key: value` fragment, or None to skip.

    List/set values become native Cypher arrays; everything else is a string.
    """
    if v is None:
        return None
    if isinstance(v, (list, set)):
        elems = ", ".join(f'"{_clean_cypher_str(i)}"' for i in list(v))
        return f'{k}: [{elems}]'
    return f'{k}: "{_clean_cypher_str(v)}"'


def add_edges_source_to_target(edges, source_uuid, target_uuid, edge_label,props={}):
    """
    Add an edge from source_uuid to target_uuid with the given edge_label.
    If the edge already exists, increment the count.
    """
    if source_uuid is not None and target_uuid is not None:
        if source_uuid not in edges:
            edges[source_uuid] = {}
        if target_uuid not in edges[source_uuid]:
            edges[source_uuid][target_uuid] = {}
        if edge_label not in edges[source_uuid][target_uuid]:
            edges[source_uuid][target_uuid][edge_label] = {}
        for k,v in props.items():
            edges[source_uuid][target_uuid][edge_label][k]=v



def populate_paper_edges(edges,paper_id_to_uuid_mapper, paper_nodes_uuids):
    """
    Populate edges between paper nodes.
    """
    for paper_id in tqdm(paper_id_to_uuid_mapper.keys(), desc="Populating paper edges"):
        paper_uuid = paper_id_to_uuid_mapper[paper_id]
        
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids,'PAPER_BELONGS_TO_FIELD','Field')
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids,'PAPER_SOLVES_PROBLEM','Problem')
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids,'PAPER_APPLIES_METHOD','Method')
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids, 'PAPER_HAS_INNOVATION','Innovation')
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids, 'PAPER_WORKS_ON_TASK','Task')
        add_paper_edges_from_dict(edges,paper_id,paper_uuid,paper_nodes_uuids, 'PAPER_PUBLISHED_IN_VENUE','Venue')

        for each_model_uuid in paper_nodes_uuids[paper_id].get("Source", []):
            add_edges_source_to_target(edges, paper_uuid, each_model_uuid, 'PAPER_USES_MODEL')
        
        # model_uuid= paper_nodes_uuids[paper_id].get("Model", None)
        task_uuid= paper_nodes_uuids[paper_id].get("Task", None)
        method_uuid=paper_nodes_uuids[paper_id].get("Method", None)
        problem_uuid=paper_nodes_uuids[paper_id].get("Problem", None)
        innovation_uuid=paper_nodes_uuids[paper_id].get("Innovation", None)

        add_edges_source_to_target(edges, method_uuid, problem_uuid, 'METHOD_SOLVES_PROBLEM')
        add_edges_source_to_target(edges, method_uuid, task_uuid, 'METHOD_WORKS_ON_TASK')
        add_edges_source_to_target(edges, method_uuid, innovation_uuid, 'METHOD_HAS_INNOVATION')
        # add_edges_source_to_target(edges, method_uuid, model_uuid, 'METHOD_USES_MODEL')

        # add_edges_source_to_target(edges, model_uuid, task_uuid, 'MODEL_WORKS_FOR_TASK')
        # add_edges_source_to_target(edges, model_uuid, problem_uuid, 'MODEL_SOLVES_PROBLEM')
        add_edges_source_to_target(edges, task_uuid, problem_uuid, 'TASK_FACES_PROBLEM')


        # Add edges to Authors
        for author_uuid in paper_nodes_uuids[paper_id].get("Author", []):
            add_edges_source_to_target(edges, paper_uuid, author_uuid, 'PAPER_AUTHORED_BY')
        
        # Add edges to Keywords
        for keyword_uuid in paper_nodes_uuids[paper_id].get("Keyword", []):
            add_edges_source_to_target(edges, paper_uuid, keyword_uuid, 'PAPER_HAS_KEYWORD')
        
        for model_uuid, observational_dataset_uuid, metric_uuid, result_uuid in paper_nodes_uuids[paper_id].get("ResultTriplet", []):
            add_edges_source_to_target(edges, paper_uuid, observational_dataset_uuid, 'PAPER_EXPERIMENTS_ON_OBSERVATIONAL_DATASET')
            add_edges_source_to_target(edges, paper_uuid, metric_uuid, 'PAPER_USES_METRIC')
            add_edges_source_to_target(edges, paper_uuid, result_uuid, 'PAPER_HAS_RESULT')

            add_edges_source_to_target(edges, model_uuid, observational_dataset_uuid, 'MODEL_EXPERIMENTS_ON_OBSERVATIONAL_DATASET')
            add_edges_source_to_target(edges, model_uuid, metric_uuid, 'MODEL_USES_METRIC')
            add_edges_source_to_target(edges, model_uuid, result_uuid, 'MODEL_HAS_RESULT')
            add_edges_source_to_target(edges, result_uuid, metric_uuid, 'RESULT_HAS_METRIC')

            add_edges_source_to_target(edges, task_uuid, observational_dataset_uuid, 'TASK_EXPERIMENTS_ON_OBSERVATIONAL_DATASET')
            add_edges_source_to_target(edges, task_uuid, metric_uuid, 'TASK_USES_METRIC')

            add_edges_source_to_target(edges, method_uuid, observational_dataset_uuid, 'METHOD_EXPERIMENTS_ON_OBSERVATIONAL_DATASET')
            add_edges_source_to_target(edges, method_uuid, metric_uuid, 'METHOD_USES_METRIC')
            add_edges_source_to_target(edges, method_uuid, result_uuid, 'METHOD_HAS_RESULT')

    with open("../extraction/output/entity_linked_papers/v2_processed_papers_entity_linking_post_llm.json", "r") as f:
        processed_papers_entity_linked = json.load(f)
    
    # Create relations between entities
    for paper_id, paper_data in tqdm(processed_papers_entity_linked.items(), desc="Populating entity relations"):
        paper_uuid = paper_id_to_uuid_mapper[paper_id]

        for edge_label, edge_data in paper_data.get("create_relations",{}).items():
            for relation_info in edge_data:
                source_entity = relation_info["source_entity"]
                source_entity_label = relation_info["source_entity_type"]
                target_entity = relation_info["target_entity"]
                target_entity_label = relation_info["target_entity_type"]
                supporting_paragraph = relation_info.get("supporting_paragraph")
                # Either endpoint may be geographic, in which case one name can resolve to
                # several nodes (see resolve_entity_uuids) and the relation is recorded
                # against each of them.
                source_uuids = resolve_entity_uuids(source_entity, source_entity_label, nodes)
                target_uuids = resolve_entity_uuids(target_entity, target_entity_label, nodes)

                for source_uuid in source_uuids:
                    for target_uuid in target_uuids:
                        if source_uuid not in edges:
                            edges[source_uuid] = {}
                        if target_uuid not in edges[source_uuid]:
                            edges[source_uuid][target_uuid]={}
                        edge_props = edges[source_uuid][target_uuid].setdefault(edge_label, {})
                        edge_props.setdefault("added_from_paper_id", []).append(paper_uuid)
                        # Each triplet carries a supporting paragraph; one edge can be
                        # backed by multiple triplets (across paragraphs/papers), so
                        # accumulate the evidences into a deduped list.
                        if supporting_paragraph:
                            evidences = edge_props.setdefault("supporting_paragraphs", [])
                            if supporting_paragraph not in evidences:
                                evidences.append(supporting_paragraph)

        # Mentions logic
        for entity_label, entity_instances in paper_data.get("mentions",{}).items():
            for entity_instance in entity_instances:
                # For geolocation the same name may have multiple ids, e.g. Florida is in
                # the US as well as Uruguay, so every match gets a PAPER_MENTIONS edge.
                for entity_uuid in resolve_entity_uuids(
                    entity_instance, entity_label, nodes, restrict_to_label=False
                ):
                    add_edges_source_to_target(edges, paper_uuid, entity_uuid, 'PAPER_MENTIONS')

                
    
        


def _normalize_openalex_id(ref):
    """OpenAlex referenced_works are full URLs (https://openalex.org/W123...), while
    Paper node ids (paper_id_to_uuid_mapper keys) are the bare work id (W123...).
    Return the bare id so the two can be matched."""
    if not ref:
        return None
    return str(ref).rstrip("/").rsplit("/", 1)[-1]


def populate_citation_edges(edges, paper_id_to_uuid_mapper, source_file):
    """Add (cited)-[:CITED_BY]->(citing) edges from each paper's referenced_works.

    For a paper A that references paper B (B in A's referenced_works), B is cited by
    A, so the edge is (B)-[:CITED_BY]->(A). An edge is created only when BOTH A and B
    are Paper nodes already in the graph; referenced works absent from the KG (a large
    share, since referenced_works spans all of OpenAlex) are skipped.
    """
    try:
        with open(source_file, "r") as f:
            citation_papers = json.load(f)
    except FileNotFoundError:
        print(f"[CITED_BY] source not found: {source_file}; skipping citation edges")
        return

    created = 0
    skipped_not_in_kg = 0
    citing_with_refs = 0
    for citing_id, paper_data in tqdm(citation_papers.items(), desc="Populating CITED_BY edges"):
        # The citing paper must itself be a Paper node in the current graph.
        citing_uuid = paper_id_to_uuid_mapper.get(citing_id)
        if citing_uuid is None:
            continue
        referenced_works = paper_data.get("referenced_works") or []
        if referenced_works:
            citing_with_refs += 1
        for ref in referenced_works:
            cited_id = _normalize_openalex_id(ref)
            cited_uuid = paper_id_to_uuid_mapper.get(cited_id)
            if cited_uuid is None:
                skipped_not_in_kg += 1
                continue
            add_edges_source_to_target(edges, cited_uuid, citing_uuid, 'CITED_BY')
            created += 1
    print(f"[CITED_BY] citing papers with refs: {citing_with_refs:,} | "
          f"edges created: {created:,} | refs skipped (not Paper nodes in KG): {skipped_not_in_kg:,}")


def populate_geographical_edges(edges,region_entity_mapper,geonameid_to_uuid_mapper):
    # Edges from geographical regions to geographical regions
    continents_df= pd.read_csv("geolocation/continents.csv",keep_default_na=False)
    countries_df= pd.read_csv("geolocation/countries.csv",keep_default_na=False)
    cities_df= pd.read_csv("geolocation/cities.csv")
    csd_df = pd.read_csv("geolocation/country_subdivision.csv")

    # for each city, get the country_code. Find the country code in iso column of countries_df
    # and get the geonameid of the country. Create an edge from city to country
    for city_geonameid, data in cities_df.set_index('geonameid').to_dict(orient='index').items():
        city_geonameid=str(city_geonameid)
        if city_geonameid in geonameid_to_uuid_mapper:
            city_uuid=geonameid_to_uuid_mapper[str(city_geonameid)]
            country_code=data['country_code']
            match_values=countries_df[countries_df['iso'] == country_code]['geonameid'].values
            if len(match_values) > 0:
                country_geonameid=str(match_values[0])
                country_uuid=geonameid_to_uuid_mapper[country_geonameid]
                edges[city_uuid][country_uuid]=defaultdict(dict)
                edges[city_uuid][country_uuid]['IN_COUNTRY']={}
    
    # for each Country_Subdivision, get the country_code. Find the country code in iso column of countries_df
    # and get the geonameid of the country. Create an edge from city to country
    for csd_geonameid, data in csd_df.set_index('geonameid').to_dict(orient='index').items():
        csd_geonameid=str(csd_geonameid)
        if csd_geonameid in geonameid_to_uuid_mapper:
            csd_uuid=geonameid_to_uuid_mapper[str(csd_geonameid)]
            country_code=data['code'].split(".")[0]
            match_values=countries_df[countries_df['iso'] == country_code]['geonameid'].values
            if len(match_values) > 0:
                country_geonameid=str(match_values[0])
                country_uuid=geonameid_to_uuid_mapper[country_geonameid]
                edges[csd_uuid][country_uuid]=defaultdict(dict)
                edges[csd_uuid][country_uuid]['IN_COUNTRY']={}

    # for each country, get the continent. Find the continent code in iso column of continents_df
    # and get the geonameid of the continent. Create an edge from country to continent
    for country_geonameid, data in countries_df.set_index('geonameid').to_dict(orient='index').items():
        country_geonameid=str(country_geonameid)
        if country_geonameid in geonameid_to_uuid_mapper:
            country_uuid=geonameid_to_uuid_mapper[str(country_geonameid)]
            continent_code=data['continent']
            match_values=continents_df[continents_df['iso'] == continent_code]['geonameid'].values
            if len(match_values)>0:
                continent_geonameid=str(match_values[0])
                continent_uuid=geonameid_to_uuid_mapper[continent_geonameid]
                edges[country_uuid][continent_uuid]=defaultdict(dict)
                edges[country_uuid][continent_uuid]['IN_CONTINENT']={}
    
    # Edges from RCM to geographical regions
    for region_entity, data in region_entity_mapper.items():
        if "geonameids" in data:
            for geonameid in data["geonameids"]:
                geonameid = str(geonameid)
                location_uuid=geonameid_to_uuid_mapper[geonameid]
                edges[region_entity][location_uuid]=defaultdict(dict)
                edges[region_entity][location_uuid]['COVERS_REGION']={}



def model_match(name1,dict1):
    """
    Match a model name with a dictionary of model names.
    Returns True if match else False.
    """
    for key in dict1:
        if name1 == key:
            return True
    return False

def create_realm_to_uuid_dict(nodes, realm_to_uuid):
    """
    Create a dictionary mapping realm names to their UUIDs.
    """
    for k, v in nodes["Realm"].items():
        realm_to_uuid[v['name']]=k
    return realm_to_uuid

def create_or_get_model_component(component_name,model_component_to_uuid,nodes):
    """
    Create or get a model component node.
    Returns the UUID of the model component.
    """
    if component_name not in model_component_to_uuid:
        # tmp_uuid = "SourceComponent_" + str(uuid.uuid4())
        tmp_uuid = get_entity_id(component_name, "SourceComponent")
        model_component_to_uuid[component_name] = tmp_uuid
        nodes["SourceComponent"][tmp_uuid] = {"uuid": tmp_uuid}
        nodes["SourceComponent"][tmp_uuid]["name"] = component_name
        return tmp_uuid
    return model_component_to_uuid[component_name]

def populate_model_component_data(nodes,edges,cmip6_model_metadata):
    """
    Populate model component nodes from CMIP6 model metadata.
    """
    model_metadata_info=cmip6_model_metadata
    entity_label="Source"
    realm_to_uuid = {}
    model_component_to_uuid={}
    create_realm_to_uuid_dict(nodes, realm_to_uuid)
    
    model_realm_model_component={}
    for k,v in nodes[entity_label].items():
        if model_match(v["name"],model_metadata_info):
            model_metadata=model_metadata_info[v["name"]]
            associated_realm={}
            for realm_key,component_info in model_metadata.items():
                if realm_key not in realm_to_uuid:
                    continue
                for each_component_info in component_info:
                    model_component_uuid=create_or_get_model_component(each_component_info,model_component_to_uuid,nodes)
                    realm_uuid_key=realm_to_uuid[realm_key]
                    if realm_uuid_key in associated_realm:
                        associated_realm[realm_uuid_key].append(model_component_uuid)
                    else:
                        associated_realm[realm_uuid_key]=[model_component_uuid]
            if len(associated_realm) > 0:
                model_realm_model_component[k]=associated_realm

    for source_uuid, realm_component_dict in model_realm_model_component.items():
        for realm_uuid, component_uuids in realm_component_dict.items():
            for component_uuid in component_uuids:
                edges[source_uuid][component_uuid]=defaultdict(dict)
                if 'HAS_SOURCE_COMPONENT' not in edges[source_uuid][component_uuid]:
                    edges[source_uuid][component_uuid]['HAS_SOURCE_COMPONENT'] = {}
                edges[source_uuid][realm_uuid]=defaultdict(dict)
                if 'APPLIES_TO_REALM' not in edges[source_uuid][realm_uuid]:
                    edges[source_uuid][realm_uuid]['APPLIES_TO_REALM'] = {}
                edges[component_uuid][realm_uuid]=defaultdict(dict)
                if 'APPLIES_TO_REALM' not in edges[component_uuid][realm_uuid]:
                    edges[component_uuid][realm_uuid]['APPLIES_TO_REALM'] = {}

# create a uuid for each entity
ENTITY_ID_TO_UUID = defaultdict(dict)
ENTITY_UUID_TO_ID = defaultdict(dict)
def get_entity_id(entity_key, entity_label):
    """
    Get a unique id for each entity
    """
    entity_key = entity_key.lower()
    if entity_label not in ENTITY_ID_TO_UUID:
        ENTITY_ID_TO_UUID[entity_label] = {}
    if entity_key not in ENTITY_ID_TO_UUID[entity_label]:
        # Reuse the existing-node uuid when this natural key was in the live graph.
        tmp_uuid = RETAINED_UUID.get(entity_label, {}).get(entity_key)
        if tmp_uuid is None:
            # Name-keyed labels are minted from the raw source value but stored (and so
            # read back) after _clean_cypher_str has dropped quotes and control chars, so
            # any name containing those never matched and was re-minted on every rebuild.
            tmp_uuid = RETAINED_UUID.get(entity_label, {}).get(
                _cypher_value(entity_key).lower())
        if tmp_uuid is None:
            tmp_uuid = entity_label + '_' + str(uuid.uuid4())
            while tmp_uuid in ENTITY_UUID_TO_ID[entity_label]:
                tmp_uuid = entity_label + '_' + str(uuid.uuid4())
        ENTITY_ID_TO_UUID[entity_label][entity_key] = tmp_uuid
        ENTITY_UUID_TO_ID[entity_label][tmp_uuid] = entity_key
    return ENTITY_ID_TO_UUID[entity_label][entity_key]

nodes = defaultdict(dict)
edges = defaultdict(dict) # edges["from"] -> edges["to"] -> edges["type"] -> set(dataset_id1, dataset_id2, ...)
region_entity_mapper = defaultdict(dict)
geonameid_to_uuid_mapper = {}

# Retain existing-node uuids on a full rebuild (query the live graph BEFORE the load
# wipes it). No-op / all-new on the first build or with --no-retain-uuids.
if not args.no_retain_uuids:
    _rmap = build_retained_uuid_map(
        args.neo4j_uri, (args.neo4j_user, args.neo4j_password), args.neo4j_database)
    for _lab, _d in _rmap.items():
        RETAINED_UUID[_lab] = _d

# for record in tqdm(datasets_n[:500] + datasets_c[:500], desc="Processing records", total=len(datasets_n) + len(datasets_c)):
populate_geographical_nodes(nodes,geonameid_to_uuid_mapper)
populate_natural_hazard_type_nodes(nodes)
populate_climate_zone_nodes(nodes)
seed_gcmd_nodes(nodes)
for record in tqdm(combined, desc="Processing records", total=len(combined)):
    dataset_id = record["id"]
    record_entities = defaultdict(set)
    
    # Create entity nodes
    for key in record:
        if key in ENTITY_KEY_TO_LABEL:
            label = ENTITY_KEY_TO_LABEL[key]
            for entity_i, entity_id in enumerate(record[key]):
                if entity_id.lower() == "none":
                    continue
                # if the entity_id is found before, use the existing uuid
                elif entity_id.lower() in ENTITY_ID_TO_UUID[label]:
                    entity_uuid = get_entity_id(entity_id, label)
                    # update the name if it was originally in all lower case
                    if nodes[label][entity_uuid]['name'].islower():
                        nodes[label][entity_uuid]['name'] = entity_id
                else:
                    entity_uuid = get_entity_id(entity_id, label)
                    nodes[label][entity_uuid] = {"uuid": entity_uuid, "name": entity_id}
                    
                record_entities[label].add(entity_uuid)
                
                
                # if it is a label with attributes, fill in attributes
                if label in KEY_TO_ATTRIBUTES:
                    for attr in KEY_TO_ATTRIBUTES[label]:
                        if attr in record:
                            attr_id = record[attr][entity_i]
                            # if attr in nodes[label][entity_uuid] and attr_id != nodes[label][entity_uuid][attr]:
                            #     print(f"Conflict in {entity_id} for {attr}: {nodes[label][entity_uuid][attr]} vs {attr_id}")
                            if attr in nodes[label][entity_uuid] and attr_id not in nodes[label][entity_uuid][attr]:
                                nodes[label][entity_uuid][attr].append(attr_id)
                            else:
                                nodes[label][entity_uuid][attr] = [attr_id]
                
                populate_geographical_data(region_entity_mapper,geolocation_lookup,continents_df,domain_to_geoname_ids,label, entity_uuid, record)

    # Create edges
    for source_label, target_label, edge_type in RELATIONSHIP_SOURCE_TO_TARGET:
        if source_label in record_entities and target_label in record_entities:
            for source_uuid in record_entities[source_label]:
                for target_uuid in record_entities[target_label]:
                    if source_uuid != target_uuid:
                        if target_uuid not in edges[source_uuid]:
                            # edges[source_uuid][target_uuid] = defaultdict(set)
                            edges[source_uuid][target_uuid] = {}
                        # edges[source_uuid][target_uuid][edge_type].add(dataset_id)
                        if edge_type not in edges[source_uuid][target_uuid]:
                            edges[source_uuid][target_uuid][edge_type]={"dataset_count":1}
                        else:
                            edges[source_uuid][target_uuid][edge_type]['dataset_count']+=1

populate_geographical_edges(edges,region_entity_mapper,geonameid_to_uuid_mapper)
populate_model_component_data(nodes,edges,cmip6_model_metadata)
paper_id_to_uuid_mapper, paper_nodes_uuids = populate_paper_nodes(nodes)
populate_paper_edges(edges, paper_id_to_uuid_mapper, paper_nodes_uuids)
populate_citation_edges(edges, paper_id_to_uuid_mapper, args.citation_source)



# get statistics
num_nodes, num_edges = 0, 0
for label, node_dict in nodes.items():
    num_nodes += len(node_dict)
for from_node, to_dict in edges.items():
    for to_node, edge_dict in to_dict.items():
        num_edges += len(edge_dict)
print(f"Number of nodes: {num_nodes:,}")
print(f"Number of edges: {num_edges:,}")

if UNRESOLVED_OBS_MODELS:
    skipped = sum(UNRESOLVED_OBS_MODELS.values())
    print(f"\n[obs models] {skipped:,} observation triplets name a model with no Source node "
          f"({len(UNRESOLVED_OBS_MODELS):,} distinct names); their MODEL_* edges were skipped.")
    print("[obs models] most frequent unresolved names: "
          + ", ".join(f"{n} ({c})" for n, c in UNRESOLVED_OBS_MODELS.most_common(8)))


# Get current date in string
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Generate Cypher scripts
with open(f'neo4j/cypher/nodes_{current_date}.cypher', 'w') as f:
    for label, node_dict in tqdm(nodes.items(), desc="Writing nodes", total=len(nodes)):
        for uid, properties in node_dict.items():
            if len(properties) == 0:
                f.write(f'CREATE (:{label} {{uuid: "{uid}"}});\n')
            else:
                props = []
                for k, v in properties.items():
                    if k == "uuid":
                        continue
                    prop = format_cypher_property(k, v)
                    if prop is not None:
                        props.append(prop)
                props = ", ".join(props)
                f.write(f'CREATE (:{label} {{uuid: "{uid}", {props}}});\n')

with open(f'neo4j/cypher/edges_{current_date}.cypher', 'w') as f:
    for from_node, to_dict in tqdm(edges.items(), desc="Writing edges", total=len(edges)):
        for to_node, edge_dict in to_dict.items():
            # Label every MATCH so the load uses a per-label uuid index (seek) instead
            # of a label-less full scan. uuids are `Label_<uuid4>` and uuid4 contains no
            # underscore, so rsplit('_', 1)[0] recovers the label (even for dangling refs).
            from_label = from_node.rsplit('_', 1)[0]
            to_label = to_node.rsplit('_', 1)[0]
            f.write(f'MATCH (a:`{from_label}` {{uuid: "{from_node}"}})\n')
            f.write(f'MATCH (b:`{to_label}` {{uuid: "{to_node}"}})\n')
            for edge_type, value_dict in edge_dict.items():
                # dataset_ids_str = ", ".join([f'"{dataset_id}"' for dataset_id in dataset_ids])
                # f.write(f'CREATE (a)-[:{edge_type} {{dataset_ids: [{dataset_ids_str}]}}]->(b)\n')
                props = []
                for k, v in value_dict.items():
                    prop = format_cypher_property(k, v)
                    if prop is not None:
                        props.append(prop)
                edge_query=f'CREATE (a)-[:{edge_type}]->(b)\n'
                if len(props) > 0:
                    props = ", ".join(props)
                    edge_query=f'CREATE (a)-[:{edge_type} {{{props}}}]->(b)\n'
                f.write(edge_query)
            f.write(';\n')
        
with open(f'neo4j/json/nodes_{current_date}.json', 'w') as f:
    json.dump(nodes, f, indent=4)
    
# convert set to list
# for from_node, to_dict in edges.items():
#     for to_node, edge_dict in to_dict.items():
#         for edge_type, dataset_ids in edge_dict.items():
#             edge_dict[edge_type] = list(dataset_ids)
            
with open(f'neo4j/json/edges_{current_date}.json', 'w') as f:
    json.dump(edges, f, indent=4)
