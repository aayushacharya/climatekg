#!/usr/bin/env python3
"""
rdf_utils.py

Helpers used by export_rdf.py to turn the n10s Turtle dumps into a single,
self-describing ontology file that WIDOCO (https://climatepub4kg.github.io/ontology/)
can document without any manual post-editing.

Three things happen here:

1. The n10s default IRI prefix ``neo4j://graph.schema#`` is rewritten to the
   published namespace ``https://climatepub4kg.github.io/ontology#`` (both in the
   ontology dump and in the instance dump).
2. The ontology dump is *enriched*: an ``owl:Ontology`` header carrying the site
   title/description/licence/version metadata WIDOCO renders, plus an
   ``rdfs:comment``/``skos:definition`` for every class, object property and
   datatype property (this is what fills WIDOCO's cross-reference section, which
   was previously blank).
3. Ontology + instances are merged into the unified ``climatepub4kg.ttl``.

Usage (prefix rewrite + enrichment of a single file):
  python rdf_utils.py input.ttl
"""

import datetime as dt
import sys

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS
from tqdm import tqdm

OLD = "neo4j://graph.schema#"
NEW = "https://climatepub4kg.github.io/ontology#"

# The ontology's own IRI (the namespace without the trailing '#'). WIDOCO looks for
# an owl:Ontology node; everything hanging off it becomes the documentation header.
ONTOLOGY_IRI = "https://climatepub4kg.github.io/ontology"

# Instances live in their own namespace. It is not bound as a prefix: every instance
# IRI is {ID_NS}{Class}/{id}, and a '/' cannot appear in a Turtle prefixed name.
ID_NS = "https://climatepub4kg.github.io/id/"

CLM = Namespace(NEW)
VANN = Namespace("http://purl.org/vocab/vann/")
BIBO = Namespace("http://purl.org/ontology/bibo/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

# ─────────────────────────────────────────────────────────────────────────────
# Ontology-level metadata (the WIDOCO landing page)
#
# Edit these constants to change what the published site says. CREATORS /
# PUBLISHER are the attribution shown under "Authors"; add individual people
# there when you want them credited by name.
# ─────────────────────────────────────────────────────────────────────────────

ONTOLOGY_TITLE = "ClimatePub4KG Ontology"

ONTOLOGY_ABSTRACT = (
    "ClimatePub4KG is a knowledge graph that links Earth System Grid Federation "
    "(ESGF) climate-model metadata to the peer-reviewed literature that uses it. "
    "This ontology defines the classes, object properties and datatype properties "
    "used by the graph: CMIP models, experiments, variables and the rest of the "
    "ESGF controlled vocabularies; publications and the fields, methods, tasks, "
    "problems, results and evaluation metrics extracted from them; and the "
    "geographic, instrument and phenomenon vocabularies that ground both sides in "
    "shared reference terms."
)

ONTOLOGY_DESCRIPTION = """ClimatePub4KG connects two bodies of climate-science knowledge that are normally kept apart: the structured metadata that describes climate-model simulations, and the scientific literature that analyses them.

The ontology is organised around four groups of terms.

ESGF and CMIP model metadata. Classes such as Source (a climate model), SourceComponent, Experiment, SubExperiment, ExperimentFamily, Activity, Project, MIPEra, Institute, Variable, Realm, Frequency, Resolution, GridLabel, Member, Ensemble, Forcing and Domain follow the CMIP6 controlled vocabularies and the facets used by ESGF search. Model-description classes (SourceType, SimulationType, PhysicalScheme, PhysicalFeature, Metric) capture how a model is built and how it behaves.

Scientific publications. A Paper is an OpenAlex-identified work, connected to its Author, Venue and Keyword records, and to the research content extracted from its full text: Field, Problem, Task, Method, Innovation, Result, EvaluationMetric and ObservationalDataset. The same content classes attach to models and methods as well as papers, so a finding can be traced from the claim to the model that produced it.

Geography. Continent, Country, Country_Subdivision, City and No_Country_Region are grounded in the GeoNames gazetteer and keep their GeoNames identifiers, bounding boxes and administrative codes. ClimateZone gives a coarse Koppen-Geiger style classification.

Observations and phenomena. Instrument and Platform align with the NASA GCMD keyword vocabularies; Teleconnection, Ocean_Circulation, Weather_Event, Natural_Hazard and NaturalHazardType name the large-scale patterns and events discussed in the corpus.

Instance IRIs live in a separate namespace, https://climatepub4kg.github.io/id/, and follow the pattern {id-namespace}{ClassName}/{identifier}. For ESGF classes the identifier is the URL-encoded entity name, so a model IRI reads .../id/Source/CESM2; a short disambiguating suffix is appended in the rare case where one name is used by more than one entity of the same class. Everything else is minted from the node's internal uuid.

Provenance is carried on the terms themselves: paper_id and added_from_paper_id record which publication an extracted entity or relationship came from, supporting_paragraphs keeps the sentences that justified it, and dataset_count records how many ESGF datasets support a metadata relationship. PAPER_MENTIONS is the general-purpose bridge from a publication to any entity mentioned in its text.

The graph is built by an LLM extraction pipeline over open-access climate papers followed by entity linking against the ESGF, GeoNames and GCMD vocabularies, so instance data should be treated as machine-generated and evidence-linked rather than hand-curated."""

# Attribution. Plain strings are rendered verbatim by WIDOCO.
CREATORS = ["ClimatePub4KG project"]
CONTRIBUTORS = ["Aayush Acharya", "Adamu Mustapha", "Huitong Pan", "Longin Jan Latecki", "Eduard Dragut"]
PUBLISHER = "ClimatePub4KG project"

ONTOLOGY_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
ONTOLOGY_HOMEPAGE = "https://climatepub4kg.github.io/ontology/"
ONTOLOGY_PREFIX = "clm"

# Upstream vocabularies the terms are derived from / aligned with.
ONTOLOGY_SOURCES = [
    "https://esgf.llnl.gov/",
    "https://github.com/WCRP-CMIP/CMIP6_CVs",
    "https://www.geonames.org/",
    "https://gcmd.earthdata.nasa.gov/KeywordViewer/",
    "https://openalex.org/",
]

# ─────────────────────────────────────────────────────────────────────────────
# Term definitions
#
# The tables themselves live in ontology_definitions.py, which imports nothing, so
# that v2/site can read the class definitions for its entity-type tooltips without
# taking on rdflib. They are re-exported here under their original names.
# ─────────────────────────────────────────────────────────────────────────────

from ontology_definitions import (  # noqa: E402  (kept with the other definitions)
    CLASS_DEFINITIONS,
    DATATYPE_PROPERTY_DEFINITIONS,
    OBJECT_PROPERTY_DEFINITIONS,
)

# n10s stores its own configuration and namespace-prefix bookkeeping in nodes whose
# labels start with '_'. The /rdf/neo4j/onto endpoint picks their property keys up as
# datatype properties, which would otherwise show up in the published documentation.
PREFIX_ARTIFACT_TERMS = {
    "clm", "clmid", "climid", "rdf", "rdfs", "owl", "xsd", "skos", "dc", "dcterms",
    "vann", "foaf", "geo", "xlink", "ex", "sch", "sh", "void", "neo", "n10s",
}


def remap_term(term):
    if isinstance(term, URIRef):
        s = str(term)
        if s.startswith(OLD):
            return URIRef(NEW + s[len(OLD):])
    return term


def _local_name(term):
    """Local part of a term in our ontology namespace, or None if it is not ours."""
    s = str(term)
    return s[len(NEW):] if s.startswith(NEW) else None


def bind_prefixes(g):
    """Bind the prefixes used by the enriched ontology so the Turtle stays readable."""
    g.bind("clm", CLM, override=True)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("vann", VANN)
    g.bind("bibo", BIBO)
    g.bind("foaf", FOAF)


def drop_internal_terms(g):
    """Remove n10s bookkeeping terms (``_``-prefixed, plus namespace-prefix artifacts)
    so they do not appear in the published documentation. Returns the names dropped."""
    dropped = set()
    for subject in set(g.subjects()):
        local = _local_name(subject)
        if local is None:
            continue
        if local.startswith("_") or local in PREFIX_ARTIFACT_TERMS:
            dropped.add(local)
            for triple in list(g.triples((subject, None, None))):
                g.remove(triple)
            for triple in list(g.triples((None, None, subject))):
                g.remove(triple)
    return dropped


def add_ontology_header(g, version=None):
    """Add the owl:Ontology node WIDOCO turns into the documentation header."""
    version = version or dt.datetime.now().strftime("%Y-%m-%d")
    onto = URIRef(ONTOLOGY_IRI)

    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal(ONTOLOGY_TITLE, lang="en")))
    g.add((onto, DCTERMS.title, Literal(ONTOLOGY_TITLE, lang="en")))
    g.add((onto, DCTERMS.abstract, Literal(ONTOLOGY_ABSTRACT, lang="en")))
    g.add((onto, DCTERMS.description, Literal(ONTOLOGY_DESCRIPTION, lang="en")))
    g.add((onto, RDFS.comment, Literal(ONTOLOGY_ABSTRACT, lang="en")))

    g.add((onto, OWL.versionInfo, Literal(version)))
    g.add((onto, OWL.versionIRI, URIRef(f"{ONTOLOGY_IRI}/{version}")))
    g.add((onto, DCTERMS.issued, Literal(version)))
    g.add((onto, DCTERMS.modified, Literal(version)))

    g.add((onto, DCTERMS.license, URIRef(ONTOLOGY_LICENSE)))
    g.add((onto, DCTERMS.publisher, Literal(PUBLISHER, lang="en")))
    for creator in CREATORS:
        g.add((onto, DCTERMS.creator, Literal(creator, lang="en")))
    for contributor in CONTRIBUTORS:
        g.add((onto, DCTERMS.contributor, Literal(contributor, lang="en")))

    g.add((onto, VANN.preferredNamespacePrefix, Literal(ONTOLOGY_PREFIX)))
    g.add((onto, VANN.preferredNamespaceUri, Literal(NEW)))
    g.add((onto, FOAF.homepage, URIRef(ONTOLOGY_HOMEPAGE)))
    g.add((onto, BIBO.status, URIRef("http://purl.org/ontology/bibo/status/published")))
    for source in ONTOLOGY_SOURCES:
        g.add((onto, DCTERMS.source, URIRef(source)))

    return onto


def add_term_definitions(g, onto=None, warn=True):
    """Attach an rdfs:comment / skos:definition to every class and property in the
    graph, using the definition tables above. Terms without a definition are reported
    so they can be added here rather than hand-edited into the generated Turtle."""
    tables = [
        (OWL.Class, CLASS_DEFINITIONS),
        (OWL.ObjectProperty, OBJECT_PROPERTY_DEFINITIONS),
        (OWL.DatatypeProperty, DATATYPE_PROPERTY_DEFINITIONS),
    ]

    defined, undefined = 0, []
    for term_type, definitions in tables:
        for subject in set(g.subjects(RDF.type, term_type)):
            local = _local_name(subject)
            if local is None:
                continue
            definition = definitions.get(local)
            if definition is None:
                undefined.append(f"{term_type.split('#')[-1]} {local}")
                continue
            if not any(g.triples((subject, RDFS.label, None))):
                g.add((subject, RDFS.label, Literal(local, lang="en")))
            g.set((subject, RDFS.comment, Literal(definition, lang="en")))
            g.set((subject, SKOS.definition, Literal(definition, lang="en")))
            if onto is not None:
                g.add((subject, RDFS.isDefinedBy, onto))
            defined += 1

    if warn and undefined:
        print(f"⚠️  {len(undefined)} term(s) have no definition in ontology_definitions.py:")
        for term in sorted(undefined):
            print(f"      {term}")
    return defined, undefined


def enrich_ontology(g, version=None):
    """Turn a bare n10s schema dump into a documentable ontology: drop the n10s
    bookkeeping terms, add the ontology header, and define every remaining term."""
    bind_prefixes(g)
    dropped = drop_internal_terms(g)
    if dropped:
        print(f"Dropped {len(dropped)} n10s bookkeeping term(s) from the ontology")
    onto = add_ontology_header(g, version=version)
    defined, _ = add_term_definitions(g, onto=onto)
    print(f"Annotated {defined} ontology term(s) with definitions")
    return g


def add_custom_prefix_ontology(inp, version=None):
    """Rewrite the n10s prefix in an ontology dump and enrich it in place."""
    g = Graph()
    # rdflib will auto-detect Turtle; you can pass format="turtle" if needed
    g.parse(inp)
    outp = inp

    g2 = Graph()

    # Carry over namespace bindings, swapping any that match OLD
    seen = set()
    for prefix, ns in g.namespaces():
        ns_str = str(ns)
        if ns_str == OLD:
            g2.bind(prefix, Namespace(NEW))
            seen.add(prefix)
        else:
            g2.bind(prefix, Namespace(ns_str))
            seen.add(prefix)

    # If there wasn't an explicit binding for the OLD prefix, you can add one (optional)
    if OLD not in [str(ns) for _, ns in g.namespaces()]:
        g2.bind("clm", Namespace(NEW))

    # Remap triples
    for s, p, o in g:
        g2.add((remap_term(s), remap_term(p), remap_term(o)))

    # Add the ontology header and the per-term definitions WIDOCO documents
    enrich_ontology(g2, version=version)

    # Serialize as Turtle
    g2.serialize(destination=outp, format="turtle")


def merge_ontology_and_instance_ttl(owl_file, instance_file):
    # Create a new graph for the merged data
    merged_g = Graph()

    # Load the ontology file
    print("Loading ontology file...")
    merged_g.parse(owl_file)
    print(f"Loaded {len(merged_g)} triples from ontology")

    # Load the triples file
    print("Loading triples file...")
    triples_before = len(merged_g)
    merged_g.parse(instance_file)
    triples_added = len(merged_g) - triples_before
    print(f"Loaded {triples_added} additional triples from triples file")

    # Keep the readable prefixes on the merged output too
    bind_prefixes(merged_g)

    # Display total triples
    print(f"\nTotal triples in merged graph: {len(merged_g)}")

    return merged_g


def handle_neo4j_prefix_instance(instance_file):
    # input/output files
    input_file = instance_file
    output_file = instance_file

    # define old and new namespace URIs
    old_ns = "neo4j://graph.schema#"
    new_ns = "https://climatepub4kg.github.io/ontology#"

    # create RDF graph and parse input file
    g = Graph()
    g.parse(input_file)

    # create namespace objects
    old = Namespace(old_ns)
    new = Namespace(new_ns)

    # create a new graph to store updated triples
    new_g = Graph()

    # copy namespaces, replacing the old one with the new prefix
    for prefix, ns in g.namespaces():
        if str(ns) == old_ns:
            new_g.bind("clm", new)  # new prefix
        else:
            new_g.bind(prefix, ns)

    # iterate over all triples and replace old namespace URIs
    for s, p, o in tqdm(g):
        s_new = str(s).replace(old_ns, new_ns) if str(s).startswith(old_ns) else s
        p_new = str(p).replace(old_ns, new_ns) if str(p).startswith(old_ns) else p
        o_new = str(o).replace(old_ns, new_ns) if str(o).startswith(old_ns) else o
        new_g.add((new_g.resource(s_new).identifier,
                   new_g.resource(p_new).identifier,
                   new_g.resource(o_new).identifier))

    # serialize back to RDF
    new_g.serialize(destination=output_file)

    print(f"✅ Prefix replaced and saved to {output_file}")


def export_unified_ttl(owl_file, instance_file, output_file, version=None):
    add_custom_prefix_ontology(owl_file, version=version)
    handle_neo4j_prefix_instance(instance_file)
    merge_ontology_and_instance_ttl(owl_file, instance_file).serialize(destination=output_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rdf_utils.py input.ttl")
        sys.exit(1)
    add_custom_prefix_ontology(sys.argv[1])
