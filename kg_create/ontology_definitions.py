#!/usr/bin/env python3
"""
ontology_definitions.py

The human-readable definition of every term in the ClimatePub4KG ontology: one
sentence or two per class, object property and datatype property.

`rdf_utils.add_term_definitions` attaches these to the exported ontology as
`rdfs:comment` / `skos:definition`, which is what WIDOCO renders at
https://climatepub4kg.github.io/ontology/. The visualization site (`v2/site`)
reads the class table to show each entity type's definition as a tooltip, so a
term is defined once, here, and both surfaces follow.

This module deliberately imports nothing: the site depends on fastapi, neo4j and
requests only, and must not pull rdflib in to read a dictionary.

Adding a term: put its definition in the table below that matches its kind. A
term that reaches the export without one is reported by name when export_rdf.py
runs, so nothing silently ships undefined.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Class definitions
# ─────────────────────────────────────────────────────────────────────────────

CLASS_DEFINITIONS = {
    # ── ESGF / CMIP model metadata ──────────────────────────────────────────
    "Activity": (
        "A Model Intercomparison Project (MIP) activity within a MIP era, such as CMIP, "
        "ScenarioMIP or DAMIP, that defines and coordinates a related set of climate model "
        "experiments. Corresponds to the ESGF activity_id facet."
    ),
    "Source": (
        "A climate or Earth system model: a named, versioned model configuration that produces "
        "simulation output. Corresponds to the ESGF source_id facet."
    ),
    "SourceType": (
        "The category of a climate model, such as AOGCM, AGCM, BGC or CHEM, describing which "
        "Earth system components the model couples. Corresponds to the ESGF source_type facet."
    ),
    "SourceComponent": (
        "A submodel of a climate model, such as its atmosphere, ocean, sea-ice, land-surface or "
        "aerosol component, usually together with the component's own version label."
    ),
    "Model": (
        "Legacy class for a climate model. Superseded by Source and retained only so that older "
        "IRIs continue to resolve; new data should use Source."
    ),
    "Experiment": (
        "A prescribed simulation protocol that models are run under, such as historical, piControl "
        "or ssp585. Corresponds to the ESGF experiment_id facet."
    ),
    "ExperimentFamily": (
        "A grouping of experiments that share a common protocol or scientific purpose, such as the "
        "DECK, historical or ScenarioMIP families."
    ),
    "SubExperiment": (
        "A refinement of an experiment that fixes additional initialisation conditions, for example "
        "a particular start year of a decadal prediction. Corresponds to the ESGF sub_experiment_id "
        "facet."
    ),
    "Variable": (
        "A geophysical output quantity reported by a model, such as tas (near-surface air "
        "temperature) or pr (precipitation), described by its CF standard name, long name and "
        "units. Corresponds to the ESGF variable_id facet."
    ),
    "Realm": (
        "A component of the Earth system that a model, model component, experiment or activity "
        "addresses, such as atmos, ocean, land, seaIce, landIce, aerosol or atmosChem."
    ),
    "Frequency": (
        "The temporal sampling frequency at which model output is reported, such as mon, day, 6hr "
        "or fx (time-invariant). Corresponds to the ESGF frequency facet."
    ),
    "Resolution": (
        "The nominal spatial resolution of a model, experiment or variable, expressed as a distance "
        "such as 100 km. Corresponds to the ESGF nominal_resolution facet."
    ),
    "GridLabel": (
        "The label identifying the grid a dataset is reported on, such as gn (the model's native "
        "grid) or gr (a regridded product). Corresponds to the ESGF grid_label facet."
    ),
    "Institute": (
        "The institution that develops a climate model or carries out an experiment. Corresponds to "
        "the ESGF institution_id facet."
    ),
    "Project": (
        "A research project or coordinated programme that models, experiments and datasets belong "
        "to, such as CMIP6, CORDEX or input4MIPs."
    ),
    "MIPEra": (
        "A generation of the Model Intercomparison Project, such as CMIP5, CMIP6 or CMIP6Plus."
    ),
    "Member": (
        "A variant label identifying one simulation of an experiment as a combination of "
        "realisation, initialisation, physics and forcing indices, such as r1i1p1f1. Corresponds to "
        "the ESGF member_id / variant_label facet."
    ),
    "Ensemble": (
        "A set of simulations of the same experiment that differ only in their realisation, "
        "initialisation, physics or forcing settings, used to sample uncertainty."
    ),
    "Forcing": (
        "An external forcing agent or forcing dataset prescribed by an experiment, such as "
        "greenhouse gas concentrations, aerosol emissions, land use or solar variability."
    ),
    "Domain": (
        "A named spatial domain over which a project or regional climate model is run, such as the "
        "CORDEX domains EUR-11 or AFR-44."
    ),
    "RCM": (
        "A regional climate model: a limited-area model run at higher resolution over a specific "
        "domain, driven at its boundaries by a global model or a reanalysis."
    ),
    "SimulationType": (
        "A characteristic style of simulation a model is run in, such as fully coupled, "
        "AMIP-style (prescribed sea surface temperatures), nudged or time-slice."
    ),
    "PhysicalScheme": (
        "A parameterisation or numerical scheme implemented in a model, such as a convection, "
        "radiation, cloud microphysics or boundary-layer scheme."
    ),
    "PhysicalFeature": (
        "A physical process or property represented in a model, such as interactive aerosols, "
        "dynamic vegetation, an interactive carbon cycle or a prognostic ice sheet."
    ),
    "Metric": (
        "A characteristic quantity reported for a model, such as its equilibrium climate "
        "sensitivity, transient climate response or effective radiative forcing."
    ),
    # ── Scientific publications ─────────────────────────────────────────────
    "Paper": (
        "A scientific publication about climate science, identified by its OpenAlex work "
        "identifier and, where available, its DOI. Papers are the provenance anchor for the "
        "extracted content in the graph."
    ),
    "Author": "A person credited as an author of a publication.",
    "Venue": (
        "The journal, conference, book series or repository in which a publication appeared."
    ),
    "Keyword": (
        "A subject keyword or concept assigned to a publication, taken from the publication's "
        "bibliographic record."
    ),
    "Field": (
        "The research field or discipline a publication belongs to, such as regional climate "
        "modelling or detection and attribution."
    ),
    "Problem": (
        "A research problem, limitation or open question that a publication, method or task "
        "addresses."
    ),
    "Method": (
        "A method, technique, model configuration or analysis procedure that a publication applies."
    ),
    "Task": (
        "A concrete research task that a publication or method carries out, such as bias "
        "correction, downscaling or extreme-event detection."
    ),
    "Innovation": (
        "A novel contribution claimed by a publication or method, such as a new scheme, dataset or "
        "analysis approach."
    ),
    "Result": (
        "A reported finding or quantitative outcome of a publication, model or method."
    ),
    "EvaluationMetric": (
        "A measure used to evaluate a model, method, task or result, such as root-mean-square "
        "error, bias, correlation or skill score."
    ),
    "ObservationalDataset": (
        "An observational, reanalysis or reference dataset used to evaluate or drive a model, such "
        "as ERA5, GPCP or HadCRUT."
    ),
    # ── Observations and phenomena ──────────────────────────────────────────
    "Instrument": (
        "A measuring instrument or sensor that observes geophysical variables, such as a satellite "
        "radiometer, radar or radiosonde. Aligned with the NASA GCMD instruments vocabulary."
    ),
    "Platform": (
        "A platform that carries observing instruments, such as a satellite, aircraft, ship, buoy "
        "or ground station. Aligned with the NASA GCMD platforms vocabulary."
    ),
    "Teleconnection": (
        "A large-scale mode of climate variability that links distant regions, such as ENSO, the "
        "North Atlantic Oscillation or the Pacific Decadal Oscillation. Aligned with the NASA GCMD "
        "science keywords."
    ),
    "Ocean_Circulation": (
        "An ocean current or circulation feature, such as the Atlantic Meridional Overturning "
        "Circulation, the Gulf Stream or the Antarctic Circumpolar Current. Aligned with the NASA "
        "GCMD science keywords."
    ),
    "Weather_Event": (
        "A weather or climate event discussed in the literature, such as a particular heatwave, "
        "storm, cold spell or extreme rainfall episode."
    ),
    "Natural_Hazard": (
        "A natural hazard studied in the corpus, such as a named flood, drought, wildfire or "
        "tropical cyclone."
    ),
    "NaturalHazardType": (
        "The category a natural hazard belongs to, such as flood, drought, heatwave, wildfire or "
        "tropical cyclone."
    ),
    # ── Geography ───────────────────────────────────────────────────────────
    "Continent": "A continent, as identified in the GeoNames gazetteer.",
    "Country": (
        "A country or dependent territory, as identified in the GeoNames gazetteer, together with "
        "its ISO codes and country-level attributes."
    ),
    "Country_Subdivision": (
        "A first-level administrative subdivision of a country, such as a state, province, region "
        "or canton, as identified in the GeoNames gazetteer."
    ),
    "City": (
        "A populated place, such as a city, town or village, as identified in the GeoNames "
        "gazetteer."
    ),
    "No_Country_Region": (
        "A named geographic region that is neither a country nor an administrative subdivision, "
        "such as a river basin, mountain range, sea, gulf or ocean sector, as identified in the "
        "GeoNames gazetteer."
    ),
    "Water_Bodies": (
        "A named body of water, such as a sea, lake or gulf. Retained for backwards compatibility; "
        "water bodies are currently represented as No_Country_Region instances."
    ),
    "ClimateZone": (
        "A broad climate zone (tropical, arid, temperate, cold or polar) following the top level of "
        "the Koppen-Geiger climate classification."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────
# Object property definitions
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_PROPERTY_DEFINITIONS = {
    # ── ESGF / CMIP metadata ────────────────────────────────────────────────
    "APPLIES_TO_REALM": (
        "Relates a climate model, model component or experiment to an Earth system realm it "
        "addresses."
    ),
    "ASSOCIATED_WITH_MEMBER": (
        "Relates a climate model to a variant label (member) under which its output was produced."
    ),
    "BELONGS_TO_MIP_ERA": (
        "Relates a climate model or experiment family to the MIP generation it belongs to."
    ),
    "COVERS_DOMAIN": (
        "Relates a project or regional climate model to the spatial domain it is run over."
    ),
    "COVERS_REGION": (
        "Relates a regional climate model to a geographic region that falls within its domain."
    ),
    "DRIVEN_BY_SOURCE": (
        "Relates a regional climate model to the global model that supplies its lateral boundary "
        "conditions."
    ),
    "FOCUSES_ON_REALM": (
        "Relates a MIP activity to the Earth system realm its experiments focus on."
    ),
    "GENERATED_BY_ACTIVITY": (
        "Relates a climate model to the MIP activity whose experiments produced its output."
    ),
    "HAS_METRIC": (
        "Relates a climate model to a characteristic quantity reported for it, such as its climate "
        "sensitivity."
    ),
    "HAS_PHYSICAL_FEATURE": (
        "Relates a climate model to a physical process or property it represents."
    ),
    "HAS_SIMULATION_TYPE": (
        "Relates a climate model to a style of simulation it is run in."
    ),
    "HAS_SOURCE_COMPONENT": (
        "Relates a climate model, or one of its components, to a constituent submodel."
    ),
    "HAS_SPATIAL_RESOLUTION": (
        "Relates a climate model, regional model, experiment or variable to its nominal spatial "
        "resolution."
    ),
    "HAS_SUBSEQUENT_VERSION": (
        "Relates a climate model or model component to a later version of itself."
    ),
    "HAS_SUB_EXPERIMENT": (
        "Relates an experiment to one of its sub-experiments."
    ),
    "INCLUDES_ENSEMBLE_MEMBER": (
        "Relates an experiment to an ensemble of simulations carried out for it."
    ),
    "INCLUDES_EXPERIMENT": (
        "Relates an experiment family to one of its member experiments."
    ),
    "INHERITED_FROM": (
        "Relates a climate model to an earlier model it was derived from."
    ),
    "IS_OF_TYPE": (
        "Relates an entity to its category: a climate model to its source type, or a natural "
        "hazard to its hazard type."
    ),
    "PARTICIPATED_IN": (
        "Relates an institution to a project it took part in."
    ),
    "PART_OF_ENSEMBLE": (
        "Relates a climate model or sub-experiment to an ensemble it contributes to."
    ),
    "PART_OF_PROJECT": (
        "Relates a climate model, regional model, experiment or activity to a project it belongs to."
    ),
    "PERFORMED_BY_INSTITUTE": (
        "Relates an experiment to the institution that performed it."
    ),
    "PRODUCED_BY_INSTITUTE": (
        "Relates a climate model to the institution that developed it."
    ),
    "PRODUCES_VARIABLE": (
        "Relates a climate model or model component to a geophysical variable it outputs."
    ),
    "SAMPLED_AT_FREQUENCY": (
        "Relates a climate model to a temporal frequency at which its output is reported."
    ),
    "USED_IN_EXPERIMENT": (
        "Relates a climate model or regional climate model to an experiment it was run for."
    ),
    "USES_FORCING": (
        "Relates an experiment to an external forcing it prescribes."
    ),
    "USES_PHYSICAL_SCHEME": (
        "Relates a climate model to a parameterisation or numerical scheme it implements."
    ),
    # ── Observations ────────────────────────────────────────────────────────
    "MEASURES_VARIABLE": (
        "Relates an observing instrument to a geophysical variable it measures."
    ),
    "MOUNTED_ON": (
        "Relates an observing instrument to the platform that carries it."
    ),
    # ── Publications ────────────────────────────────────────────────────────
    "PAPER_APPLIES_METHOD": "Relates a publication to a method it applies.",
    "PAPER_AUTHORED_BY": "Relates a publication to one of its authors.",
    "PAPER_BELONGS_TO_FIELD": "Relates a publication to the research field it belongs to.",
    "PAPER_EXPERIMENTS_ON_OBSERVATIONAL_DATASET": (
        "Relates a publication to an observational or reanalysis dataset it experiments on."
    ),
    "PAPER_HAS_INNOVATION": "Relates a publication to a novel contribution it claims.",
    "PAPER_HAS_KEYWORD": "Relates a publication to a subject keyword assigned to it.",
    "PAPER_HAS_RESULT": "Relates a publication to a finding it reports.",
    "PAPER_MENTIONS": (
        "Relates a publication to any entity mentioned in its text. This is the general-purpose "
        "provenance bridge between the literature and the rest of the graph, and is asserted "
        "whenever entity linking matched a mention to a node."
    ),
    "PAPER_PUBLISHED_IN_VENUE": "Relates a publication to the venue it appeared in.",
    "PAPER_SOLVES_PROBLEM": "Relates a publication to a research problem it addresses.",
    "PAPER_USES_METRIC": "Relates a publication to an evaluation metric it uses.",
    "PAPER_USES_MODEL": "Relates a publication to a climate model it uses or evaluates.",
    "PAPER_WORKS_ON_TASK": "Relates a publication to a research task it works on.",
    "CITED_BY": (
        "Relates a publication to another publication that cites it, as recorded in the "
        "bibliographic reference data."
    ),
    # ── Extracted research content ──────────────────────────────────────────
    "METHOD_EXPERIMENTS_ON_OBSERVATIONAL_DATASET": (
        "Relates a method to an observational or reanalysis dataset it is applied to."
    ),
    "METHOD_HAS_INNOVATION": "Relates a method to a novel contribution it introduces.",
    "METHOD_HAS_RESULT": "Relates a method to a finding obtained with it.",
    "METHOD_SOLVES_PROBLEM": "Relates a method to a research problem it addresses.",
    "METHOD_USES_METRIC": "Relates a method to an evaluation metric used to assess it.",
    "METHOD_WORKS_ON_TASK": "Relates a method to a research task it is used for.",
    "MODEL_EXPERIMENTS_ON_OBSERVATIONAL_DATASET": (
        "Relates a climate model to an observational or reanalysis dataset it is compared against."
    ),
    "MODEL_HAS_RESULT": "Relates a climate model to a finding reported about it.",
    "MODEL_USES_METRIC": "Relates a climate model to an evaluation metric used to assess it.",
    "RESULT_HAS_METRIC": "Relates a reported finding to the evaluation metric it is expressed in.",
    "TASK_EXPERIMENTS_ON_OBSERVATIONAL_DATASET": (
        "Relates a research task to an observational or reanalysis dataset it is carried out on."
    ),
    "TASK_FACES_PROBLEM": "Relates a research task to a problem that makes it difficult.",
    "TASK_USES_METRIC": "Relates a research task to an evaluation metric used to score it.",
    # ── Geography and phenomena ─────────────────────────────────────────────
    "IN_CONTINENT": "Relates a country to the continent it lies in.",
    "IN_COUNTRY": "Relates a city or administrative subdivision to the country that contains it.",
    "HAS_CLIMATE_ZONE": "Relates a geographic region to a climate zone it falls in.",
    "AFFECTED_REGION": "Relates a natural hazard to a geographic region it affected.",
    "IMPACTED_REGION": "Relates a weather or climate event to a geographic region it impacted.",
    "INFLUENCES_REGION": (
        "Relates a teleconnection pattern to a geographic region whose climate it influences."
    ),
    "OBSERVED_IN_REGION": (
        "Relates an ocean circulation feature to a geographic region it is observed in."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────
# Datatype property definitions
# ─────────────────────────────────────────────────────────────────────────────

DATATYPE_PROPERTY_DEFINITIONS = {
    # ── Identity and provenance ─────────────────────────────────────────────
    "name": (
        "The primary human-readable name of an entity. For ESGF classes the name is also the "
        "identifier used to mint the entity's IRI."
    ),
    "names": "Alternative names recorded for an entity (legacy; superseded by alternatenames).",
    "uuid": "Stable internal identifier of the entity within the knowledge graph.",
    "uri": (
        "External identifier IRI of the entity, for example its OpenAlex URL for papers, authors, "
        "venues and keywords."
    ),
    "source_uri": (
        "The external identifier IRI the entity was built from, kept aside when the RDF export "
        "mints the canonical clmid IRI into uri. Holds the OpenAlex URL for papers, authors, "
        "venues and keywords."
    ),
    "id": "OpenAlex work identifier of a publication.",
    "doi": "Digital Object Identifier of a publication.",
    "paper_id": (
        "OpenAlex identifier of the publication this entity was extracted from (single-valued)."
    ),
    "added_from_paper_id": (
        "OpenAlex identifiers of the publications from which this entity or relationship was "
        "extracted."
    ),
    "supporting_paragraphs": (
        "Passages from the source publication that were used as evidence for an extracted "
        "relationship."
    ),
    "dataset_count": (
        "Number of ESGF datasets supporting a metadata relationship, used as a weight on facet "
        "co-occurrence."
    ),
    # ── Publications ────────────────────────────────────────────────────────
    "title": "Title of a publication.",
    "abstract": "Abstract of a publication.",
    "publication_year": "Year in which a publication appeared.",
    "type": "Kind of publication venue, such as journal, conference, book series or repository.",
    # ── ESGF / CMIP attributes ──────────────────────────────────────────────
    "experiment_title": "Full descriptive title of a CMIP experiment.",
    "cf_standard_name": "CF standard name of a variable.",
    "variable_long_name": "Long, descriptive name of a variable.",
    "variable_units": "Physical units in which a variable is reported.",
    "rcm_version": "Version label of a regional climate model.",
    # ── GCMD alignment ──────────────────────────────────────────────────────
    "gcmd_uri": "IRI of the matching concept in the NASA GCMD keyword vocabularies.",
    "gcmd_uuid": "UUID of the matching concept in the NASA GCMD keyword vocabularies.",
    # ── GeoNames attributes ─────────────────────────────────────────────────
    "geonameid": "GeoNames identifier of a geographic entity.",
    "asciiname": "ASCII transliteration of a geographic name, as recorded by GeoNames.",
    "alternatenames": "Alternative and localised names of a place, as recorded by GeoNames.",
    "latitude": "Latitude of a place in decimal degrees (WGS 84).",
    "longitude": "Longitude of a place in decimal degrees (WGS 84).",
    "north": "Northern edge of the entity's bounding box, in decimal degrees.",
    "south": "Southern edge of the entity's bounding box, in decimal degrees.",
    "east": "Eastern edge of the entity's bounding box, in decimal degrees.",
    "west": "Western edge of the entity's bounding box, in decimal degrees.",
    "feature_class": "GeoNames feature class: a single letter giving the broad kind of place.",
    "feature_code": "GeoNames feature code: a fine-grained classification of the kind of place.",
    "country_code": "ISO 3166-1 alpha-2 code of the country containing the place.",
    "cc2": "Additional country codes for a place that spans more than one country.",
    "admin1_code": "Code of the first-level administrative division containing the place.",
    "admin2_code": "Code of the second-level administrative division containing the place.",
    "admin3_code": "Code of the third-level administrative division containing the place.",
    "admin4_code": "Code of the fourth-level administrative division containing the place.",
    "code": "Administrative code of a country subdivision, in GeoNames country.admin1 form.",
    "population": "Population recorded for the place.",
    "elevation": "Elevation of the place in metres, where recorded.",
    "dem": (
        "Digital elevation model value in metres at the place's coordinates (SRTM3 or GTOPO30)."
    ),
    "timezone": "IANA time zone identifier of the place.",
    "modification_date": "Date on which the GeoNames record was last modified.",
    "iso": "ISO 3166-1 alpha-2 code of a country, or the two-letter GeoNames continent code.",
    "iso3": "ISO 3166-1 alpha-3 code of a country.",
    "isonumeric": "ISO 3166-1 numeric code of a country.",
    "fips": "FIPS 10-4 code of a country.",
    "equivalent_fips_code": "Equivalent FIPS code recorded for a country.",
    "capital": "Capital city of a country.",
    "area_sqkm": "Area of a country in square kilometres.",
    "continent": "Two-letter GeoNames continent code of a country.",
    "country": "Country name as recorded in the GeoNames country table.",
    "currencycode": "ISO 4217 code of a country's currency.",
    "currencyname": "Name of a country's currency.",
    "languages": "Languages spoken in a country, as language tags.",
    "neighbours": "ISO country codes of a country's land neighbours.",
    "phone": "International dialling prefix of a country.",
    "postal_code_format": "Template describing the format of a country's postal codes.",
    "postal_code_regex": "Regular expression validating a country's postal codes.",
    "tld": "Country-code top-level Internet domain of a country.",
}
