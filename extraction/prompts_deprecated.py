"""
This is a unified file that houses all the prompts relevant for information extraction. 
"""
TITLE_EXTRACTION="""What is the title of the paper?"""
FIELD_EXTRACTION="""Summarize the research field of the paper in one phrase. Example: 'climate change'"""
KEYWORDS_EXTRACTION="""Extract 3-5 keywords summarizing the content, topic, and field of the paper. Each keyword should be separated by a comma. Example: climate change, global warming, greenhouse gases"""
PROBLEM_EXTRACTION="""Describe the problem the paper focuses on solving (e.g., shortcomings of existing methods,
challenges in the task, factors affecting performance). Summarize in one sentence (max 50 words). Example: Existing climate models struggle to accurately simulate extreme precipitation events due to coarse spatial resolution and inadequate representation of convective processes."""
METHOD_EXTRACTION="""Summarize the method proposed in the paper to solve the problem. Summarize in one sentence (max 50 words). Example: We propose a bias-correction method using quantile mapping to improve the accuracy of downscaled temperature projections from regional climate models."""
MODEL_EXTRACTION="""Name the climate model(s) this paper talks about. If nothing is proposed, answer 'no'. If present, the model name should be a single word in abbreviated form. Correct example: 'ACCESS-CM2'. Wrong example: 'ACCESS-CM2(Australian Community Climate and Earth System Simulator Climate Model Version 2).'"""
TASK_EXTRACTION="""Summarize the task the paper focuses on solving. Summarize in one sentence (max 50 words). Example: The task is to project future temperature and precipitation patterns over South Asia using downscaled CMIP6 climate model outputs."""
PAPER_SCREENING_PROMPT="""Does this paper have an observational dataset which the model produces or is evaluated upon, and does it have a metric? Answer with 'yes' only if it fulfills both conditions, otherwise answer with 'no'. Example: 'yes' or 'no'."""
OBSERVATION_AND_METRIC_EXTRACTION="Extract the observational dataset, evaluation metrics, and results for each climate model only from the table in the paper. The metrics should be entirely quantitative not descriptive."
INNOVATION_EXTRACTION=""""Describe the innovation or contribution of the paper in one sentence (max 50 words). Example: This paper introduces a novel bias-correction technique that significantly improves the accuracy of regional climate model projections for extreme precipitation events in South Asia."""
NO_EXPLANATION_PROMPT="""Do not put any additional comment or explanation."""
# Water bodies removed as it is merged with No_Country_Region - Water_Bodies: Oceans, seas, and lakes (e.g., Pacific Ocean).

ENTITY_TYPE_DEFINITIONS = """
Use ESGF/CMIP/CORDEX controlled vocabulary meanings for ESGF-derived types, and use WCRP/ES-DOC/Essential Model Documentation style model documentation for model-design terms.

Activity: CMIP/ESGF activity_id; a coordinated MIP or modeling activity such as CMIP, ScenarioMIP, AerChemMIP, C4MIP, or CORDEX. Do not use this for a single scenario or experiment.
ExperimentFamily: A family of related experiments sharing a scientific objective, usually broader than a single experiment_id.
Experiment: CMIP/ESGF experiment_id; a specific experiment, scenario, or protocol such as historical, piControl, ssp585, amip, or abrupt-4xCO2. Do not use this for Activity, Project, Member, or ensemble labels.
SubExperiment: CMIP sub_experiment_id; a controlled sub-case under an experiment, often start-year or hindcast/forecast configuration. Use "none" only if the text explicitly gives it as a metadata value.
Source: CMIP/ESGF source_id or comparable source model/dataset identifier, usually the registered climate model or reanalysis source that produced data, such as GFDL-ESM4 or ERA5. Do not use this for papers, institutions, components, variables, or experiments.
SourceType: CMIP/ESGF source_type; the model configuration class, such as AOGCM, AGCM, OGCM, LAND, AER, CHEM, BGC, ISM, or RCM.
SourceComponent: A named model component or submodel documented as part of a source, such as CICE4.1, MOM, JSBACH, NEMO, UKCA, or ECHAM. It can be an atmosphere, ocean, land, sea-ice, aerosol, chemistry, land-ice, or biogeochemistry component, but the realm label itself is not the component.
PhysicalScheme: A named or clearly described representation, parameterization, closure, formulation, or algorithm used by a source/component to represent a physical, chemical, biogeochemical, or surface-exchange process that is unresolved or simplified in the model. Examples: shallow convection scheme, Chikira-Sugiyama convection scheme, ecRad radiation scheme, cloud microphysics scheme, K-profile parameterization, or orographic gravity wave drag parameterization.
PhysicalFeature: A structural, numerical, grid, coordinate, dynamical-core, vertical-level, coupling, or configuration property of a source/component, following WCRP/ES-DOC model documentation practice. Examples: tripolar grid, cubed-sphere grid, finite-volume dynamical core, hybrid sigma-pressure vertical coordinate, Arakawa C-grid, 80 vertical levels, model top at 0.01 hPa, terrain-following grid, or prescribed-aerosol configuration.
SimulationType: The kind of simulation or run behavior, such as transient, equilibrium, control, initialized, coupled, atmosphere-only, or slab-ocean simulation. Do not use this for Experiment IDs.
Metric: A quantitative evaluation, diagnostic, or performance measure such as RMSE, bias, correlation, equilibrium climate sensitivity, transient climate response, or aerosol optical depth error. Do not use this for model design features.
Project: ESGF top-level project or collection such as CMIP6, CORDEX, obs4MIPs, or input4MIPs. Keep distinct from Activity when the paper gives both.
Institute: CMIP/ESGF institution_id or institution name responsible for model development, data production, or simulations.
Variable: CMIP/CF data variable or requested output quantity, including CMIP variable_id, long name, units, or CF standard name, such as tas, pr, sea surface temperature, or air_temperature.
Realm: CMIP/ESGF realm; controlled Earth-system realm such as atmos, ocean, land, seaIce, aerosol, atmosChem, landIce, or ocnBgchem.
Frequency: CMIP/ESGF frequency; temporal sampling such as fx, mon, day, 6hr, 3hr, or subhr.
Resolution: ESGF nominal_resolution or explicit spatial grid size/resolution bin such as 100 km, 1 degree, 0.25 deg, or 192 x 144. Do not use this for grid architecture.
Ensemble: A collection of related model runs or members. For CMIP6 variant labels, prefer Member.
Member: CMIP6 member_id or variant_label, usually r<i>i<j>p<k>f<l>, such as r1i1p1f1.
MIPEra: CMIP generation or era such as CMIP5 or CMIP6.
RCM: CORDEX regional climate model identifier/name used for fine-resolution regional simulations, such as REMO, RACMO, RegCM, HIRHAM, or ALADIN.
Domain: CORDEX or regional modeling domain identifier/geographic domain such as EUR-11, AFR-44, NAM-22, or a named regional domain.
Continent: A large continuous landmass such as Asia or Africa.
Country: A sovereign state or territory such as India or USA.
Country_Subdivision: Administrative unit within a country, such as California.
City: Urban locality such as Paris.
No_Country_Region: Region not represented as a country/subdivision/city, including open ocean regions, basins, seas, poles, deserts, and named non-sovereign regions.
Instrument: Observing device or sensor such as AVHRR, MODIS, radiometer, or lidar.
Platform: Physical carrier for an instrument such as ENVISAT, Aqua, Terra, aircraft, station, buoy, or satellite.
Weather_Event: A specific weather event or event class such as storm, drought, heatwave, or flood event.
Teleconnection: Large-scale climate mode or pattern such as ENSO, NAO, PDO, or IOD.
Ocean_Circulation: Ocean circulation feature or current system such as AMOC, Gulf Stream, or overturning circulation.
Natural_Hazard: A natural hazard occurrence or named hazard affecting systems, such as tsunami, earthquake, wildfire, or flood.
NaturalHazardType: Controlled hazard type, preferably NOAA-style event type such as Flood, Tornado, Wildfire, or Drought.
ClimateZone: Climate classification based on temperature and precipitation; use only Tropical, Dry, Temperate, Continental, or Polar.
"""

EXTRACTION_BOUNDARIES = """
Decision table:
- PhysicalScheme: answers "what representation/parameterization/closure/formulation/algorithm does the model use for a process?"
- PhysicalFeature: answers "what structural, numerical, grid, coordinate, dynamical-core, vertical-level, coupling, or configuration property does the model have?"
- Metric: answers "what quantitative evaluation or diagnostic measure is used or reported?"

Negative boundaries:
- Realm labels such as land, ocean, atmos, seaIce, aerosol, or atmosChem are Realm, not SourceComponent.
- SourceComponent names such as CICE4.1, JSBACH, MOM, NEMO, UKCA, ECHAM, CAM, CLM, or MPIOM are SourceComponent, not PhysicalScheme.
- CF standard names, CMIP variable_id values, and physical quantities such as tas, pr, air_temperature, precipitation, and sea surface temperature are Variable, not PhysicalFeature or PhysicalScheme.
- ESGF nominal_resolution values such as 100 km or 1 degree are Resolution. Grid architecture or coordinate design such as tripolar grid, cubed-sphere grid, or hybrid sigma-pressure coordinate can be PhysicalFeature.
- Experiments/scenarios such as historical, piControl, ssp126, ssp370, and ssp585 are Experiment, not Activity or Project.
- Activity/Project labels such as CMIP6, ScenarioMIP, CORDEX, and obs4MIPs are not Experiment.
- Do not classify a process name alone as PhysicalScheme unless the text identifies a scheme, parameterization, closure, formulation, algorithm, or named representation for that process.

Extraction discipline:
- Prefer omission over weak inference.
- Extract only if the paper explicitly supports the entity or relation.
- Keep the exact model, component, scheme, feature, variable, experiment, and member string when available.
- Do not invent generalized labels when the paper provides a specific name or version.
"""

ENTITY_EXTRACTION_PROMPT=f"""
You are given a scientific paper (or paragraph from it) related to climate modeling or Earth system science. The knowledge graph is primarily composed of ESGF metadata. Your task is to extract entity instances of entity types listed below.
The definitions of the entity types are given below:
{ENTITY_TYPE_DEFINITIONS}

{EXTRACTION_BOUNDARIES}

ONLY PROVIDE THEM IF THEY ARE PRESENT IN THE PAPER.
"""

KG_LINKING_PROMPT=f"""
You are given a scientific paper (or paragraph from it) related to climate modeling or Earth system science. The knowledge graph is primarily composed of ESGF metadata. Your task is to extract (source, relation, target) triplets with type that match the ontology below. For each extracted triplet, provide all supporting paragraphs from the text and explicitly resolve any anaphoras (e.g., "it", "this model") into their full entity names.
The ontology is given below in the form of (source type, target type, relation):
RELATIONSHIP_SOURCE_TO_TARGET = [
                                 ('Source','Source','HAS_SUBSEQUENT_VERSION'),
                                 ('Source','Source','INHERITED_FROM'),
                                 ('Source','SourceComponent','HAS_SOURCE_COMPONENT'),
                                 ('SourceComponent','SourceComponent','HAS_SUBSEQUENT_VERSION'),
                                 ('SourceComponent','SourceComponent','HAS_SOURCE_COMPONENT'),
                                 ('SourceComponent','Variable','PRODUCES_VARIABLE'),
                                 ('SourceComponent','Realm','APPLIES_TO_REALM'),
                                 ('Source','PhysicalScheme','USES_PHYSICAL_SCHEME'),
                                 ('Source','PhysicalFeature','HAS_PHYSICAL_FEATURE'),
                                 ('Source','SimulationType','HAS_SIMULATION_TYPE'),
                                 ('Source','Metric','HAS_METRIC'),
                                 ('Instrument','Variable','MEASURES_VARIABLE'),
                                 ('Instrument','Platform','MOUNTED_ON'),
                                 ('Weather_Event', 'Continent', 'IMPACTED_REGION'),
                                 ('Weather_Event', 'Country', 'IMPACTED_REGION'),
                                 ('Weather_Event', 'Country_Subdivision', 'IMPACTED_REGION'),
                                 ('Weather_Event', 'City', 'IMPACTED_REGION'), 
                                 ('Weather_Event', 'No_Country_Region', 'IMPACTED_REGION'),
                                 ('Teleconnection', 'Continent', 'INFLUENCES_REGION'),
                                 ('Teleconnection', 'Country', 'INFLUENCES_REGION'),
                                 ('Teleconnection', 'Country_Subdivision', 'INFLUENCES_REGION'),
                                 ('Teleconnection', 'City', 'INFLUENCES_REGION'), 
                                 ('Teleconnection', 'No_Country_Region', 'INFLUENCES_REGION'),
                                 ('Ocean_Circulation', 'Continent', 'OBSERVED_IN_REGION'),
                                 ('Ocean_Circulation', 'Country', 'OBSERVED_IN_REGION'),
                                 ('Ocean_Circulation', 'Country_Subdivision', 'OBSERVED_IN_REGION'),
                                 ('Ocean_Circulation', 'City', 'OBSERVED_IN_REGION'), 
                                 ('Ocean_Circulation', 'No_Country_Region', 'OBSERVED_IN_REGION'),
                                 ('Natural_Hazard', 'Continent', 'AFFECTED_REGION'),
                                 ('Natural_Hazard', 'Country', 'AFFECTED_REGION'),
                                 ('Natural_Hazard', 'Country_Subdivision', 'AFFECTED_REGION'),
                                 ('Natural_Hazard', 'City', 'AFFECTED_REGION'), 
                                 ('Natural_Hazard', 'No_Country_Region', 'AFFECTED_REGION'),
                                 ('Natural_Hazard', 'NaturalHazardType', 'IS_OF_TYPE'),
                                 ('No_Country_Region', 'ClimateZone', 'HAS_CLIMATE_ZONE')
                                 ]
The definitions of the entity types are given below:
{ENTITY_TYPE_DEFINITIONS}

{EXTRACTION_BOUNDARIES}

The relation definitions are given below:
INCLUDES_EXPERIMENT: Links an ExperimentFamily to the Experiments it encompasses. Indicates membership in a broader family of experiments.
HAS_SUB_EXPERIMENT: Connects an Experiment to a SubExperiment that refines, narrows, or extends it.
IS_OF_TYPE: Connects a Source (e.g., a model) to its SourceType (e.g., GCM, RCM, observational dataset) or a Natural_Hazard to its NaturalHazardType.
APPLIES_TO_REALM: Indicates that a Source, SourceComponent, or Experiment is associated with a specific Realm (e.g., atmosphere, ocean, land surface).
FOCUSES_ON_REALM: Specifies that an Activity is designed to study or emphasize a particular Realm.
PARTICIPATED_IN: Connects an Institute to a Project in which it was involved.
GENERATED_BY_ACTIVITY: Indicates that a Source (e.g., dataset or model output) was produced as part of a specific Activity.
PRODUCED_BY_INSTITUTE: Links a Source to the Institute responsible for its production.
BELONGS_TO_MIP_ERA: Indicates that a Source or ExperimentFamily is part of a specific MIP Era (e.g., CMIP5, CMIP6).
USED_IN_EXPERIMENT: Links a Source or RCM to the Experiment in which it was employed.
PRODUCES_VARIABLE: Specifies that a Source or SourceComponent outputs a particular Variable.
HAS_SUBSEQUENT_VERSION: Links a Source or SourceComponent to its direct newer version.
INHERITED_FROM: Indicates that a Source has been derived or inherited characteristics from another Source.
HAS_SOURCE_COMPONENT: Connects a Source to its constituent SourceComponents, or a SourceComponent to its subcomponents.
USES_PHYSICAL_SCHEME: Indicates that a Source employs a particular PhysicalScheme (e.g., radiation scheme, convection scheme).
HAS_PHYSICAL_FEATURE: Links a Source to a PhysicalFeature that characterizes it (e.g., resolution, topography treatment).
HAS_SIMULATION_TYPE: Connects a Source to the SimulationType it represents (e.g., control run, historical, scenario).
HAS_METRIC: Indicates that a Source is associated with a specific Metric used to evaluate its performance.
PART_OF_PROJECT: Indicates that a Source, RCM, Experiment, or Activity belongs to a given Project.
ASSOCIATED_WITH_MEMBER: Connects a Source to a specific Member of an ensemble.
SAMPLED_AT_FREQUENCY: Indicates the Frequency (e.g., daily, monthly) at which a Source produces outputs.
HAS_SPATIAL_RESOLUTION: Specifies the Resolution (spatial scale) of a Source, Variable, RCM, or Experiment.
PART_OF_ENSEMBLE: Links a Source or SubExperiment to an Ensemble.
USES_FORCING: Indicates that an Experiment incorporates a specific Forcing (e.g., GHGs, aerosols, solar).
INCLUDES_ENSEMBLE_MEMBER: Connects an Experiment to the Ensemble members it includes.
DRIVEN_BY_SOURCE: Indicates that an RCM is driven by a specific Source (typically a GCM).
COVERS_DOMAIN: Specifies that an RCM or Project is defined over a particular Domain (regional boundary).
MEASURES_VARIABLE: Links an Instrument to the Variable it is designed to measure.
MOUNTED_ON: Indicates that an Instrument is installed on a specific Platform (e.g., satellite, station).
PERFORMED_BY_INSTITUTE: Indicates that an Experiment was carried out by a particular Institute.
COVERS_REGION: Specifies that an RCM spans a given region (continent, country, subdivision, city, water body, etc.).
IMPACTED_REGION: Indicates that a Weather_Event affected a particular region.
INFLUENCES_REGION: Links a Teleconnection to the region it influences.
OBSERVED_IN_REGION: Indicates that an Ocean_Circulation phenomenon has been observed in a particular region.
AFFECTED_REGION: Indicates that a Natural_Hazard affected a particular region.
IN_CONTINENT: Connects a Country to the Continent it belongs to.
IN_COUNTRY: Connects a Country_Subdivision or City to its parent Country.
HAS_CLIMATE_ZONE: Links a No_Country_Region to its ClimateZone classification.

ONLY PROVIDE THE TRIPLETS IF THEY ARE PRESENT IN THE PAPER.
"""

# VALIDATION_PROMPT="""Given the following extracted (source, relation, target) triplet from a scientific paper related to climate modeling or Earth system science. validate whether the triplet is correct based on the paragraph context provided. Answer with 'valid' if the triplet is supported by the context, otherwise answer with 'invalid'. Do not provide an explanation.
# Extracted Triplet:
# Source: {source}
# Relation: {relation}
# Target: {target}
# Paragraph context: {paragraph}
# """

VALIDATION_PROMPT = """
You are given a paragraph from a scientific paper related to climate modeling or Earth system science, and a candidate
(source instance, relation, target instance) triplet extracted from it. The knowledge graph is primarily composed of ESGF
metadata and related controlled vocabularies.

Each instance is accompanied by its semantic type (e.g., Source, Experiment, Variable, Institution, Region).

Your task is to determine whether the paragraph reasonably supports the given triplet.
You may assume:
- The source or target may not be mentioned verbatim but can be inferred from context.
- Domain knowledge may be used to resolve abbreviations or implicit mentions.
- The paragraph may contain anaphora or implicit references.

The triplet should be considered **valid** if the relationship is explicitly stated or clearly implied by the paragraph.
It should be considered **invalid** only if the relationship is contradicted or unsupported by the text.

Respond with **only** one of the following labels:
- "valid"
- "invalid"

Do not provide explanations or additional text.

Extracted Triplet:
Source: {source}
Source Type: {source_type}
Relation: {relation}
Target: {target}
Target Type: {target_type}

Paragraph context:
{paragraph}
"""



GPT_VALIDATION_PROMPT="""You are given a paragraph related to climate modeling or Earth system science, and a (source instance, relation, target instance) triplet extracted from it. The knowledge graph is primarily composed of ESGF metadata. Your task is to verify if the triplet supports the paragraph from the paper. Assume anaphoras are there in the paragraph. Use your own knowledge to insert the source or target to the context of the paragraph. Answer with 'valid' or 'invalid'. Do not provide an explanation.
Extracted Triplet:
Source: {source}
Relation: {relation}
Target: {target}
Paragraph context: {paragraph}
"""

EMBEDDING_FALLBACK_LLM_PROMPT="""Given the following key word/phrase of an entity type extracted from paragraph context, output which of the instances it most likely refers to. If none of the instances match, output 'no match'. Do not provide an explanation.
Key Word/Phrase: {key_word_phrase}
Entity type: {entity_type}
Instances:
{instances}
Paragraph context: {paragraph}
"""
