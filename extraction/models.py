# Pydantic models for information extraction tasks.

from pydantic import BaseModel

class Kg_Extraction_Model(BaseModel):
    source_entity_instance: str
    source_entity_type: str
    relation: str
    target_entity_instance: str
    target_entity_type: str
    paragraph: str

class Observation_Metric_Model(BaseModel):
    model: str
    observational_dataset: str
    evaluation_metric: str
    result: str

class Entity_Extraction_Model(BaseModel):
    entity_instance: str
    entity_type: str