"""
This file contains utility functions for information extraction tasks.
"""

from prompts_deprecated import NO_EXPLANATION_PROMPT
from google.genai import types, Client
from models import Kg_Extraction_Model,Entity_Extraction_Model
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from consts import EMBEDDING_MATCH_ENTITIES
import json
import numpy as np
# from huggingface_hub import InferenceClient
# from vllm import LLM, SamplingParams
# from vllm.sampling_params import StructuredOutputsParams


# def load_vllm_model(model_name: str) -> LLM:
#     """
#     Docstring for load_vllm_model
    
#     :param model_name: Description
#     :type model_name: str
#     :return: Description
#     :rtype: LLM
#     """
#     llm = LLM(model=model_name)
#     return llm

# def generate_structured_content_vllm(llm: LLM, prompt: str, schema):
#     """
#     Docstring for generate_structured_content_vllm
    
#     :param llm: Description
#     :type llm: LLM
#     :param prompt: Description
#     :type prompt: str
#     :param schema: Description
#     """

#     structured_outputs_params=StructuredOutputsParams(
#         json=schema.model_json_schema()
#     )

#     params=SamplingParams(structured_outputs=structured_outputs_params)

#     response=llm.generate(prompts=prompt,sampling_params=params)

#     structured_data=schema.model_validate_json(response[0].outputs[0].text)
#     return structured_data




# def load_huggingface_client(token: str, provider: str = "auto") -> InferenceClient:
#     """
#     Load the HuggingFace Inference Client with the provided token.
#     """
#     client = InferenceClient(token=token, provider=provider)
#     return client


# def generate_structured_content_huggingface(client: InferenceClient,model_name: str, prompt: str, schema):
#     """
#     Generate content using a HuggingFace model.
#     """

#     response_format = {
#         "type": "json_schema",
#         "json_schema": {
#             "name": "SchemaResponse",
#             "schema": schema.model_json_schema(),
#             "strict": True
#         }
#     }

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that extracts information as per the schema."},
#         {"role": "user", "content": prompt}
#     ]

#     response= client.chat_completion(
#         model=model_name,
#         messages=messages,
#         response_format=response_format
#     )

#     structured_data = response.choices[0].message.content

#     return structured_data


def variable_to_long_name_lookup(variable_name, lookup_dict):
    """
    Convert a variable name to its long name using the provided lookup dictionary.
    """
    if lookup_dict is None:
        with open("output/variable_name_to_long_name.json","r") as f:
            lookup_dict=json.load(f)
    return lookup_dict[variable_name]

def variable_from_long_name(long_name, lookup_dict):
    """
    Convert a long name to its variable name using the provided lookup dictionary.
    If not found, return the long name itself.
    """
    if lookup_dict is None:
        with open("output/variable_name_to_long_name.json","r") as f:
            lookup_dict=json.load(f)
    reverse_lookup={v:k for k,v in lookup_dict.items()}
    return reverse_lookup.get(long_name,long_name)

def load_embedding_files():
    """
    Load the embedding files for entity matching.
    """
    embedding_files = {}
    for entity_type in EMBEDDING_MATCH_ENTITIES.keys():
        with open(f"output/embeddings/{entity_type}.json", 'r') as f:
            embedding_file = json.load(f)
        keys=[]
        embeddings=[]
        for k,v in embedding_file.items():
            keys.append(k)
            embeddings.append(np.array(v))
        embedding_files[entity_type]={
            "keys": keys,
            "embeddings": np.array(embeddings)
        }
    return embedding_files

def embedding_match(embed_model,entity_type, entity_instance, embedding_files, threshold=0.8):
    """
    Match the entity instance with the embedding files.
    Returns the key if a match is found, None otherwise.
    """
    if entity_type not in embedding_files:
        return None 
    
    embeddings = embedding_files[entity_type]["embeddings"]
    keys = embedding_files[entity_type]["keys"]
    query_embedding = embed_model.get_text_embedding(entity_instance)
    query = np.array(query_embedding).reshape(-1,1)
    scores = embeddings @ query
    index = np.argmax(scores)
    if scores[index] >= threshold:
        return keys[index]
    return None

def embedding_match_with_tolerance(embed_model,entity_type, entity_instance, embedding_files, threshold=0.8, tolerance_threshold=0.65):
    """
    Match the entity instance with the embedding beyond tolerance.
    Returns all the keys matched above threshold, None otherwise.
    """
    if entity_type not in embedding_files:
        return None 
    
    embeddings = embedding_files[entity_type]["embeddings"]
    keys = embedding_files[entity_type]["keys"]
    query_embedding = embed_model.get_text_embedding(entity_instance)
    query = np.array(query_embedding).reshape(-1,1)
    scores = embeddings @ query
    indices= np.where(scores >= tolerance_threshold)
    if len(indices[0])==0:
        return None
    return [keys[i] for i in indices[0]]

def consolidate_entities(instance: str, entity_type: str, embed_model, embedding_files: dict):
    """
    Consolidate entities from different sources.
    """
    embedding=embed_model.get_text_embedding(instance)
    embedding_np=np.array(embedding)
    if entity_type not in embedding_files:
        embedding_files[entity_type]={
            "keys": [instance],
            "embeddings": np.array([embedding_np])
        }
        return instance
    else:
        matched_instance=embedding_match(embed_model,entity_type, instance, embedding_files, threshold=0.8)
        if matched_instance is not None:
            return matched_instance
        else:
            embedding_files[entity_type]["keys"].append(instance)
            embedding_files[entity_type]["embeddings"]=np.vstack([embedding_files[entity_type]["embeddings"], embedding_np]) 
            return instance



    

def entity_exists(session,entity_type, entity_name):
    query = f"""
    MATCH (n:{entity_type})
    WHERE n.name = $name
    RETURN n.name AS name
    """
    if entity_type in ["Source"]:
        entity_name = entity_name.replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
        query = f"""
            MATCH (n:{entity_type})
            WHERE $name=REPLACE(REPLACE(REPLACE(REPLACE(n.name, "-", ""), "_", ""),".","")," ", "")
            RETURN n.name AS name
        """
    result = session.run(query, name=entity_name)
    output=result.single()
    return output if output is None else output["name"]




def check_pdf(file_path):
    """
    Check if a PDF file is corrupted or has no pages.
    
    Returns:
        tuple: (status, message)
        status: 'OK', 'CORRUPTED', 'NO_PAGES', 'ERROR'
    """
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Check if PDF has pages
            num_pages = len(reader.pages)
            
            if num_pages == 0:
                return 'NO_PAGES', 'PDF has 0 pages'
            
            # Try to access first page to verify structure
            try:
                _ = reader.pages[0]
                return 'OK', f'Valid PDF with {num_pages} page(s)'
            except Exception as e:
                return 'CORRUPTED', f'Cannot read pages: {str(e)}'
                
    except PyPDF2.errors.PdfReadError as e:
        return 'CORRUPTED', f'PDF read error: {str(e)}'
    except Exception as e:
        return 'ERROR', f'Unexpected error: {str(e)}'


def construct_openalex_metadata_url(paper_id: str) -> str:
    """
    Construct the OpenAlex metadata URL for a given paper ID.
    """
    base_url = "https://openalex.org/"
    return f"{base_url}{paper_id}"

def deprecated_parse_llm_response_general(response):
    """
    Parse the response from the LLM.
    """
    # Parse the content between <start> and <end> tags
    if response and response.text:
        start_tag = "<start>"
        end_tag = "<end>"
        start_index = response.text.find(start_tag)
        end_index = response.text.find(end_tag)

        if start_index != -1 and end_index != -1:
            content = response.text[start_index+len(start_tag):end_index].strip()
            return content
        else:
            return response.text.strip()
    return None

def prompt_llm(client: Client, prompt, cache, document, model_name="gemini-2.5-pro", config: types.GenerateContentConfig=None):
    """
    Prompt the LLM with a given prompt and return the response.
    """
    if config is None and cache is not None:
        generated_config = types.GenerateContentConfig(
            cached_content=cache.name
        )
    else:
        generated_config = config

    final_prompt=prompt + "\n" + NO_EXPLANATION_PROMPT
    contents=final_prompt
    if cache is None:
        contents=[document,final_prompt]
    # Remove formatting prompt and parse general.
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generated_config
    )

    if config is not None:
        return None if response is None else response.parsed
    
    return response.text.strip()

def parse_triplets(triplet_str):
    """
    Parse the triplets from the response string.
    """
    triplets = []
    lines = triplet_str.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line:
            parts=line.split(",")
            if len(parts) == 3:
                obs_dataset = parts[0].strip()
                metric = parts[1].strip()
                result = parts[2].strip()
                if obs_dataset[0] == "(":
                    obs_dataset = obs_dataset[1:].strip()
                if result[-1] == ")":
                    result = result[:-1].strip()
                triplets.append((obs_dataset, metric, result))
    return triplets

def graph_triplets_type_checking(triplets,ontology):
    """
    triplets: not exactly triplets, list of tuples ((source instance, source type), relation, (target instance, target type))
    ontology: list of tuples (source type, target type, relation)
    """
    filtered_triplets = []
    for triplet in triplets:
        _, source_type = triplet[0]
        relation = triplet[1]
        _, target_type = triplet[2]

        # Check if the relation is valid for the given source and target types
        if (source_type, target_type, relation) in ontology:
            filtered_triplets.append(triplet)

    return filtered_triplets



def deprecated_parse_entity_extraction_response(response,all_entity_types):
    """
    Parse the response from the entity extraction prompt.
    """
    tuples = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(",")
            if len(parts) == 2:
                entity_instance = parts[0].replace("(","").strip()
                entity_type = parts[1].replace(")","").strip()
                if entity_type in all_entity_types:
                    tuples.append((entity_instance, entity_type))
    return tuples

def validate_entity_extraction_response(response: list[Entity_Extraction_Model], all_entity_types):
    """
    Validate and parse the response from the entity extraction prompt.
    """
    validated_entities=[]
    for item in response:
        if item.entity_type not in all_entity_types:
            continue
        validated_entities.append(item)
    return validated_entities

def deprecated_parse_kg_extraction_response(response,ontology):
    """
    Parse the response from the KG extraction prompt.
    """
    """
    Parse the triplets from the response string.
    """
    triplets = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line:
            parts=line.split("--")
            # Parse string in the format (Source Entity, Source Type)--[Relation]-->(Target Entity, Target Type)
            if len(parts) == 3:
                source_instance_and_type = parts[0].replace("(","").replace(")","").strip()
                source_instance= source_instance_and_type.split(",")[0].strip()
                source_type = source_instance_and_type.split(",")[1].strip()
                relation = parts[1].replace("[","").replace("]","").strip()
                target_instance_and_type = parts[2].replace(">","").replace("(","").replace(")","").strip()
                target_instance= target_instance_and_type.split(",")[0].strip()
                target_type = target_instance_and_type.split(",")[1].strip()
                triplets.append(tuple(((source_instance, source_type), relation, (target_instance, target_type))))
    return graph_triplets_type_checking(triplets, ontology)

def validate_kg_extraction_response(response: list[Kg_Extraction_Model], ontology):
    """
    Validate and parse the response from the KG extraction prompt.
    """
    triplets = []
    for item in response:
        source_type = item.source_entity_type
        relation = item.relation
        target_type = item.target_entity_type

        if (source_type, target_type, relation) not in ontology:
            continue

        triplets.append(item)
    
    return triplets

def csv_adapter(file_path):
    """
    Adapter to read CSV files and return a list of dictionaries.
    Each dictionary represents a row in the CSV file.
    """
    import csv
    data = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data