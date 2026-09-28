"""
This script extracts model components from the CMIP6 source_id JSON file using a language model.
It processes the descriptions of model components, filters out certain values, and uses a language model to
generate a more structured output. The results are saved in a new JSON file.
"""
from llama_index.llms.ollama import Ollama
import json
from tqdm import tqdm
from prompts import EXTRACT_MODEL_COMPONENTS_FROM_CMIP_CV

llm=Ollama(model="qwen3:32b", request_timeout=360.0)
with open("other_sources/CMIP6_source_id.json", "r") as f:
    cmip6_source_id = json.load(f)

model_components=cmip6_source_id["source_id"]
new_model_components={}
for k,v in tqdm(model_components.items()):
    for each_realm,component in v["model_component"].items(): 
        if component["description"] in ["none","prescribed","specified","fixed","interactive","varies"]:
            continue
        input_text = EXTRACT_MODEL_COMPONENTS_FROM_CMIP_CV.format(input_text=component["description"])
        response = llm.complete(prompt=input_text+ " /no_think")
        response_text = response.text.split("</think>")[-1].strip()
        if response_text == "none":
            continue
        if k not in new_model_components:
            new_model_components[k] = {}
        tmp=response_text.split(",")
        tmp = [x.strip() for x in tmp if x.strip()]  # Remove empty    
        new_model_components[k][each_realm]=tmp

with open("output/new_model_components.json", "w") as f:
    json.dump(new_model_components, f, indent=4)