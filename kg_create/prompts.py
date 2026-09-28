EXTRACT_MODEL_COMPONENTS_FROM_CMIP_CV="""
Your job is to extract model component name from unstructured text. Output just the model component name. I will provide some examples.

Input text: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
Output text: MetUM-HadGEM3-GA7.1

Input text: none, prescribed MACv2-SP
Output text: MACv2-SP

Input text: Canadian Model of Ocean Carbon (CMOC); NPZD ecosystem with OMIP prescribed carbonate chemistry
Output text: CMOC

If there is more than one model component name, separate them with a comma.
If it does not contain model component name, just return "none".

Input text: {input_text}
"""

__all__ =[ "EXTRACT_MODEL_COMPONENTS_FROM_CMIP_CV"]