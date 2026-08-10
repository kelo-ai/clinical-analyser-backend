"""
Extracts medication names mentioned in a consultation (from the SOAP note's
Subjective and Plan sections, where meds are most likely to be mentioned).
"""

from llm_client import call_llm


def extract_medications(soap_subjective: str, soap_plan: str) -> list[str]:
    """
    Returns a list of medication names mentioned (brand or generic).
    Returns an empty list if none are found - this is a normal, valid result,
    not an error.
    """
    prompt = f"""Extract every medication name mentioned in the text below.
Include both brand names and generic names exactly as written. Do not
include dosage forms, supplements unless clearly medication-like, or
non-medication treatments (e.g. "physical therapy" is not a medication).

Subjective: {soap_subjective}

Plan: {soap_plan}

Return ONLY valid JSON matching this schema, nothing else:
{{
  "medications": [string, ...]
}}
If no medications are mentioned, return {{"medications": []}}
"""

    result = call_llm(prompt, temperature=0.0)  # deterministic extraction, not generation

    if "medications" not in result or not isinstance(result["medications"], list):
        raise ValueError(f"Expected a 'medications' list, got: {result}")

    return result["medications"]