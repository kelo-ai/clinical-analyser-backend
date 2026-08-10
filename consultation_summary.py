"""
Generates a short, plain-language recap of a consultation for quick
scanning on a doctor's dashboard. Distinct from the SOAP note (formal
clinical documentation) - this is meant to be skimmed in seconds across
a list of many consultations, not read as the clinical record itself.
"""

from llm_client import call_llm


def generate_consultation_summary(subjective: str, objective: str, assessment: str, plan: str) -> dict:
    """Returns a dict with keys: summary_text (string), key_points (list of strings)."""
    prompt = f"""Summarize this consultation for a doctor quickly scanning their
dashboard - they will read this in a few seconds, not as the clinical record itself
(that's the SOAP note, which already exists separately).

Subjective: {subjective}
Objective: {objective}
Assessment: {assessment}
Plan: {plan}

Provide:
1. summary_text: ONE short paragraph (2-3 sentences max) plainly describing what
   this visit was about and what was decided.
2. key_points: 2-4 short bullet-style phrases capturing the most important
   takeaways (e.g. "Recurrent back pain, cost barriers to therapy", "Heating pad +
   alternative therapies recommended").

Return ONLY valid JSON, no extra text, no markdown fences, no double-quote
characters inside string values:
{{
  "summary_text": "string",
  "key_points": [string, ...]
}}
"""

    result = call_llm(prompt)

    required_keys = {"summary_text", "key_points"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"Summary response missing expected fields: {result}")

    return result