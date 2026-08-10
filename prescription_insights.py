"""
Generates patient-facing, plain-language explanations of prescribed
medications. This is the most safety-sensitive generator in the system,
since it's read directly by patients rather than filtered through a doctor
first.

Hard rules enforced in the prompt:
  - NEVER include specific dosing, frequency, or "how much to take" - that's
    the doctor's/pharmacist's job, stated on the actual prescription label.
  - NEVER tell the patient to start, stop, or change their medication.
  - ALWAYS include a reminder to talk to their doctor or pharmacist for any
    medical questions.
  - General, educational information only - not personalized medical advice.
"""

from llm_client import call_llm


def generate_prescription_insight(medication_name: str) -> dict:
    """
    Returns a dict with keys: plain_language_purpose, general_usage_notes,
    common_things_to_know (list of strings).
    """
    prompt = f"""You are writing a short, plain-language educational note for a
PATIENT (not a clinician) about a medication they were prescribed: {medication_name}

STRICT RULES - these are non-negotiable:
- Do NOT include any specific dose, frequency, or timing instructions (e.g. do not
  say "take 500mg twice daily") - the patient's actual prescription label and their
  doctor/pharmacist are the source for that, not this note.
- Do NOT tell the patient to start, stop, increase, or decrease their medication.
- Do NOT diagnose or suggest what condition they have - only describe what this
  medication is generally used for in general terms.
- Write in simple, non-technical language a patient without medical training can
  understand.
- Always end common_things_to_know with a reminder to talk to their doctor or
  pharmacist with any questions about their specific situation.

Provide:
1. plain_language_purpose: 1-2 simple sentences on what this type of medication is
   generally used for.
2. general_usage_notes: 1-2 simple sentences of general, non-dosage information
   (e.g. "this is often taken with food" is fine; a specific dose is not).
3. common_things_to_know: 2-4 short, general educational points a patient might
   find helpful to know, written simply. The LAST item must always be a reminder
   to consult their doctor or pharmacist for questions specific to them.

Return ONLY valid JSON, no extra text, no markdown fences, no double-quote
characters inside string values:
{{
  "plain_language_purpose": "string",
  "general_usage_notes": "string",
  "common_things_to_know": [string, ...]
}}
"""

    result = call_llm(prompt)

    required_keys = {"plain_language_purpose", "general_usage_notes", "common_things_to_know"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"Prescription insight response missing expected fields: {result}")

    return result