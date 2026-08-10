"""
Generates a structured SOAP note from a diarized transcript.

Critical design point: the prompt explicitly tells the model which speaker
role said what, so it can correctly separate Subjective (patient-reported)
from Objective (doctor-observed) content - this only works because the
transcription module preserved speaker_role per segment.
"""

from llm_client import call_llm


def build_transcript_text(segments: list[dict]) -> str:
    """
    Converts diarized segments into a labeled transcript string for the
    prompt, e.g.:
        [DOCTOR]: How long have you had this cough?
        [PATIENT]: About a week now, mostly at night.
    Falls back to the raw speaker_label if a role hasn't been assigned yet.
    """
    lines = []
    for seg in segments:
        role = (seg.get("speaker_role") or seg.get("speaker_label", "unknown")).upper()
        lines.append(f"[{role}]: {seg['text']}")
    return "\n".join(lines)


def generate_soap_note(segments: list[dict]) -> dict:
    """
    Returns a dict with keys: subjective, objective, assessment, plan.
    """
    transcript_text = build_transcript_text(segments)

    prompt = f"""You are a clinical documentation assistant. Convert the following
doctor-patient consultation transcript into a structured SOAP note.

IMPORTANT: You are producing a DRAFT for a licensed doctor to review, edit,
and approve. Do not invent findings, diagnoses, or details that are not
supported by the transcript. If information for a section is not present
in the transcript, say so explicitly (e.g. "No objective findings recorded
in this transcript") rather than fabricating content.

Transcript (speaker roles are labeled):
{transcript_text}

Structure your output as:
- Subjective: what the PATIENT reported (symptoms, history, complaints, in their own words/context)
- Objective: what the DOCTOR observed or measured (exam findings, vitals mentioned, clinical observations)
- Assessment: the DOCTOR's stated or clearly implied clinical impression/diagnosis, based only on what's in the transcript
- Plan: treatment plan, prescriptions, follow-up instructions mentioned by the doctor

Return ONLY valid JSON, no extra text, no markdown fences. Do not use double
quote characters inside the string values themselves (e.g. when referencing
something the patient or doctor said) - paraphrase instead of quoting
directly, to keep the JSON valid:
{{
  "subjective": "string",
  "objective": "string",
  "assessment": "string",
  "plan": "string"
}}
"""

    result = call_llm(prompt, temperature=0.1)  # low temperature - documentation, not creative writing

    required_keys = {"subjective", "objective", "assessment", "plan"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"Gemini response missing expected SOAP fields: {result}")

    return result