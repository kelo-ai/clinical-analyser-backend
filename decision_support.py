"""
AI decision-support module. Generates advisory suggestions for the doctor
to consider - explicitly NOT diagnoses, NOT directives, and NOT conclusions.

Safety framing is the entire point of this module: every output must read
as "you may want to consider X" rather than "X is true." The doctor is
always the final decision-maker. This module never writes to the EMR
directly and never modifies the SOAP note - it only surfaces suggestions
the doctor may act on entirely at their own discretion.
"""

from llm_client import call_llm


def generate_decision_support(subjective: str, objective: str, assessment: str, plan: str) -> dict:
    """
    Returns a dict with keys: differential_considerations, documentation_gaps,
    suggested_followup_questions, risk_flags - each a list of strings.
    """
    prompt = f"""You are assisting a licensed doctor by surfacing things worth a second
look before they finalize their documentation. You are NOT diagnosing the patient,
NOT overriding the doctor's clinical judgment, and NOT making treatment decisions.
The doctor has already written their own assessment and plan below - your job is
only to suggest additional angles they may want to consider, phrased as questions
or possibilities, never as facts or conclusions.

SOAP Note written by the doctor:
Subjective: {subjective}
Objective: {objective}
Assessment: {assessment}
Plan: {plan}

Provide FOUR lists:

1. differential_considerations: Other clinical possibilities that MIGHT be worth
   considering given the Subjective/Objective, phrased tentatively (e.g. "Could
   also consider ruling out X" not "Patient has X"). Return an empty list if the
   presentation is straightforward and nothing else seems relevant - do not
   force alternatives that don't fit.

2. documentation_gaps: Places where the Plan doesn't seem to connect clearly to
   the stated Assessment, or where documentation seems incomplete. Return an
   empty list if the note is internally consistent.

3. suggested_followup_questions: Questions the doctor might want to ask the
   patient at a future visit, based on things mentioned but not fully explored
   in the Subjective. Return an empty list if none seem relevant.

4. risk_flags: Anything mentioned that could be clinically significant but
   doesn't appear to have been addressed in the Assessment or Plan. Return an
   empty list if nothing stands out.

Be conservative - an empty list in any category is a normal, good outcome when
there's nothing meaningful to add. Do not manufacture suggestions just to fill
the lists.

Return ONLY valid JSON, no extra text, no markdown fences, no double-quote
characters inside string values:
{{
  "differential_considerations": [string, ...],
  "documentation_gaps": [string, ...],
  "suggested_followup_questions": [string, ...],
  "risk_flags": [string, ...]
}}
"""

    result = call_llm(prompt, temperature=0.2)

    required_keys = {"differential_considerations", "documentation_gaps",
                      "suggested_followup_questions", "risk_flags"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"Decision support response missing expected fields: {result}")

    return result