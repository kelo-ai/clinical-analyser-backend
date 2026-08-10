"""
Client for calling the separate EMR System service from the AI Clinical
Assistant. This is the integration boundary between the two independent
systems - clinical_backend produces clinical data (SOAP notes, medication
findings); this client pushes approved/finalized data into the EMR's
permanent patient record.

If the EMR service is unreachable, these functions log the failure but do
NOT block or fail the calling operation (e.g. approving a SOAP note should
still succeed even if the EMR is temporarily down - the AI Clinical
Assistant's own record is still saved either way).
"""

import requests

EMR_BASE_URL = "http://localhost:8002"


def push_visit_record(patient_id: str, doctor_id: str, subjective: str,
                       objective: str, assessment: str, plan: str,
                       source_consultation_id: str) -> bool:
    """
    Pushes an approved SOAP note into the EMR as a permanent visit record.
    Returns True on success, False on failure (non-blocking for the caller).
    """
    try:
        response = requests.post(
            f"{EMR_BASE_URL}/visits",
            json={
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "subjective": subjective,
                "objective": objective,
                "assessment": assessment,
                "plan": plan,
                "source_consultation_id": source_consultation_id,
            },
            timeout=5,
        )
        if response.status_code != 200:
            print(f"EMR push_visit_record failed: {response.status_code} {response.text}")
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(f"EMR service unreachable (push_visit_record): {e}")
        return False


def push_medication_history(patient_id: str, medication_name: str,
                             rxcui: str | None, source_consultation_id: str) -> bool:
    """
    Pushes one medication finding into the EMR's permanent medication history.
    """
    try:
        response = requests.post(
            f"{EMR_BASE_URL}/medications",
            json={
                "patient_id": patient_id,
                "medication_name": medication_name,
                "rxcui": rxcui,
                "source_consultation_id": source_consultation_id,
            },
            timeout=5,
        )
        if response.status_code != 200:
            print(f"EMR push_medication_history failed: {response.status_code} {response.text}")
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(f"EMR service unreachable (push_medication_history): {e}")
        return False
