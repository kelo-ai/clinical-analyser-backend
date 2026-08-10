"""
Pydantic schemas - define the shape of data going in/out of the API,
separate from the database models (models.py).
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ConsultationCreate(BaseModel):
    doctor_id: str
    patient_id: str
    audio_source: str  # "live" | "upload"


class ConsultationOut(BaseModel):
    id: str
    doctor_id: str
    patient_id: str
    audio_source: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class TranscriptSegmentOut(BaseModel):
    id: str
    speaker_label: str
    speaker_role: Optional[str]
    start_time: float
    end_time: float
    text: str

    class Config:
        from_attributes = True


class SpeakerMapping(BaseModel):
    # e.g. {"speaker_0": "doctor", "speaker_1": "patient"}
    mapping: dict[str, str]


class SOAPNoteOut(BaseModel):
    id: str
    consultation_id: str
    subjective: Optional[str]
    objective: Optional[str]
    assessment: Optional[str]
    plan: Optional[str]
    status: str
    doctor_subjective: Optional[str]
    doctor_objective: Optional[str]
    doctor_assessment: Optional[str]
    doctor_plan: Optional[str]
    generated_at: datetime
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class SOAPNoteReview(BaseModel):
    """
    Doctor's review submission. Any field left as None means "approve the
    AI-generated version as-is" for that section; a provided string means
    "I'm overriding this section with my own text."
    """
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    action: str  # "approve" | "reject"


class PolypharmacyCheckOut(BaseModel):
    id: str
    consultation_id: str
    status: str  # pending | processing | completed | failed
    medications_found: Optional[str]   # JSON string, parsed client-side
    interactions_found: Optional[str]  # JSON string, parsed client-side
    error_message: Optional[str]
    review_status: str
    created_at: datetime
    completed_at: Optional[datetime]
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class PolypharmacyReview(BaseModel):
    action: str  # "approve" | "dismiss"


class DecisionSupportOut(BaseModel):
    id: str
    consultation_id: str
    status: str
    differential_considerations: Optional[str]
    documentation_gaps: Optional[str]
    suggested_followup_questions: Optional[str]
    risk_flags: Optional[str]
    error_message: Optional[str]
    review_status: str
    created_at: datetime
    completed_at: Optional[datetime]
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class DecisionSupportReview(BaseModel):
    action: str  # "acknowledge" | "dismiss"


class ConsultationSummaryOut(BaseModel):
    id: str
    consultation_id: str
    status: str
    summary_text: Optional[str]
    key_points: Optional[str]
    error_message: Optional[str]
    generated_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class PrescriptionInsightOut(BaseModel):
    id: str
    consultation_id: str
    patient_id: str
    medication_name: str
    status: str
    plain_language_purpose: Optional[str]
    general_usage_notes: Optional[str]
    common_things_to_know: Optional[str]
    error_message: Optional[str]
    generated_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class WellnessCheckinCreate(BaseModel):
    patient_id: str
    doctor_id: str
    message: Optional[str] = None


class WellnessCheckinOut(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    message: Optional[str]
    is_read: str
    created_at: datetime

    class Config:
        from_attributes = True