"""
ORM models for the transcription module.

Consultation: one recording/upload session between a doctor and patient.
TranscriptSegment: one diarized utterance within a consultation, tagged
with which speaker said it (raw label first, then mapped to doctor/patient).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(String, primary_key=True, default=generate_uuid)
    doctor_id = Column(String, nullable=False)
    patient_id = Column(String, nullable=False)

    audio_source = Column(String, nullable=False)   # "live" | "upload"
    status = Column(String, default="created")       # created | uploaded | transcribing | transcribed | reviewed
    audio_file_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    segments = relationship("TranscriptSegment", back_populates="consultation", cascade="all, delete-orphan")
    soap_note = relationship("SOAPNote", back_populates="consultation", uselist=False, cascade="all, delete-orphan")
    polypharmacy_check = relationship("PolypharmacyCheck", back_populates="consultation", uselist=False, cascade="all, delete-orphan")
    decision_support = relationship("DecisionSupportInsight", back_populates="consultation", uselist=False, cascade="all, delete-orphan")
    summary = relationship("ConsultationSummary", back_populates="consultation", uselist=False, cascade="all, delete-orphan")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False)

    speaker_label = Column(String, nullable=False)    # raw from Deepgram, e.g. "speaker_0"
    speaker_role = Column(String, nullable=True)      # mapped later: "doctor" | "patient"

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)

    consultation = relationship("Consultation", back_populates="segments")


class SOAPNote(Base):
    __tablename__ = "soap_notes"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False, unique=True)

    subjective = Column(Text, nullable=True)   # patient-reported symptoms/history
    objective = Column(Text, nullable=True)    # doctor's findings/exam results
    assessment = Column(Text, nullable=True)   # diagnosis/clinical impression
    plan = Column(Text, nullable=True)         # treatment plan, prescriptions, follow-up

    status = Column(String, default="pending_review")  # pending_review | approved | edited | rejected

    # If the doctor edits any section, store their final version here per field,
    # so the original AI draft is preserved for audit purposes (never overwritten).
    doctor_subjective = Column(Text, nullable=True)
    doctor_objective = Column(Text, nullable=True)
    doctor_assessment = Column(Text, nullable=True)
    doctor_plan = Column(Text, nullable=True)

    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at = Column(DateTime, nullable=True)

    consultation = relationship("Consultation", back_populates="soap_note")


class PolypharmacyCheck(Base):
    __tablename__ = "polypharmacy_checks"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False, unique=True)

    # Async status: this check runs as a background task, not inline with
    # the request, so the app never blocks the doctor's UI waiting for it.
    status = Column(String, default="pending")  # pending | processing | completed | failed

    medications_found = Column(Text, nullable=True)   # JSON list of extracted medication names
    interactions_found = Column(Text, nullable=True)   # JSON list of interaction results from RxNorm
    error_message = Column(Text, nullable=True)

    review_status = Column(String, default="pending_review")  # pending_review | approved | dismissed

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    consultation = relationship("Consultation", back_populates="polypharmacy_check")


class DecisionSupportInsight(Base):
    """
    Advisory suggestions for the doctor to consider - never presented as
    findings or conclusions. The doctor remains the sole decision-maker;
    this module's entire purpose is to surface things worth a second look,
    not to diagnose or direct treatment.
    """
    __tablename__ = "decision_support_insights"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False, unique=True)

    status = Column(String, default="pending")  # pending | processing | completed | failed

    # Each stored as a JSON list of strings
    differential_considerations = Column(Text, nullable=True)
    documentation_gaps = Column(Text, nullable=True)
    suggested_followup_questions = Column(Text, nullable=True)
    risk_flags = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)

    # Doctor's final disposition - required before this is considered "resolved"
    review_status = Column(String, default="pending_review")  # pending_review | acknowledged | dismissed

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    consultation = relationship("Consultation", back_populates="decision_support")


class ConsultationSummary(Base):
    """
    Short, plain-language recap of a consultation - distinct from the SOAP
    note, which is the formal clinical record. This is meant for quick
    scanning on a doctor's dashboard (e.g. "what happened in this visit?"
    across a list of many consultations), not for documentation purposes.
    """
    __tablename__ = "consultation_summaries"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False, unique=True)

    status = Column(String, default="pending")  # pending | processing | completed | failed
    summary_text = Column(Text, nullable=True)   # 2-3 sentence plain-language recap
    key_points = Column(Text, nullable=True)      # JSON list of short bullet points

    error_message = Column(Text, nullable=True)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    consultation = relationship("Consultation", back_populates="summary")


class WellnessCheckin(Base):
    """
    A simple patient-initiated 'I'm doing fine' check-in, notifying the
    doctor who most recently treated them. Not tied to a specific
    consultation - just a lightweight wellness signal the doctor sees on
    their dashboard.
    """
    __tablename__ = "wellness_checkins"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, nullable=False)
    doctor_id = Column(String, nullable=False)  # the doctor being notified

    message = Column(Text, nullable=True)  # optional note from the patient
    is_read = Column(String, default="unread")  # "unread" | "read"

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PrescriptionInsight(Base):
    """
    Patient-facing plain-language explanation of a prescribed medication.
    Distinct from clinical documentation: written for a patient to
    understand, not a clinician. Always includes a reminder to consult
    their doctor/pharmacist for medical questions - this is informational
    only, never a substitute for professional advice.
    """
    __tablename__ = "prescription_insights"

    id = Column(String, primary_key=True, default=generate_uuid)
    consultation_id = Column(String, ForeignKey("consultations.id"), nullable=False)
    patient_id = Column(String, nullable=False)

    medication_name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending | processing | completed | failed

    plain_language_purpose = Column(Text, nullable=True)   # what it's generally used for
    general_usage_notes = Column(Text, nullable=True)       # general, non-dosage informational notes
    common_things_to_know = Column(Text, nullable=True)     # JSON list, e.g. common general advice

    error_message = Column(Text, nullable=True)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)