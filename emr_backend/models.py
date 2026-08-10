"""
Core EMR data models. Unlike the AI Clinical Assistant's per-consultation
records, these represent the LONGITUDINAL patient record - accumulated
history across every visit, not just one.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Date
from sqlalchemy.orm import relationship

from database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=generate_uuid)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False, unique=True, index=True)
    date_of_birth = Column(Date, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    insurance_provider = Column(String, nullable=True)

    # Known allergies and chronic conditions - persistent across all visits,
    # not tied to any single consultation.
    known_allergies = Column(Text, nullable=True)
    chronic_conditions = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    visits = relationship("VisitRecord", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("MedicationHistoryEntry", back_populates="patient", cascade="all, delete-orphan")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(String, primary_key=True, default=generate_uuid)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False, unique=True, index=True)
    specialty = Column(String, nullable=True)
    license_number = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class VisitRecord(Base):
    """
    One entry in a patient's visit history. This is what gets created when
    a consultation is approved in the AI Clinical Assistant - the EMR's
    permanent record of that visit having happened.
    """
    __tablename__ = "visit_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(String, nullable=False)  # references Doctor.id

    visit_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Populated from the approved SOAP note in the AI Clinical Assistant
    subjective = Column(Text, nullable=True)
    objective = Column(Text, nullable=True)
    assessment = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)

    # Traceability back to the source system - lets you look up the original
    # transcript/audio in the AI Clinical Assistant if ever needed.
    source_consultation_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="visits")


class MedicationHistoryEntry(Base):
    """
    One medication entry in a patient's accumulated medication history.
    Distinct from a single visit's SOAP note - this persists across the
    patient's entire relationship with the clinic, letting a doctor see
    "everything this patient has ever been prescribed," not just today's.
    """
    __tablename__ = "medication_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)

    medication_name = Column(String, nullable=False)
    rxcui = Column(String, nullable=True)  # RxNorm identifier, if resolved

    prescribed_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String, default="active")  # active | discontinued

    source_consultation_id = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="medications")
