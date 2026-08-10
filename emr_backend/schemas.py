from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


class PatientCreate(BaseModel):
    full_name: str
    phone_number: str
    date_of_birth: Optional[date] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    insurance_provider: Optional[str] = None
    known_allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None


class PatientOut(BaseModel):
    id: str
    full_name: str
    phone_number: str
    date_of_birth: Optional[date]
    contact_phone: Optional[str]
    contact_email: Optional[str]
    insurance_provider: Optional[str]
    known_allergies: Optional[str]
    chronic_conditions: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PatientUpdate(BaseModel):
    """All fields optional - only fields actually provided get updated
    (partial update), so the doctor doesn't need to resend everything."""
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    insurance_provider: Optional[str] = None
    known_allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None


class DoctorCreate(BaseModel):
    full_name: str
    phone_number: str
    specialty: Optional[str] = None
    license_number: Optional[str] = None


class DoctorOut(BaseModel):
    id: str
    full_name: str
    phone_number: str
    specialty: Optional[str]
    license_number: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class VisitRecordCreate(BaseModel):
    """
    Sent by the AI Clinical Assistant when a SOAP note is approved.
    This is the integration point between the two systems.
    """
    patient_id: str
    doctor_id: str
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    source_consultation_id: Optional[str] = None


class VisitRecordOut(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    visit_date: datetime
    subjective: Optional[str]
    objective: Optional[str]
    assessment: Optional[str]
    plan: Optional[str]
    source_consultation_id: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MedicationHistoryCreate(BaseModel):
    """Sent by the AI Clinical Assistant after polypharmacy detection extracts medications."""
    patient_id: str
    medication_name: str
    rxcui: Optional[str] = None
    source_consultation_id: Optional[str] = None


class MedicationHistoryOut(BaseModel):
    id: str
    patient_id: str
    medication_name: str
    rxcui: Optional[str]
    prescribed_date: datetime
    status: str
    source_consultation_id: Optional[str]

    class Config:
        from_attributes = True


class PatientFullRecord(BaseModel):
    """The complete longitudinal view - everything about one patient."""
    patient: PatientOut
    visits: list[VisitRecordOut]
    medications: list[MedicationHistoryOut]