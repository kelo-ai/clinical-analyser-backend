"""
EMR System - independent service, separate database, separate deployment
from the AI Clinical Assistant.

Endpoints:
  POST /patients                         - register a new patient
  GET  /patients/{id}                    - get patient demographics
  GET  /patients/{id}/full-record        - get complete longitudinal history
  POST /doctors                          - register a new doctor

  POST /visits                           - add a visit record (called by AI Clinical Assistant)
  GET  /patients/{id}/visits             - get a patient's visit history

  POST /medications                      - add a medication history entry (called by AI Clinical Assistant)
  GET  /patients/{id}/medications        - get a patient's full medication history

Run with:
    uvicorn main:app --reload --port 8002
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Patient, Doctor, VisitRecord, MedicationHistoryEntry
from schemas import (
    PatientCreate, PatientOut, PatientUpdate, DoctorCreate, DoctorOut,
    VisitRecordCreate, VisitRecordOut,
    MedicationHistoryCreate, MedicationHistoryOut,
    PatientFullRecord,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="EMR System")

# Allows the AI Clinical Assistant (a different service/port) and any
# frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "EMR System running."}


# ============================================================
# PATIENTS
# ============================================================
@app.post("/patients", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(Patient).filter(Patient.phone_number == payload.phone_number).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A patient with this phone number is already registered ({existing.full_name}). "
                   f"Use the login-by-phone endpoint instead of registering again."
        )

    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.get("/patients/login/{phone_number}", response_model=PatientOut)
def login_patient_by_phone(phone_number: str, db: Session = Depends(get_db)):
    """
    Exact-match lookup by phone number - this is the actual 'login' a
    returning patient uses, disambiguating from same-name collisions that
    the fuzzy name search alone can't resolve.
    """
    patient = db.query(Patient).filter(Patient.phone_number == phone_number).first()
    if not patient:
        raise HTTPException(status_code=404, detail="No patient found with this phone number.")
    return patient


@app.patch("/patients/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: str, payload: PatientUpdate, db: Session = Depends(get_db)):
    """
    Updates an existing patient's editable fields (allergies, chronic
    conditions, contact info). This is what lets a doctor revisit a
    returning patient and update their record - the patient's identity
    (id, phone_number) never changes, only these details.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)

    db.commit()
    db.refresh(patient)
    return patient


@app.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patients", response_model=list[PatientOut])
def search_patients(name: str | None = None, db: Session = Depends(get_db)):
    """
    Search patients by name (case-insensitive partial match). If no name
    is given, returns all patients - useful for a doctor searching for
    a patient when starting a new consultation, without needing to
    already know their exact ID.
    """
    query = db.query(Patient)
    if name:
        query = query.filter(Patient.full_name.ilike(f"%{name}%"))
    return query.order_by(Patient.full_name).limit(50).all()


@app.get("/patients/{patient_id}/full-record", response_model=PatientFullRecord)
def get_patient_full_record(patient_id: str, db: Session = Depends(get_db)):
    """
    The complete longitudinal view: patient demographics + every visit +
    every medication ever recorded. This is what a doctor would pull up
    to see a patient's entire history, not just one consultation.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    visits = db.query(VisitRecord).filter(
        VisitRecord.patient_id == patient_id
    ).order_by(VisitRecord.visit_date.desc()).all()

    medications = db.query(MedicationHistoryEntry).filter(
        MedicationHistoryEntry.patient_id == patient_id
    ).order_by(MedicationHistoryEntry.prescribed_date.desc()).all()

    return PatientFullRecord(patient=patient, visits=visits, medications=medications)


# ============================================================
# DOCTORS
# ============================================================
@app.post("/doctors", response_model=DoctorOut)
def create_doctor(payload: DoctorCreate, db: Session = Depends(get_db)):
    existing = db.query(Doctor).filter(Doctor.phone_number == payload.phone_number).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A doctor with this phone number is already registered ({existing.full_name}). "
                   f"Use the login-by-phone endpoint instead of registering again."
        )

    doctor = Doctor(**payload.model_dump())
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return doctor


@app.get("/doctors/login/{phone_number}", response_model=DoctorOut)
def login_doctor_by_phone(phone_number: str, db: Session = Depends(get_db)):
    """Exact-match lookup by phone number - the doctor's actual login."""
    doctor = db.query(Doctor).filter(Doctor.phone_number == phone_number).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="No doctor found with this phone number.")
    return doctor


@app.get("/doctors", response_model=list[DoctorOut])
def search_doctors(name: str | None = None, db: Session = Depends(get_db)):
    """Search doctors by name (case-insensitive partial match)."""
    query = db.query(Doctor)
    if name:
        query = query.filter(Doctor.full_name.ilike(f"%{name}%"))
    return query.order_by(Doctor.full_name).limit(50).all()


# ============================================================
# VISIT RECORDS - integration point with AI Clinical Assistant
# ============================================================
@app.post("/visits", response_model=VisitRecordOut)
def create_visit_record(payload: VisitRecordCreate, db: Session = Depends(get_db)):
    """
    Called by the AI Clinical Assistant when a doctor approves a SOAP note.
    This is the actual integration point between the two systems - the
    AI module produces a SOAP note, and once approved, it gets permanently
    recorded here as part of the patient's history.
    """
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {payload.patient_id} not found in EMR")

    visit = VisitRecord(**payload.model_dump())
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


@app.get("/patients/{patient_id}/visits", response_model=list[VisitRecordOut])
def get_patient_visits(patient_id: str, db: Session = Depends(get_db)):
    visits = db.query(VisitRecord).filter(
        VisitRecord.patient_id == patient_id
    ).order_by(VisitRecord.visit_date.desc()).all()
    return visits


# ============================================================
# MEDICATION HISTORY - integration point with AI Clinical Assistant
# ============================================================
@app.post("/medications", response_model=MedicationHistoryOut)
def add_medication_history(payload: MedicationHistoryCreate, db: Session = Depends(get_db)):
    """
    Called by the AI Clinical Assistant after polypharmacy detection
    extracts medications from an approved consultation.
    """
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {payload.patient_id} not found in EMR")

    entry = MedicationHistoryEntry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.get("/patients/{patient_id}/medications", response_model=list[MedicationHistoryOut])
def get_patient_medications(patient_id: str, db: Session = Depends(get_db)):
    """
    Returns EVERY medication ever recorded for this patient - the full
    history, useful for a doctor checking for interactions with something
    prescribed months or years ago, not just in today's visit.
    """
    medications = db.query(MedicationHistoryEntry).filter(
        MedicationHistoryEntry.patient_id == patient_id
    ).order_by(MedicationHistoryEntry.prescribed_date.desc()).all()
    return medications