"""
AI Clinical Assistant - Speech-to-Text Transcription Module

Endpoints:
  POST   /consultations                     - create a new consultation
  POST   /consultations/{id}/audio/upload   - upload a pre-recorded audio file
  WS     /consultations/{id}/audio/stream   - live recording, streamed to Deepgram
  GET    /consultations/{id}/transcript     - get diarized transcript segments
  PATCH  /consultations/{id}/speakers       - map speaker_label -> doctor/patient

Run with:
    uvicorn main:app --reload --port 8001
"""

import json
import asyncio
import websockets
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import (
    Consultation, TranscriptSegment, SOAPNote, PolypharmacyCheck,
    DecisionSupportInsight, ConsultationSummary, PrescriptionInsight, WellnessCheckin,
)
from schemas import (
    ConsultationCreate, ConsultationOut, TranscriptSegmentOut, SpeakerMapping,
    SOAPNoteOut, SOAPNoteReview, PolypharmacyCheckOut, PolypharmacyReview,
    DecisionSupportOut, DecisionSupportReview,
    ConsultationSummaryOut, PrescriptionInsightOut,
    WellnessCheckinCreate, WellnessCheckinOut,
)
from deepgram_client import transcribe_file, DEEPGRAM_API_KEY, WS_URL
from soap_generator import generate_soap_note
from medication_extractor import extract_medications
from rxnorm_client import check_interactions
from emr_client import push_visit_record, push_medication_history
from decision_support import generate_decision_support
from consultation_summary import generate_consultation_summary
from prescription_insights import generate_prescription_insight

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Clinical Assistant - Transcription Module")


# ============================================================
# CREATE CONSULTATION
# ============================================================
@app.post("/consultations", response_model=ConsultationOut)
def create_consultation(payload: ConsultationCreate, db: Session = Depends(get_db)):
    consultation = Consultation(
        doctor_id=payload.doctor_id,
        patient_id=payload.patient_id,
        audio_source=payload.audio_source,
        status="created",
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return consultation


@app.get("/doctors/{doctor_id}/consultations", response_model=list[ConsultationOut])
def list_doctor_consultations(doctor_id: str, db: Session = Depends(get_db)):
    """
    Lists every consultation for a given doctor, most recent first - what the
    Doctor Dashboard's consultation list actually calls.
    """
    consultations = (
        db.query(Consultation)
        .filter(Consultation.doctor_id == doctor_id)
        .order_by(Consultation.created_at.desc())
        .all()
    )
    return consultations


# ============================================================
# UPLOAD PATH: pre-recorded audio file
# ============================================================
@app.post("/consultations/{consultation_id}/audio/upload")
async def upload_audio(consultation_id: str, audio: UploadFile = File(...), db: Session = Depends(get_db)):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    audio_bytes = await audio.read()

    consultation.status = "transcribing"
    db.commit()

    try:
        content_type = audio.content_type or "audio/*"
        segments = transcribe_file(audio_bytes, content_type=content_type)
    except Exception as e:
        consultation.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Store each diarized segment
    for seg in segments:
        db.add(TranscriptSegment(
            consultation_id=consultation.id,
            speaker_label=seg["speaker_label"],
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            text=seg["text"],
        ))

    consultation.status = "transcribed"
    db.commit()

    return {"status": "transcribed", "segment_count": len(segments)}


# ============================================================
# LIVE PATH: real-time streaming during consultation
# ============================================================
@app.websocket("/consultations/{consultation_id}/audio/stream")
async def stream_audio(websocket: WebSocket, consultation_id: str):
    """
    Client (Flutter app) connects here and streams raw PCM16 audio chunks
    as they're recorded. This function relays those chunks to Deepgram's
    live API and forwards transcription results back to the client in
    real time. Segments are also saved to the DB as they arrive.
    """
    await websocket.accept()

    db = next(get_db())
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        await websocket.close(code=4004, reason="Consultation not found")
        return

    consultation.status = "transcribing"
    db.commit()

    try:
        async with websockets.connect(
            WS_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        ) as deepgram_ws:

            async def relay_client_to_deepgram():
                """Forward audio chunks from the Flutter client to Deepgram."""
                try:
                    while True:
                        chunk = await websocket.receive_bytes()
                        await deepgram_ws.send(chunk)
                except WebSocketDisconnect:
                    await deepgram_ws.close()

            async def relay_deepgram_to_client():
                """Forward Deepgram's live transcription results back to the client,
                and persist finished utterances to the database."""
                async for message in deepgram_ws:
                    data = json.loads(message)

                    # Deepgram sends interim + final results; only save finalized utterances
                    if data.get("is_final") and data.get("channel"):
                        alt = data["channel"]["alternatives"][0]
                        text = alt.get("transcript", "")
                        if text.strip():
                            speaker = data.get("channel", {}).get("alternatives", [{}])[0].get("speaker", 0)
                            start = data.get("start", 0)
                            duration = data.get("duration", 0)

                            segment = TranscriptSegment(
                                consultation_id=consultation.id,
                                speaker_label=f"speaker_{speaker}",
                                start_time=start,
                                end_time=start + duration,
                                text=text,
                            )
                            db.add(segment)
                            db.commit()

                    # Forward raw Deepgram message to the client for live display
                    await websocket.send_text(message)

            await asyncio.gather(relay_client_to_deepgram(), relay_deepgram_to_client())

    except WebSocketDisconnect:
        pass
    finally:
        consultation.status = "transcribed"
        db.commit()
        db.close()


# ============================================================
# GET TRANSCRIPT
# ============================================================
@app.get("/consultations/{consultation_id}/transcript", response_model=list[TranscriptSegmentOut])
def get_transcript(consultation_id: str, db: Session = Depends(get_db)):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.consultation_id == consultation_id)
        .order_by(TranscriptSegment.start_time)
        .all()
    )
    return segments


# ============================================================
# MAP SPEAKERS: speaker_0/speaker_1 -> doctor/patient
# ============================================================
@app.patch("/consultations/{consultation_id}/speakers")
def map_speakers(consultation_id: str, payload: SpeakerMapping, db: Session = Depends(get_db)):
    """
    payload.mapping example: {"speaker_0": "doctor", "speaker_1": "patient"}
    Applies the mapping to every segment in this consultation.
    """
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    segments = db.query(TranscriptSegment).filter(TranscriptSegment.consultation_id == consultation_id).all()

    updated = 0
    for seg in segments:
        if seg.speaker_label in payload.mapping:
            seg.speaker_role = payload.mapping[seg.speaker_label]
            updated += 1

    db.commit()
    return {"updated_segments": updated}


@app.get("/consultations/{consultation_id}/speaker-summary")
def get_speaker_summary(consultation_id: str, db: Session = Depends(get_db)):
    """
    Diagnostic endpoint: shows how many distinct speaker_labels Deepgram
    actually detected, and how many segments fall under each. If everything
    shows up as a single label (e.g. only 'speaker_0'), that means Deepgram's
    diarization failed to distinguish the two voices in this recording -
    no amount of remapping fixes that; a clearer/cleaner audio source is
    needed instead.
    """
    segments = db.query(TranscriptSegment).filter(TranscriptSegment.consultation_id == consultation_id).all()

    label_counts: dict[str, int] = {}
    for seg in segments:
        label_counts[seg.speaker_label] = label_counts.get(seg.speaker_label, 0) + 1

    return {
        "total_segments": len(segments),
        "distinct_speakers_detected": len(label_counts),
        "segments_per_speaker": label_counts,
        "diarization_likely_failed": len(label_counts) <= 1 and len(segments) > 1,
    }


@app.patch("/consultations/{consultation_id}/transcript/{segment_id}/role")
def correct_segment_role(consultation_id: str, segment_id: str, payload: dict, db: Session = Depends(get_db)):
    """
    Corrects a single segment's speaker_role, for when the bulk mapping
    (based on Deepgram's speaker_label) got individual lines wrong -
    common when diarization confidence is low mid-conversation, even if
    it mostly got the two voices right.

    payload example: {"role": "doctor"} or {"role": "patient"}
    """
    segment = db.query(TranscriptSegment).filter(
        TranscriptSegment.id == segment_id,
        TranscriptSegment.consultation_id == consultation_id,
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    role = payload.get("role")
    if role not in ("doctor", "patient"):
        raise HTTPException(status_code=400, detail="role must be 'doctor' or 'patient'")

    segment.speaker_role = role
    db.commit()
    return {"segment_id": segment_id, "speaker_role": role}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Clinical Assistant transcription module running."}


# ============================================================
# SOAP NOTE GENERATION
# ============================================================
@app.post("/consultations/{consultation_id}/soap-note/generate", response_model=SOAPNoteOut)
def generate_note(consultation_id: str, db: Session = Depends(get_db)):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.consultation_id == consultation_id)
        .order_by(TranscriptSegment.start_time)
        .all()
    )

    if not segments:
        raise HTTPException(status_code=400, detail="No transcript available for this consultation yet")

    # Check that speaker roles have been assigned - SOAP quality depends on this
    unmapped = [s for s in segments if not s.speaker_role]
    if unmapped:
        raise HTTPException(
            status_code=400,
            detail=f"{len(unmapped)} segment(s) have no speaker role assigned. "
                   f"Call PATCH /consultations/{consultation_id}/speakers first.",
        )

    segment_dicts = [
        {"speaker_role": s.speaker_role, "speaker_label": s.speaker_label, "text": s.text}
        for s in segments
    ]

    try:
        soap_data = generate_soap_note(segment_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {e}")

    # Replace any existing draft (e.g. regenerating after an edit to the transcript)
    existing = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    note = SOAPNote(
        consultation_id=consultation_id,
        subjective=soap_data["subjective"],
        objective=soap_data["objective"],
        assessment=soap_data["assessment"],
        plan=soap_data["plan"],
        status="pending_review",
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return note


# ============================================================
# GET SOAP NOTE
# ============================================================
@app.get("/consultations/{consultation_id}/soap-note", response_model=SOAPNoteOut)
def get_soap_note(consultation_id: str, db: Session = Depends(get_db)):
    note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="No SOAP note generated yet for this consultation")
    return note


# ============================================================
# DOCTOR REVIEW: approve as-is, approve with edits, or reject
# ============================================================
@app.patch("/consultations/{consultation_id}/soap-note/review", response_model=SOAPNoteOut)
def review_soap_note(consultation_id: str, payload: SOAPNoteReview, db: Session = Depends(get_db)):
    note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="No SOAP note found for this consultation")

    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    if payload.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    if payload.action == "reject":
        note.status = "rejected"
        note.reviewed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(note)
        return note

    # Approve - track whether the doctor changed anything vs accepted the AI draft verbatim
    any_edit = False
    if payload.subjective is not None:
        note.doctor_subjective = payload.subjective
        any_edit = True
    if payload.objective is not None:
        note.doctor_objective = payload.objective
        any_edit = True
    if payload.assessment is not None:
        note.doctor_assessment = payload.assessment
        any_edit = True
    if payload.plan is not None:
        note.doctor_plan = payload.plan
        any_edit = True

    note.status = "edited" if any_edit else "approved"
    note.reviewed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(note)

    # Push into the EMR as a permanent visit record. Uses the doctor's edited
    # text if provided, otherwise the original AI draft - this is the
    # integration point between the AI Clinical Assistant and the EMR System.
    push_visit_record(
        patient_id=consultation.patient_id,
        doctor_id=consultation.doctor_id,
        subjective=note.doctor_subjective or note.subjective,
        objective=note.doctor_objective or note.objective,
        assessment=note.doctor_assessment or note.assessment,
        plan=note.doctor_plan or note.plan,
        source_consultation_id=consultation_id,
    )

    return note


# ============================================================
# POLYPHARMACY DETECTION (async - never blocks the caller)
# ============================================================

def _run_polypharmacy_check(consultation_id: str):
    """
    The actual work: extract medications, look them up in RxNorm, check
    interactions. Runs in a background task AFTER the triggering request
    has already returned to the client - the doctor's app is never
    waiting on this call.

    Opens its own DB session since background tasks run outside the
    request's dependency-injected session.
    """
    import json
    from database import SessionLocal

    db = SessionLocal()
    try:
        check = db.query(PolypharmacyCheck).filter(
            PolypharmacyCheck.consultation_id == consultation_id
        ).first()
        if not check:
            return

        check.status = "processing"
        db.commit()

        note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
        if not note:
            check.status = "failed"
            check.error_message = "No SOAP note found - generate one before running this check."
            db.commit()
            return

        # Use doctor-edited text if available (more accurate than the raw AI draft),
        # otherwise fall back to the AI-generated version.
        subjective = note.doctor_subjective or note.subjective or ""
        plan = note.doctor_plan or note.plan or ""

        medications = extract_medications(subjective, plan)
        check.medications_found = json.dumps(medications)

        if len(medications) >= 2:
            result = check_interactions(medications)
            check.interactions_found = json.dumps(result)
        else:
            # Fewer than 2 medications means no interaction is possible -
            # this is a normal outcome, not an error.
            check.interactions_found = json.dumps({
                "resolved_medications": [],
                "unresolved_medications": [],
                "interactions": [],
            })

        check.status = "completed"
        check.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        check.status = "failed"
        check.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/consultations/{consultation_id}/polypharmacy/analyze", response_model=PolypharmacyCheckOut)
def trigger_polypharmacy_check(consultation_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Triggers the check and returns IMMEDIATELY with status='pending'.
    The actual extraction + RxNorm lookups happen in the background.
    The client should poll GET /polypharmacy afterward to see progress.
    """
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if not note:
        raise HTTPException(status_code=400, detail="Generate a SOAP note before running polypharmacy detection")

    # Replace any existing check (e.g. re-running after the doctor edits the SOAP note)
    existing = db.query(PolypharmacyCheck).filter(PolypharmacyCheck.consultation_id == consultation_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    check = PolypharmacyCheck(consultation_id=consultation_id, status="pending")
    db.add(check)
    db.commit()
    db.refresh(check)

    # This line returns control to the client right away; the function
    # passed here runs afterward, in the background.
    background_tasks.add_task(_run_polypharmacy_check, consultation_id)

    return check


@app.get("/consultations/{consultation_id}/polypharmacy", response_model=PolypharmacyCheckOut)
def get_polypharmacy_check(consultation_id: str, db: Session = Depends(get_db)):
    """
    Client polls this endpoint (e.g. every 2 seconds) to check progress.
    status will be: pending -> processing -> completed (or failed).
    """
    check = db.query(PolypharmacyCheck).filter(PolypharmacyCheck.consultation_id == consultation_id).first()
    if not check:
        raise HTTPException(status_code=404, detail="No polypharmacy check has been triggered for this consultation")
    return check


@app.patch("/consultations/{consultation_id}/polypharmacy/review", response_model=PolypharmacyCheckOut)
def review_polypharmacy_check(consultation_id: str, payload: PolypharmacyReview, db: Session = Depends(get_db)):
    import json as json_module

    check = db.query(PolypharmacyCheck).filter(PolypharmacyCheck.consultation_id == consultation_id).first()
    if not check:
        raise HTTPException(status_code=404, detail="No polypharmacy check found for this consultation")

    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    if payload.action not in ("approve", "dismiss"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'dismiss'")

    check.review_status = "approved" if payload.action == "approve" else "dismissed"
    check.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(check)

    # On approval, push each identified medication into the EMR's permanent
    # medication history for this patient.
    if payload.action == "approve" and check.medications_found:
        medications = json_module.loads(check.medications_found)
        interactions_data = json_module.loads(check.interactions_found) if check.interactions_found else {}
        resolved = {m["name"]: m["rxcui"] for m in interactions_data.get("resolved_medications", [])}

        for med_name in medications:
            push_medication_history(
                patient_id=consultation.patient_id,
                medication_name=med_name,
                rxcui=resolved.get(med_name),
                source_consultation_id=consultation_id,
            )

    return check


# ============================================================
# AI DECISION SUPPORT (async - advisory only, doctor decides)
# ============================================================

def _run_decision_support(consultation_id: str):
    """
    Generates advisory suggestions in the background. Requires an approved
    or edited SOAP note to exist first, since decision support reasons over
    the doctor's own documentation, not the raw transcript - this ensures
    it's reacting to what the doctor actually decided, not second-guessing
    an unreviewed AI draft.
    """
    import json as json_module
    from database import SessionLocal

    db = SessionLocal()
    try:
        insight = db.query(DecisionSupportInsight).filter(
            DecisionSupportInsight.consultation_id == consultation_id
        ).first()
        if not insight:
            return

        insight.status = "processing"
        db.commit()

        note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
        if not note:
            insight.status = "failed"
            insight.error_message = "No SOAP note found - generate and approve one first."
            db.commit()
            return

        subjective = note.doctor_subjective or note.subjective or ""
        objective = note.doctor_objective or note.objective or ""
        assessment = note.doctor_assessment or note.assessment or ""
        plan = note.doctor_plan or note.plan or ""

        result = generate_decision_support(subjective, objective, assessment, plan)

        insight.differential_considerations = json_module.dumps(result["differential_considerations"])
        insight.documentation_gaps = json_module.dumps(result["documentation_gaps"])
        insight.suggested_followup_questions = json_module.dumps(result["suggested_followup_questions"])
        insight.risk_flags = json_module.dumps(result["risk_flags"])
        insight.status = "completed"
        insight.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        insight.status = "failed"
        insight.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/consultations/{consultation_id}/decision-support/analyze", response_model=DecisionSupportOut)
def trigger_decision_support(consultation_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Requires the SOAP note to already be approved/edited (status != pending_review),
    since decision support reasons over the doctor's finalized documentation.
    Returns immediately with status='pending'; poll GET for results.
    """
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if not note:
        raise HTTPException(status_code=400, detail="Generate a SOAP note before requesting decision support")
    if note.status == "pending_review":
        raise HTTPException(
            status_code=400,
            detail="Review and approve/edit the SOAP note before requesting decision support - "
                   "this module reasons over your finalized documentation, not an unreviewed AI draft."
        )

    existing = db.query(DecisionSupportInsight).filter(
        DecisionSupportInsight.consultation_id == consultation_id
    ).first()
    if existing:
        db.delete(existing)
        db.commit()

    insight = DecisionSupportInsight(consultation_id=consultation_id, status="pending")
    db.add(insight)
    db.commit()
    db.refresh(insight)

    background_tasks.add_task(_run_decision_support, consultation_id)

    return insight


@app.get("/consultations/{consultation_id}/decision-support", response_model=DecisionSupportOut)
def get_decision_support(consultation_id: str, db: Session = Depends(get_db)):
    insight = db.query(DecisionSupportInsight).filter(
        DecisionSupportInsight.consultation_id == consultation_id
    ).first()
    if not insight:
        raise HTTPException(status_code=404, detail="No decision support analysis has been triggered for this consultation")
    return insight


@app.patch("/consultations/{consultation_id}/decision-support/review", response_model=DecisionSupportOut)
def review_decision_support(consultation_id: str, payload: DecisionSupportReview, db: Session = Depends(get_db)):
    """
    The doctor's final word on these suggestions. 'acknowledge' means they
    reviewed them (regardless of whether they acted on any); 'dismiss' means
    they're not relevant. Either way, this module never modifies the SOAP
    note or EMR directly - any resulting changes are the doctor's own,
    made through the normal SOAP note edit/review endpoint.
    """
    insight = db.query(DecisionSupportInsight).filter(
        DecisionSupportInsight.consultation_id == consultation_id
    ).first()
    if not insight:
        raise HTTPException(status_code=404, detail="No decision support analysis found for this consultation")

    if payload.action not in ("acknowledge", "dismiss"):
        raise HTTPException(status_code=400, detail="action must be 'acknowledge' or 'dismiss'")

    insight.review_status = "acknowledged" if payload.action == "acknowledge" else "dismissed"
    insight.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(insight)
    return insight


# ============================================================
# CONSULTATION SUMMARIES (Doctor Dashboard) - async
# ============================================================

def _run_consultation_summary(consultation_id: str):
    from database import SessionLocal
    import json as json_module

    db = SessionLocal()
    try:
        summary = db.query(ConsultationSummary).filter(
            ConsultationSummary.consultation_id == consultation_id
        ).first()
        if not summary:
            return

        summary.status = "processing"
        db.commit()

        note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
        if not note:
            summary.status = "failed"
            summary.error_message = "No SOAP note found - generate one first."
            db.commit()
            return

        subjective = note.doctor_subjective or note.subjective or ""
        objective = note.doctor_objective or note.objective or ""
        assessment = note.doctor_assessment or note.assessment or ""
        plan = note.doctor_plan or note.plan or ""

        result = generate_consultation_summary(subjective, objective, assessment, plan)

        summary.summary_text = result["summary_text"]
        summary.key_points = json_module.dumps(result["key_points"])
        summary.status = "completed"
        summary.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        summary.status = "failed"
        summary.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/consultations/{consultation_id}/summary/generate", response_model=ConsultationSummaryOut)
def trigger_consultation_summary(consultation_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Generates a short doctor-facing recap. Works off the SOAP note (any status -
    doesn't require approval first, since this is just a quick scan aid, not
    something that gets persisted to the EMR)."""
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    note = db.query(SOAPNote).filter(SOAPNote.consultation_id == consultation_id).first()
    if not note:
        raise HTTPException(status_code=400, detail="Generate a SOAP note before requesting a summary")

    existing = db.query(ConsultationSummary).filter(
        ConsultationSummary.consultation_id == consultation_id
    ).first()
    if existing:
        db.delete(existing)
        db.commit()

    summary = ConsultationSummary(consultation_id=consultation_id, status="pending")
    db.add(summary)
    db.commit()
    db.refresh(summary)

    background_tasks.add_task(_run_consultation_summary, consultation_id)

    return summary


@app.get("/consultations/{consultation_id}/summary", response_model=ConsultationSummaryOut)
def get_consultation_summary(consultation_id: str, db: Session = Depends(get_db)):
    summary = db.query(ConsultationSummary).filter(
        ConsultationSummary.consultation_id == consultation_id
    ).first()
    if not summary:
        raise HTTPException(status_code=404, detail="No summary generated yet for this consultation")
    return summary


@app.get("/patients/{patient_id}/consultation-summaries", response_model=list[ConsultationSummaryOut])
def get_patient_consultation_summaries(patient_id: str, db: Session = Depends(get_db)):
    """
    All consultation summaries for a patient across every visit - what the
    Patient Dashboard shows regardless of whether medications were involved.
    A visit with no prescriptions is still worth showing (what was
    discussed, what was decided) rather than leaving the dashboard empty.
    """
    summaries = (
        db.query(ConsultationSummary)
        .join(Consultation, ConsultationSummary.consultation_id == Consultation.id)
        .filter(Consultation.patient_id == patient_id)
        .filter(ConsultationSummary.status == "completed")
        .order_by(ConsultationSummary.generated_at.desc())
        .all()
    )
    return summaries


# ============================================================
# PRESCRIPTION INSIGHTS (Patient Dashboard) - async, per medication
# ============================================================

def _run_prescription_insight(insight_id: str, medication_name: str):
    from database import SessionLocal
    import json as json_module

    db = SessionLocal()
    try:
        insight = db.query(PrescriptionInsight).filter(PrescriptionInsight.id == insight_id).first()
        if not insight:
            return

        insight.status = "processing"
        db.commit()

        result = generate_prescription_insight(medication_name)

        insight.plain_language_purpose = result["plain_language_purpose"]
        insight.general_usage_notes = result["general_usage_notes"]
        insight.common_things_to_know = json_module.dumps(result["common_things_to_know"])
        insight.status = "completed"
        insight.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        insight.status = "failed"
        insight.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/consultations/{consultation_id}/prescription-insights/generate", response_model=list[PrescriptionInsightOut])
def trigger_prescription_insights(consultation_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Generates one insight per medication found in this consultation's approved
    polypharmacy check. Patient-facing - requires the polypharmacy check to be
    approved first, since we only want to explain medications the doctor has
    confirmed, not an unreviewed AI extraction.
    """
    import json as json_module

    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    check = db.query(PolypharmacyCheck).filter(PolypharmacyCheck.consultation_id == consultation_id).first()
    if not check or check.review_status != "approved":
        raise HTTPException(
            status_code=400,
            detail="Requires an approved polypharmacy check first - "
                   "run and approve POST /polypharmacy/analyze before requesting patient insights."
        )

    medications = json_module.loads(check.medications_found) if check.medications_found else []
    if not medications:
        raise HTTPException(status_code=400, detail="No medications found for this consultation")

    # Remove any existing insights for this consultation before regenerating
    db.query(PrescriptionInsight).filter(
        PrescriptionInsight.consultation_id == consultation_id
    ).delete()
    db.commit()

    created_insights = []
    for med_name in medications:
        insight = PrescriptionInsight(
            consultation_id=consultation_id,
            patient_id=consultation.patient_id,
            medication_name=med_name,
            status="pending",
        )
        db.add(insight)
        db.commit()
        db.refresh(insight)
        created_insights.append(insight)

        background_tasks.add_task(_run_prescription_insight, insight.id, med_name)

    return created_insights


@app.get("/consultations/{consultation_id}/prescription-insights", response_model=list[PrescriptionInsightOut])
def get_prescription_insights(consultation_id: str, db: Session = Depends(get_db)):
    insights = db.query(PrescriptionInsight).filter(
        PrescriptionInsight.consultation_id == consultation_id
    ).all()
    if not insights:
        raise HTTPException(status_code=404, detail="No prescription insights generated yet for this consultation")
    return insights


@app.get("/patients/{patient_id}/prescription-insights", response_model=list[PrescriptionInsightOut])
def get_patient_prescription_insights(patient_id: str, db: Session = Depends(get_db)):
    """
    All prescription insights for a patient across every consultation - what
    the Patient Dashboard would actually display (not scoped to one visit).
    """
    insights = db.query(PrescriptionInsight).filter(
        PrescriptionInsight.patient_id == patient_id
    ).order_by(PrescriptionInsight.generated_at.desc()).all()
    return insights


# ============================================================
# WELLNESS CHECK-INS ("I'm completely fine" button)
# ============================================================

@app.post("/wellness-checkins", response_model=WellnessCheckinOut)
def create_wellness_checkin(payload: WellnessCheckinCreate, db: Session = Depends(get_db)):
    """
    Patient taps 'I'm completely fine' -> creates a lightweight notification
    for their doctor. Not tied to a specific consultation.
    """
    checkin = WellnessCheckin(
        patient_id=payload.patient_id,
        doctor_id=payload.doctor_id,
        message=payload.message,
    )
    db.add(checkin)
    db.commit()
    db.refresh(checkin)
    return checkin


@app.get("/doctors/{doctor_id}/wellness-checkins", response_model=list[WellnessCheckinOut])
def get_doctor_wellness_checkins(doctor_id: str, unread_only: bool = False, db: Session = Depends(get_db)):
    """
    Doctor's notification list - every patient check-in addressed to them,
    most recent first. Pass unread_only=true to only get unread ones (for
    a notification badge count).
    """
    query = db.query(WellnessCheckin).filter(WellnessCheckin.doctor_id == doctor_id)
    if unread_only:
        query = query.filter(WellnessCheckin.is_read == "unread")
    return query.order_by(WellnessCheckin.created_at.desc()).all()


@app.patch("/wellness-checkins/{checkin_id}/read", response_model=WellnessCheckinOut)
def mark_checkin_read(checkin_id: str, db: Session = Depends(get_db)):
    checkin = db.query(WellnessCheckin).filter(WellnessCheckin.id == checkin_id).first()
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")
    checkin.is_read = "read"
    db.commit()
    db.refresh(checkin)
    return checkin


@app.get("/patients/{patient_id}/most-recent-doctor")
def get_most_recent_doctor(patient_id: str, db: Session = Depends(get_db)):
    """
    Finds the doctor from the patient's most recent consultation - used to
    auto-address the 'I'm fine' check-in without the patient needing to
    know/select which doctor treated them.
    """
    consultation = (
        db.query(Consultation)
        .filter(Consultation.patient_id == patient_id)
        .order_by(Consultation.created_at.desc())
        .first()
    )
    if not consultation:
        raise HTTPException(status_code=404, detail="No prior consultations found for this patient")
    return {"doctor_id": consultation.doctor_id}