from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from schemas import ScanResponse
from models import Scan, User
from database import get_db
from routers.auth import get_current_verified_doctor
from inference_service import AI_Agent
from datetime import datetime
import uuid
import os

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

@router.get("/patient/{patient_id}", response_model=List[ScanResponse])
def get_scans_by_patient(
    patient_id: int,
    current_user: User = Depends(get_current_verified_doctor),
    db: Session = Depends(get_db)
):
    scans = db.query(Scan).filter(Scan.patient_id == patient_id, Scan.doctor_id == current_user.id).all()
    return scans


@router.post("/upload", response_model=ScanResponse)
def upload_scan(
    background_tasks: BackgroundTasks,
    patient_id: int = Form(...),
    eye_side: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_verified_doctor),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_bytes = file.file.read()
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    raw_image_url = f"{BASE_URL}/uploads/{filename}"

    db_scan = Scan(
        patient_id=patient_id,
        doctor_id=current_user.id,
        eye_side=eye_side,
        raw_image_s3_url=raw_image_url,
        status="processing"
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)

    # Pass the scan.id (integer) to the background task
    background_tasks.add_task(AI_Agent.process_image, file_bytes, db_scan.id)

    return db_scan


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan_details(
    scan_id: int,
    current_user: User = Depends(get_current_verified_doctor),
    db: Session = Depends(get_db)
):
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.doctor_id == current_user.id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    return scan
