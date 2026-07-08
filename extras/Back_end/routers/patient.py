from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from schemas import PatientCreate, Patient as PatientSchema
from models import Patient, User
from database import get_db
from routers.auth import get_current_verified_doctor
import uuid

router = APIRouter()

@router.get("/", response_model=List[PatientSchema])
def get_patients(current_user: User = Depends(get_current_verified_doctor), db: Session = Depends(get_db)):
    patients = db.query(Patient).filter(Patient.doctor_id == current_user.id).all()
    return patients


@router.post("/", response_model=PatientSchema)
def create_patient(patient: PatientCreate, current_user: User = Depends(get_current_verified_doctor), db: Session = Depends(get_db)):
    patient_ref_code = f"PT-{str(uuid.uuid4())[:8].upper()}"
    
    db_patient = Patient(
        doctor_id=current_user.id,
        patient_reference_code=patient_ref_code,
        age=patient.age,
        gender=patient.gender,
        medical_history=patient.medical_history
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    return db_patient


@router.get("/{patient_id}", response_model=PatientSchema)
def get_patient_by_id(patient_id: int, current_user: User = Depends(get_current_verified_doctor), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or not assigned to this doctor.")
    return patient
