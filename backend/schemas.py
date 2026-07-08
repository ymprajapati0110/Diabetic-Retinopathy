from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime

# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    medical_license: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: str
    medical_license: str
    role: str

    model_config = ConfigDict(from_attributes=True)

class UserInDB(UserResponse):
    hashed_password: str

# Patient Schemas
class PatientCreate(BaseModel):
    age: int
    gender: str
    medical_history: Dict[str, Any] = {}

class Patient(BaseModel):
    id: int
    doctor_id: int
    patient_reference_code: str
    age: int
    gender: str
    medical_history: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)

# Scan Schemas
class ScanResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    eye_side: str
    raw_image_s3_url: str
    gradcam_image_s3_url: Optional[str] = None
    dr_prediction_level: Optional[int] = None
    regression_score: Optional[float] = None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
