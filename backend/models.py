from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    medical_license = Column(String(255), nullable=False)
    role = Column(String(50), default="pending")

    patients = relationship("Patient", back_populates="doctor")
    scans = relationship("Scan", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    patient_reference_code = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(50), nullable=False)
    medical_history = Column(JSON, default={})

    doctor = relationship("User", back_populates="patients")
    scans = relationship("Scan", back_populates="patient")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"))
    eye_side = Column(String(50), nullable=False)
    raw_image_s3_url = Column(String(500), nullable=False)
    gradcam_image_s3_url = Column(String(500), nullable=True)
    dr_prediction_level = Column(Integer, nullable=True)
    regression_score = Column(Float, nullable=True)
    status = Column(String(50), default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="scans")
    doctor = relationship("User", back_populates="scans")
