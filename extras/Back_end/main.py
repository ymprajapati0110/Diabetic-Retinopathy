from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import auth, patient, scan
import os
from dotenv import load_dotenv
from database import engine, Base
import models

# Create MySQL tables automatically if they don't exist
Base.metadata.create_all(bind=engine)

load_dotenv()

app = FastAPI(title="Medical AI - Diabetic Retinopathy API")

# Configure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded fundus images as static files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# SQLAlchemy handles connection pooling automatically

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(patient.router, prefix="/api/patients", tags=["Patients"])
app.include_router(scan.router, prefix="/api/scans", tags=["Scans"])

@app.get("/")
def root():
    return {"message": "Medical AI API is running"}
