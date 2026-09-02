import os
import sys

# Ensure backend directory and project root are always on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import auth, patient, scan
from dotenv import load_dotenv
from database import engine, Base
import models
from inference_service import AI_Agent

# Create database tables automatically if they don't exist
try:
    Base.metadata.create_all(bind=engine)
except Exception as db_err:
    print(f"[DB INIT ERROR] {db_err}")

load_dotenv()

app = FastAPI(title="Medical AI - Diabetic Retinopathy API")

# Configure CORS (Allow all origins for cloud deployment like Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded fundus images as static files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def root():
    return {"status": "online", "service": "Retinal Diagnostic AI API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(patient.router, prefix="/api/patients", tags=["Patients"])
app.include_router(scan.router, prefix="/api/scans", tags=["Scans"])

@app.post("/api/scans/quick-diagnose")
async def quick_diagnose(file: UploadFile = File(...), eye_side: str = Form("auto")):
    if not file.content_type.startswith("image/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    file_bytes = await file.read()
    results = await AI_Agent.predict_quick(file_bytes, eye_side=eye_side, filename=file.filename)
    return results

@app.get("/")
def root():
    return {"message": "Medical AI API is running"}
