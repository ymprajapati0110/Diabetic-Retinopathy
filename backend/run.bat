@echo off
echo Starting FastAPI Backend...
call venv\Scripts\activate.bat
python -m uvicorn main:app --reload --port 8000
