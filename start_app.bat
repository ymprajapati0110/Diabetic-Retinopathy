@echo off
echo Starting Medical AI Diabetic Retinopathy System...

:: Start the FastAPI backend in a new command prompt window
start "FastAPI Backend" cmd /k "cd backend && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

:: Start the Next.js frontend in a new command prompt window
start "Next.js Frontend" cmd /k "cd frontend && npm run dev"

echo Both servers are starting up!
echo Backend API will be available at: http://localhost:8000
echo Frontend UI will be available at:  http://localhost:3000
pause
