@echo off
echo Setting up Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
echo Setup Complete! Run run.bat to start the server.
