@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.13 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 (
  echo Python not found or virtual environment creation failed.
  echo Install Python 3.13 and retry.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check internet and Python version.
  pause
  exit /b 1
)
python scripts\verify_artifacts.py
if errorlevel 1 (
  echo Saved model verification failed. See error above.
  pause
  exit /b 1
)
streamlit run app.py
pause
