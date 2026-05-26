@echo off
cd /d "%~dp0"
echo Testing Gemini API...
python test_api.py
if errorlevel 1 (
    echo API test failed. Fix GEMINI_API_KEY in .env then retry.
    pause
    exit /b 1
)
echo Starting Streamlit...
python -m streamlit run app.py
