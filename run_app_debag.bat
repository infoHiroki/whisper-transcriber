@echo off
echo Starting Whisper Transcription Tool...
cd /d %~dp0
call venv\Scripts\activate
echo Launching Streamlit...
streamlit run app.py
pause
