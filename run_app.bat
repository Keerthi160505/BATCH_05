@echo off
title Dynamic Green AI Evaluation Framework
cd /d "%~dp0"
echo ========================================================
echo  Starting Dynamic Green AI Evaluation Framework...
echo  Opening dashboard at http://localhost:8501
echo ========================================================
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
pause
