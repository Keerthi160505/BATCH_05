# Dynamic Green AI Evaluation Framework - One-Command Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "========================================================" -ForegroundColor Green
Write-Host " Starting Dynamic Green AI Evaluation Framework..." -ForegroundColor Cyan
Write-Host " Opening dashboard at http://localhost:8501" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m streamlit run app/dashboard.py
