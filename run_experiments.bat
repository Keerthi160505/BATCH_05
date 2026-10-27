@echo off
title Dynamic Green AI Benchmark Runner
cd /d "%~dp0"
echo ========================================================
echo  Running Green AI Benchmarks across all datasets...
echo ========================================================
.\.venv\Scripts\python.exe run_experiments.py --dataset all --runs 1 --alpha 0.5
echo ========================================================
echo  Benchmark complete. Check results/tables/ and results/figures/
echo ========================================================
pause
