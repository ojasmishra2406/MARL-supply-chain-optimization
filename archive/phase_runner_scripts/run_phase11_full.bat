@echo off
cd /d %~dp0
.\.venv\Scripts\python.exe run_phase11_parallel.py > phase11_train.log 2>&1
