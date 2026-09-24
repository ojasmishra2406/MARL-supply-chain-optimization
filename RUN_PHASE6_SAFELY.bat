@echo off
title MARL Phase 6 - Safe Native Runner
echo ========================================================
echo MARL Phase 6 - Safe Native Runner
echo ========================================================
echo This window is running independently of the AI agent.
echo.
echo IMPORTANT: 
echo 1. Do NOT close this window.
echo 2. Ensure your laptop DOES NOT GO TO SLEEP.
echo.
echo Starting the parallel queue now...
echo.

call .\.venv\Scripts\activate.bat
python -u run_mappo_comm_parallel.py

echo.
echo ========================================================
echo ALL PHASE 6 RUNS COMPLETED SUCCESSFULLY.
echo You can now close this window.
pause
