Wait-Process -Id 18456 -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe run_phase11_worker.py --forecaster_type ma --seed 1
.\.venv\Scripts\python.exe run_phase11_worker.py --forecaster_type ma --seed 2
.\.venv\Scripts\python.exe run_phase11_worker.py --forecaster_type xgboost --seed 100
.\.venv\Scripts\python.exe run_phase11_worker.py --forecaster_type xgboost --seed 101
