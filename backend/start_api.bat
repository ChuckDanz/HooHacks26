@echo off
REM Start the FitCheck API locally (needs GPU venv, Aerospike must be running in Docker)
REM Usage: double-click or run from cmd

set AEROSPIKE_HOST=localhost
set AEROSPIKE_PORT=3000
set AEROSPIKE_NAMESPACE=test
set PIPELINE_DIR=C:\Projects\HooHacks26\size_vton
set PYTHON_BIN=C:\Projects\HooHacks26\venv\Scripts\python.exe

cd /d C:\Projects\HooHacks26\backend
%PYTHON_BIN% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
