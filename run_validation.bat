@echo off
setlocal

echo [1/5] Installing dependencies...
python -m pip install -r requirements.txt || goto :error

echo [2/3] Running end-to-end pipeline...
python -m tuning.orchestrator --seed 42 --run-tests || goto :error

echo [3/3] Workflow completed successfully.
goto :eof

:error
echo Step failed with error %errorlevel%.
exit /b %errorlevel%
