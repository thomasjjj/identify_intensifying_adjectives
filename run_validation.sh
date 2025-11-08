#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Installing dependencies..."
python -m pip install -r requirements.txt

echo "[2/3] Running end-to-end pipeline..."
python -m tuning.orchestrator --seed 42 --run-tests

echo "[3/3] Workflow completed successfully."
