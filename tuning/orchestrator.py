#!/usr/bin/env python3
"""One-click pipeline runner for fetching data, evaluating, and tuning thresholds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared across all steps.")
    parser.add_argument(
        "--subset",
        default="",
        help="Optional subset string passed to evaluate_dataset (filters prompt/model columns).",
    )
    parser.add_argument(
        "--max-secondary-rows",
        type=int,
        default=50000,
        help="Maximum rows sampled from the large AI-vs-human dataset (0 = use all).",
    )
    parser.add_argument(
        "--tune-metric",
        choices=["f1", "youden_j", "mcc"],
        default="f1",
        help="Metric that the auto tuner will maximize.",
    )
    parser.add_argument("--skip-tune", action="store_true", help="Skip the threshold tuning step.")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests after tuning.")
    parser.add_argument(
        "--tests-target",
        default="tests.py",
        help="Test module or discovery target passed to unittest when --run-tests is enabled.",
    )
    return parser.parse_args()


def run_step(title: str, command: list[str]) -> None:
    print(f"\n[orchestrator] {title}")
    print(f"[orchestrator] Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python_exe = sys.executable

    # Step 1: Dataset fetch/prep.
    fetch_cmd = [
        python_exe,
        "-m",
        "tuning.dataset_fetch",
        "--seed",
        str(args.seed),
        "--max-secondary-rows",
        str(args.max_secondary_rows),
    ]
    run_step("Fetching and preprocessing dataset...", fetch_cmd)

    # Step 2: Evaluation.
    eval_cmd = [
        python_exe,
        "-m",
        "tuning.evaluate_dataset",
        "--seed",
        str(args.seed),
    ]
    if args.subset:
        eval_cmd.extend(["--subset", args.subset])
    run_step("Evaluating comparator on dataset splits...", eval_cmd)

    # Step 3: Threshold tuning.
    if not args.skip_tune:
        tune_cmd = [
            python_exe,
            "-m",
            "tuning.tune_threshold",
            "--auto",
            "--metric",
            args.tune_metric,
            "--apply-test",
            "--save",
        ]
        run_step(f"Auto-tuning threshold (metric={args.tune_metric})...", tune_cmd)
    else:
        print("[orchestrator] Skipping tuning step as requested.")

    # Optional: regression tests.
    if args.run_tests:
        test_cmd = [
            python_exe,
            "-m",
            "unittest",
            args.tests_target,
            "-v",
        ]
        run_step("Running regression tests...", test_cmd)

    print("\n[orchestrator] Workflow complete.")


if __name__ == "__main__":
    main()
