"""
Standalone script to generate IOHProfiler data from existing experiment results.

This script reads the experimentlog.jsonl from a given experiment directory,
reconstructs Solution objects, and re-evaluates them on the MA-BBOB test instances
with IOH logging enabled. This produces the .dat files needed for EAF/ECDF
analysis via iohinspector.

Usage:
    python generate-ioh-data.py <experiment_dir>

Example:
    python generate-ioh-data.py results/COMPARISON-PROMPTS_20260208_032906
"""

import os
import sys
import traceback

import pandas as pd

from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB


def generate_ioh_data(experiment_dir, dims=None, budget_factor=2000, eval_timeout=300):
    """
    Generate IOHProfiler .dat files from an experiment's logged results.

    Args:
        experiment_dir (str): Path to the experiment results directory
            (must contain experimentlog.jsonl).
        dims (list): Dimensionalities to evaluate on. Defaults to [5].
        budget_factor (int): Budget factor for MA-BBOB evaluation.
        eval_timeout (int): Timeout in seconds for each evaluation.
    """
    if dims is None:
        dims = [5]

    # Validate that experiment directory exists
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    log_path = os.path.join(experiment_dir, "experimentlog.jsonl")
    if not os.path.exists(log_path):
        print(f"Error: experimentlog.jsonl not found in {experiment_dir}")
        sys.exit(1)

    # Read experiment data
    print(f"Reading experiment data from: {log_path}")
    exp_data = pd.read_json(log_path, lines=True)

    if exp_data.empty:
        print("No experiment data found. Exiting.")
        sys.exit(1)

    print(f"Found {len(exp_data)} runs in experiment log.")

    # Show summary of methods and seeds
    if "method_name" in exp_data.columns:
        print("\nRuns per method:")
        print(exp_data["method_name"].value_counts().to_string())
        print()

    # Create IOH output directory
    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    # Create a fresh MA_BBOB problem for test evaluation
    print(f"Initializing MA_BBOB problem (dims={dims}, budget_factor={budget_factor})...")
    problem = MA_BBOB(dims=dims, budget_factor=budget_factor, eval_timeout=eval_timeout)
    problem._ensure_env()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, row in exp_data.iterrows():
        method_name = row.get("method_name", "Unknown")
        seed = row.get("seed", 0)
        sol_data = row.get("solution", {})

        if not sol_data or not isinstance(sol_data, dict):
            print(f"  [{idx+1}/{len(exp_data)}] Skipping {method_name} seed={seed}: no solution data")
            skip_count += 1
            continue

        # Reconstruct the solution from logged data
        solution = Solution()
        solution.from_dict(sol_data)

        if not solution.code:
            print(f"  [{idx+1}/{len(exp_data)}] Skipping {method_name} seed={seed}: no code in solution")
            skip_count += 1
            continue

        print(f"  [{idx+1}/{len(exp_data)}] Testing {method_name} seed={seed} ({solution.name})...")
        try:
            problem.test(solution, ioh_dir=ioh_dir)
            print(f"    -> IOH data written for {solution.name}")
            success_count += 1
        except Exception as e:
            print(f"    -> Failed to generate IOH data for {solution.name}: {e}")
            traceback.print_exc()
            fail_count += 1

    problem.cleanup()

    print(f"\n--- IOH Data Generation Complete ---")
    print(f"  Output directory: {os.path.abspath(ioh_dir)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped:    {skip_count}")
    print(f"  Failed:     {fail_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Please provide the experiment directory as an argument.")
        print("Example: python generate-ioh-data.py results/COMPARISON-PROMPTS_20260208_032906")
        sys.exit(1)

    experiment_directory = sys.argv[1]

    # Optional: parse dims and budget_factor from additional args
    dims = [5]  # default
    budget_factor = 2000  # default
    eval_timeout = 300  # default

    if len(sys.argv) > 2:
        try:
            dims = [int(d) for d in sys.argv[2].split(",")]
        except ValueError:
            print(f"Warning: Could not parse dims '{sys.argv[2]}', using default [5]")
            dims = [5]

    if len(sys.argv) > 3:
        try:
            budget_factor = int(sys.argv[3])
        except ValueError:
            print(f"Warning: Could not parse budget_factor '{sys.argv[3]}', using default 2000")
            budget_factor = 2000

    generate_ioh_data(
        experiment_dir=experiment_directory,
        dims=dims,
        budget_factor=budget_factor,
        eval_timeout=eval_timeout,
    )
