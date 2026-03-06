"""
GA-LLAMEA Crossover vs No Crossover Comparison Experiment
=========================================================

This script compares GA-LLAMEA with crossover (baseline) vs without crossover.

The goal is to test the impact of the crossover operator on the performance of GA-LLAMEA.

Configurations:
- GA-LLAMEA-Crossover: simplify, crossover, random_new (baseline)
- GA-LLAMEA-No-Crossover: simplify, random_new

Both configurations use:
- Modified random_new with minimal skeleton (no full code reference)
- Same bandit parameters (discount=0.9, tau_max=0.1)
- Same population parameters (n_parents=4, n_offspring=8)
"""

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

from iohblade.methods import GA_LLaMEA_Method


if __name__ == "__main__":
    load_dotenv()

    # LLM Configuration
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file.")
    
    ai_model = "gemini-2.0-flash"

    llm = Gemini_LLM(api_key, ai_model)
    
    # Experiment Configuration
    budget = 100  # LLM queries per run
    num_runs = 5  # Multiple runs for statistical significance
    seeds = [0 + i for i in range(num_runs)]  # Seeds: [0, 1, 2, 3, 4]

    print("=" * 80)
    print("GA-LLAMEA Crossover vs No Crossover Comparison Experiment")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # 1. GA-LLAMEA with crossover (baseline)
    # Uses: simplify, crossover, random_new
    GA_LLaMEA_crossover = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-Crossover",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.9,
        tau_max=0.1,
        arm_names=["simplify", "crossover", "random_new"]
    )
    print("✓ Configured GA-LLAMEA-Crossover (baseline)")
    print("  Arms: simplify, crossover, random_new")
    print()

    # 2. GA-LLAMEA without crossover
    # Uses: simplify, random_new
    GA_LLaMEA_no_crossover = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-No-Crossover",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.9,
        tau_max=0.1,
        arm_names=["simplify", "random_new"]
    )
    print("✓ Configured GA-LLAMEA-No-Crossover")
    print("  Arms: simplify, random_new")
    print()

    methods = [GA_LLaMEA_crossover, GA_LLaMEA_no_crossover]
    
    # Generate a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/CROSSOVER-VS-NO-CROSSOVER_{timestamp}"
    logger = ExperimentLogger(experiment_dir)
    
    print(f"Results will be saved to: {experiment_dir}")
    print()
    print("=" * 80)
    print("Starting Experiment...")
    print("=" * 80)
    print()

    # Run the experiment
    experiment = MA_BBOB_Experiment(
        methods=methods,
        runs=num_runs,
        seeds=seeds,
        dims=[5],
        budget_factor=2000,
        budget=budget,
        eval_timeout=600,
        show_stdout=True,
        exp_logger=logger,
    )
    experiment()

    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()

    # --- IOH Data Generation ---
    # Re-evaluate best solutions with IOH logger to produce .dat files
    # needed for EAF/ECDF analysis via iohinspector.
    print("=" * 80)
    print("Generating IOH data for best solutions...")
    print("=" * 80)
    print()
    
    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    try:
        exp_data = logger.get_data()
        if exp_data.empty:
            print("⚠ No experiment data found, skipping IOH generation.")
        else:
            # Create a fresh MA_BBOB problem for test evaluation
            training_instances = list(range(0, 20))
            test_instances = list(range(20, 70))  # 50 test instances
            
            problem = MA_BBOB(
                dims=[5], 
                budget_factor=2000,
                training_instances=training_instances,
                test_instances=test_instances
            )
            problem._ensure_env()

            total_solutions = len(exp_data)
            for idx, (_, row) in enumerate(exp_data.iterrows(), 1):
                method_name = row.get("method_name", "Unknown")
                seed = row.get("seed", 0)
                sol_data = row.get("solution", {})

                print(f"[{idx}/{total_solutions}] Processing {method_name} seed={seed}...", end=" ")

                if not sol_data or not isinstance(sol_data, dict):
                    print("⚠ No solution data, skipping")
                    continue

                # Reconstruct the solution from logged data
                solution = Solution()
                solution.from_dict(sol_data)

                if not solution.code:
                    print("⚠ No code in solution, skipping")
                    continue

                try:
                    for test_seed in range(5):
                        np.random.seed(test_seed)
                        problem.test(solution, ioh_dir=ioh_dir)
                    print(f"✓ IOH data written for {solution.name} (5 seeds, 50 instances, 250 runs)")
                except Exception as e:
                    print(f"✗ Failed: {e}")

            problem.cleanup()
            print()
            print(f"✓ IOH data saved to: {os.path.abspath(ioh_dir)}")
    except Exception as e:
        import traceback
        print()
        print("✗ Error during IOH data generation:")
        traceback.print_exc()

    print()
    print("=" * 80)
    print("All Done!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Analyze results in: {experiment_dir}")
    print(f"2. View IOH data in: {ioh_dir}")
    print("3. Compare algorithm diversity between Crossover and No-Crossover configurations")
    print("4. Check the impact of the crossover operator on the performance of GA-LLAMEA")
    print()
