"""
GA-LLAMEA vs EoH Comparison
===========================

This script runs an experiment to compare GA-LLAMEA without warmup against EoH.
"""

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA_Method
from iohblade.methods.eoh import EoH
import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

if __name__ == "__main__":
    load_dotenv()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

    ai_model = "gemini-2.5-flash"
    llm = Gemini_LLM(project, location, ai_model)
    
    budget = 100
    num_runs = 2
    seeds = [3 + i for i in range(num_runs)]

    print("=" * 80)
    print("GA-LLAMEA vs EoH Comparison")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # Method 1: GA-LLAMEA without warm-up
    GA_LLaMEA_NoWarmup = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-NoWarmup",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.15,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False,
        min_pulls_per_arm=4,  # Short burn-in
        init_oversample=3,
    )
    print("Configured GA-LLAMEA-NoWarmup")
    print("  Arms: simplify, crossover, random_new, refine")
    print("  Warm-up: 2 pulls per arm")
    print("  Population: n_parents=4, n_offspring=8")
    print()

    # Create timestamp and dir early to give it to EoH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/COMPARISON-EOH_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Method 2: EoH
    EoH_Method = EoH(
        llm=llm,
        budget=budget,
        name="EoH",
        pop_size=4,
        output_path=os.path.join(experiment_dir, "eoh_outputs"),
        timeout=120
    )
    print("Configured EoH")
    print("  Population size: 4")
    print()

    methods = [EoH_Method]
    
    logger = ExperimentLogger(experiment_dir)
    
    print(f"Results will be saved to: {experiment_dir}")
    print()
    print("=" * 80)
    print("Starting Experiment...")
    print("=" * 80)
    print()

    experiment = MA_BBOB_Experiment(
        methods=methods,
        runs=num_runs,
        seeds=seeds,
        dims=[5],
        budget_factor=2000,
        budget=budget,
        eval_timeout=120,
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
    print("=" * 80)
    print("Generating IOH data for best solutions...")
    print("=" * 80)
    print()
    
    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    try:
        exp_data = logger.get_data()
        if exp_data.empty:
            print("No experiment data found, skipping IOH generation.")
        else:
            training_instances = list(range(0, 20))
            test_instances = list(range(20, 120))
            
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
                    print("No solution data, skipping")
                    continue

                solution = Solution()
                solution.from_dict(sol_data)

                if not solution.code:
                    print("No code in solution, skipping")
                    continue

                try:
                    for test_seed in range(5):
                        np.random.seed(test_seed)
                        problem.test(solution, ioh_dir=ioh_dir)
                    print(f"IOH data written for {solution.name} (5 seeds, 50 instances, 250 runs)")
                except Exception as e:
                    print(f"Failed: {e}")

            problem.cleanup()
            print()
            print(f"IOH data saved to: {os.path.abspath(ioh_dir)}")
    except Exception as e:
        import traceback
        print()
        print("Error during IOH data generation:")
        traceback.print_exc()

    print()
    print("=" * 80)
    print("All Done!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Analyze results in: {experiment_dir}")
    print(f"2. View IOH data in: {ioh_dir}")
    print("3. Compare GA-LLAMEA-NoWarmup and EoH.")
    print()
