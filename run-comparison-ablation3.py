"""
GA-LLAMEA Ablation: Normal Initialization vs. 8 Init Prompts
============================================================

This script compares two variants of GA-LLAMEA-NoWarmup:
1. Standard initialization (n_parents=4, init_oversample=1 -> 4 candidates)
2. Oversampled initialization (n_parents=4, init_oversample=2 -> 8 candidates)

The rest of the configuration is identical.
"""

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import GeminiAPI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA_Method
import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

if __name__ == "__main__":
    load_dotenv()

    api_key  = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.5-flash"
    llm = GeminiAPI_LLM(api_key=api_key, model=ai_model)

    budget   = 100
    num_runs = 5
    seeds    = list(range(num_runs))  # [0, 1, 2, 3, 4]

    print("=" * 80)
    print("GA-LLAMEA Ablation: Init Size 4 vs 8")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # Method 1: GA-LLAMEA Baseline (4 init candidates)
    GA_LLaMEA_Baseline = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-Baseline",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.15,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False,
        min_pulls_per_arm=0,
        init_oversample=1, # Standard: 4 * 1 = 4 candidates
    )
    print("Configured GA-LLAMEA-Baseline")
    print("  Init Candidates: 4 (init_oversample=1)")
    print()

    # Method 2: GA-LLAMEA Init-8 (8 init candidates)
    GA_LLaMEA_Init8 = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-Init8",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.15,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False,
        min_pulls_per_arm=0,
        init_oversample=2, # Experiment: 4 * 2 = 8 candidates
    )
    print("Configured GA-LLAMEA-Init8")
    print("  Init Candidates: 8 (init_oversample=2)")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/ABLATION-INIT-SIZE_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    methods = [ GA_LLaMEA_Init8]
    
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
