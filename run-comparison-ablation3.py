"""
GA-LLAMEA Ablation Study: Warm-Up Phase + Larger Population
============================================================

This script runs an ablation study to test consistency improvements for
GA-LLAMEA-WithRefine (4 arms). Both methods keep the refine arm to preserve
the high ceiling (0.85+) while adding warm-up to eliminate bad runs.

Methods:
1. GA-LLAMEA-Warmup:   4 arms + warm-up (24 evals uniform random before bandit)
2. GA-LLAMEA-LargePop: 4 arms + warm-up + larger population (n_parents=6, n_offspring=12)

Compare against existing results (loaded in the visualization notebook):
- GA-LLAMEA-WithRefine (4 arms, no warm-up): 0.81 +/- 0.06
- LLaMEA-Crossover3-NoRefine (target):       0.84 +/- 0.01
"""

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
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

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file.")
    
    ai_model = "gemini-2.5-flash"
    llm = Gemini_LLM(api_key, ai_model)
    
    budget = 100
    num_runs = 5
    seeds = [0 + i for i in range(num_runs)]

    print("=" * 80)
    print("GA-LLAMEA Warm-Up + Larger Population Ablation Study")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # Method 1: 4 arms + min pulls (same pop size as existing WithRefine)
    GA_LLaMEA_Warmup = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-Warmup",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.4,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False,
        min_pulls_per_arm=0,  # 6 * 4 arms = 24 calls (approx same as 24 warm_up_budget)
    )
    print("Configured GA-LLAMEA-Warmup")
    print("  Arms: simplify, crossover, random_new, refine")
    print("  Warm-up: Min 6 pulls per arm (burn-in phase)")
    print("  Population: n_parents=4, n_offspring=8")
    print()

    # Method 2: 4 arms + min pulls + larger population
    GA_LLaMEA_LargePop = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-LargePop",
        n_parents=6,
        n_offspring=12,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.4,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False,
        min_pulls_per_arm=6,
    )
    print("Configured GA-LLAMEA-LargePop")
    print("  Arms: simplify, crossover, random_new, refine")
    print("  Warm-up: Min 6 pulls per arm (burn-in phase)")
    print("  Population: n_parents=6, n_offspring=12")
    print()

    methods = [GA_LLaMEA_Warmup]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/ABLATION-WARMUP-LARGEPOP_{timestamp}"
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
    print("3. Compare GA-LLAMEA-Warmup and GA-LLAMEA-LargePop against existing WithRefine and Crossover3.")
    print()
