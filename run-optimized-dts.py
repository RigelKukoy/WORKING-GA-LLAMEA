"""
GA-LLAMEA Optimized DTS Parameters Test
========================================

Test the optimized DTS parameters against historical baselines:
- epsilon=0.1 (was 0.4) — 90% DTS-guided selection
- tau_max=0.1 (was 0.2) — better arm discrimination
- discount=0.95 (was 0.99) — faster adaptation
- min_pulls=3 — 12-eval burn-in with 4 arms
- stagnation_threshold=3 — per-generation tracking
- num_crossover_inspirations=3 — based on Crossover3 success
- Improved random_new prompt with descriptions

Compare against:
- LLaMEA-Crossover3-NoRefine: 0.837 ± 0.010 (consistency champion)
- GA-LLAMEA-WithRefine: 0.808 ± 0.064 (highest ceiling 0.865)
- GA-LLAMEA-Warmup: 0.806 ± 0.030 (with burn-in)
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
    print("GA-LLAMEA Optimized DTS Parameters Test")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # Uses all NEW defaults from core.py:
    # - epsilon_exploration=0.1
    # - tau_max=0.1
    # - discount=0.95
    # - min_pulls_per_arm=3
    # - n_offspring=8
    # - num_crossover_inspirations=3
    # - arm_names=["simplify", "crossover", "random_new", "refine"]
    # - stagnation_threshold=3
    GA_LLaMEA_Optimized = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-Optimized",
        # All other params use new defaults
    )
    
    print("Configured GA-LLAMEA-Optimized (new defaults)")
    print("  epsilon_exploration: 0.1 (was 0.4)")
    print("  tau_max: 0.1 (was 0.2)")
    print("  discount: 0.95 (was 0.99)")
    print("  min_pulls_per_arm: 3")
    print("  n_offspring: 8")
    print("  num_crossover_inspirations: 3")
    print("  stagnation_threshold: 3 (per-generation)")
    print("  arm_names: [simplify, crossover, random_new, refine]")
    print("  random_new prompt: includes descriptions + first-principles guidance")
    print()

    methods = [GA_LLaMEA_Optimized]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/OPTIMIZED-DTS_{timestamp}"
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

    # --- Quick Results Summary ---
    try:
        exp_data = logger.get_data()
        if not exp_data.empty:
            fitnesses = [row['solution']['fitness'] for _, row in exp_data.iterrows() 
                        if row.get('solution') and isinstance(row['solution'], dict)]
            if fitnesses:
                mean_fit = np.mean(fitnesses)
                std_fit = np.std(fitnesses)
                max_fit = np.max(fitnesses)
                print(f"Results: {mean_fit:.3f} ± {std_fit:.3f} (max: {max_fit:.3f})")
                print()
                print("Baselines for comparison:")
                print("  LLaMEA-Crossover3-NoRefine: 0.837 ± 0.010")
                print("  GA-LLAMEA-WithRefine:      0.808 ± 0.064")
                print("  GA-LLAMEA-Warmup:          0.806 ± 0.030")
    except Exception as e:
        print(f"Could not compute summary: {e}")

    print()
    print("=" * 80)
    print("All Done!")
    print("=" * 80)
