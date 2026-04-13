"""
GA-LLaMEA Run Script
====================

Runs GA-LLaMEA (Discounted Thompson Sampling operator selection) standalone.

Configuration matches the ablation study parameters:
  Arms      : simplify | crossover | random_new | refine
  Selection : D-TS bandit (discount=0.99, tau_max=0.2, epsilon=0.15)
  Init      : n_parents * init_oversample candidates → keep best n_parents
  Crossover : num_crossover_inspirations = 3 (full-code format)
  min_pulls : 0 (D-TS active from first evaluation, no burn-in)
"""

import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import AIML_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA_Method


if __name__ == "__main__":
    load_dotenv()

    api_key  = os.getenv("AIML_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    ai_model = "google/gemini-2.0-flash"

    llm = AIML_LLM(api_key=api_key, model=ai_model)

    budget   = 100
    num_runs = 5
    seeds    = list(range(num_runs))  # [0, 1, 2, 3, 4]

    NUM_CROSSOVER_INSPIRATIONS = 3
    N_PARENTS       = 4
    N_OFFSPRING     = 8
    INIT_OVERSAMPLE = 2

    print("=" * 80)
    print("GA-LLaMEA  (Discounted Thompson Sampling)")
    print("=" * 80)
    print(f"Model          : {ai_model}")
    print(f"Budget         : {budget} LLM queries per run")
    print(f"Runs           : {num_runs}  (seeds {seeds})")
    print(f"n_parents      : {N_PARENTS}  |  n_offspring: {N_OFFSPRING}")
    print(f"init_oversample: {INIT_OVERSAMPLE}  ({N_PARENTS}*{INIT_OVERSAMPLE}={N_PARENTS*INIT_OVERSAMPLE} init candidates → keep best {N_PARENTS})")
    print(f"Inspirations   : {NUM_CROSSOVER_INSPIRATIONS} (crossover arm)")
    print()

    GA_LLaMEA = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLaMEA",
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.15,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=NUM_CROSSOVER_INSPIRATIONS,
        use_init_prompt_for_random_new=False,
        init_oversample=INIT_OVERSAMPLE,
        min_pulls_per_arm=0,
    )

    print("✓ GA-LLaMEA")
    print("  Selection : Discounted Thompson Sampling (D-TS bandit)")
    print("  Arms      : simplify | crossover | random_new | refine")
    print(f"  Init      : {N_PARENTS*INIT_OVERSAMPLE} candidates → keep best {N_PARENTS} (init_oversample={INIT_OVERSAMPLE})")
    print(f"  Crossover : {NUM_CROSSOVER_INSPIRATIONS} inspiration(s), full-code format")
    print(f"  Discount  : 0.99  |  tau_max: 0.2  |  epsilon: 0.15  |  min_pulls: 0")
    print()

    # ── Experiment setup ──────────────────────────────────────────────────────

    methods = [GA_LLaMEA]

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/GA-LLAMEA_{timestamp}"
    logger         = ExperimentLogger(experiment_dir)

    print(f"Results → {experiment_dir}")
    print()
    print("=" * 80)
    print("Starting experiment...")
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
    print("Experiment complete!")
    print("=" * 80)
    print()

    # ── IOH data generation ───────────────────────────────────────────────────

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
            training_instances = list(range(0, 20))
            test_instances     = list(range(20, 120))

            problem = MA_BBOB(
                dims=[5],
                budget_factor=2000,
                training_instances=training_instances,
                test_instances=test_instances,
            )
            problem._ensure_env()

            total = len(exp_data)
            for idx, (_, row) in enumerate(exp_data.iterrows(), 1):
                method_name = row.get("method_name", "Unknown")
                seed        = row.get("seed", 0)
                sol_data    = row.get("solution", {})

                print(f"[{idx}/{total}] {method_name} seed={seed}...", end=" ")

                if not sol_data or not isinstance(sol_data, dict):
                    print("⚠ No solution data, skipping")
                    continue

                solution = Solution()
                solution.from_dict(sol_data)

                if not solution.code:
                    print("⚠ No code, skipping")
                    continue

                try:
                    for test_seed in range(5):
                        np.random.seed(test_seed)
                        problem.test(solution, ioh_dir=ioh_dir)
                    print(f"✓  {solution.name}")
                except Exception as e:
                    print(f"✗ {e}")

            problem.cleanup()
            print()
            print(f"✓ IOH data saved to: {os.path.abspath(ioh_dir)}")

    except Exception:
        import traceback
        print()
        print("✗ IOH data generation failed:")
        traceback.print_exc()

    print()
    print("=" * 80)
    print("All done!")
    print("=" * 80)
    print()
    print(f"Results in : {experiment_dir}")
    print(f"IOH data   : {ioh_dir}")
    print()
