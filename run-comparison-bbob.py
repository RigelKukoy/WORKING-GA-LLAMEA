
from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM
from iohblade.methods import GA_LLaMEA_Method
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import BBOB_SBOX
import ioh
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.0-flash"
    llm = Gemini_LLM(api_key, ai_model)
    budget = 100

    num_runs = 5
    seeds = [0 + i for i in range(num_runs)]

    # GA-LLAMEA with 4 arms (simplify, crossover, random_new, refine_weakness)
    GA_LLaMEA_Baseline = GA_LLaMEA_Method(
        llm,
        budget=budget,
        name="GA-LLAMEA-Improved",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.9,
        tau_max=0.1
    )

    methods = [GA_LLaMEA_Baseline]

    # BBOB function IDs to use (a representative subset of the 24 functions)
    fids = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]

    # Training: instances 1-5 per function, Test: instances 5-15 per function
    training_instances = [(f, i) for f in fids for i in range(1, 6)]
    test_instances = [(f, i) for f in fids for i in range(5, 16)]

    # Generate a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/COMPARISON-BBOB_{timestamp}"
    logger = ExperimentLogger(experiment_dir)

    # Create BBOB problem with IOH logging enabled during training
    bbob_problem = BBOB_SBOX(
        training_instances=training_instances,
        test_instances=test_instances,
        dims=[5],
        budget_factor=2000,
        eval_timeout=300,
        name="BBOB",
        problem_type=ioh.ProblemClass.BBOB,
        full_ioh_log=True,
        ioh_dir=f"{logger.dirname}/ioh",
    )

    print(f"Starting BBOB Experiment: GA-LLAMEA. Results will be saved to {experiment_dir}")
    experiment = Experiment(
        methods=methods,
        problems=[bbob_problem],
        runs=num_runs,
        seeds=seeds,
        show_stdout=True,
        exp_logger=logger,
        budget=budget,
    )
    experiment()

    # --- IOH Data Generation ---
    # Re-evaluate best solutions on test instances with IOH logger
    # to produce .dat files for EAF/ECDF analysis via iohinspector.
    print("\n--- Generating IOH data for best solutions ---")
    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    try:
        exp_data = logger.get_data()
        if exp_data.empty:
            print("No experiment data found, skipping IOH generation.")
        else:
            # Create a fresh BBOB_SBOX problem for test evaluation with ioh_dir set
            test_problem = BBOB_SBOX(
                training_instances=training_instances,
                test_instances=test_instances,
                dims=[5],
                budget_factor=2000,
                eval_timeout=600,
                name="BBOB",
                problem_type=ioh.ProblemClass.BBOB,
                full_ioh_log=True,
                ioh_dir=ioh_dir,
            )
            test_problem._ensure_env()

            for _, row in exp_data.iterrows():
                method_name = row.get("method_name", "Unknown")
                seed = row.get("seed", 0)
                sol_data = row.get("solution", {})

                if not sol_data or not isinstance(sol_data, dict):
                    print(f"  Skipping {method_name} seed={seed}: no solution data")
                    continue

                # Reconstruct the solution from logged data
                solution = Solution()
                solution.from_dict(sol_data)

                if not solution.code:
                    print(f"  Skipping {method_name} seed={seed}: no code in solution")
                    continue

                print(f"  Testing {method_name} seed={seed} ({solution.name})...")
                try:
                    test_problem.test(solution)
                    print(f"    IOH data written for {solution.name}")
                except Exception as e:
                    print(f"    Failed to generate IOH data for {solution.name}: {e}")

            test_problem.cleanup()
            print(f"IOH data saved to: {os.path.abspath(ioh_dir)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during IOH data generation: {e}")
