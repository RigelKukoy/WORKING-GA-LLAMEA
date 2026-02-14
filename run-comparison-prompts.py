
from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.methods import LLaMEA, GA_LLaMEA_Method, EoH
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.5-flash"
    llm = Gemini_LLM(api_key, ai_model)
    budget = 100 
    
    num_runs = 10
    # Generates seeds starting from 4: [4, 5, 6, ...]
    seeds = [0 + i for i in range(num_runs)]

    # 1. LLaMEA with Prompt 5 ("Refine", "New", "Simplify")
    mutation_prompts5 = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
        "Refine and simplify the selected algorithm to improve it.", #simplify
    ]
    LLaMEA_Prompt5 = LLaMEA(
        llm, 
        budget=budget, 
        name="LLaMEA-Prompt5", 
        mutation_prompts=mutation_prompts5, 
        n_parents=4, 
        n_offspring=8, 
        elitism=True
    )

    # 2. LLaMEA with Modified Prompt 1 ("Refine or redesign...")
    mutation_prompts1_mod = [
        "Refine or redesign the selected algorithm to improve it.",
    ]
    LLaMEA_Prompt1_Mod = LLaMEA(
        llm, 
        budget=budget, 
        name="LLaMEA-Prompt1-Mod", 
        mutation_prompts=mutation_prompts1_mod, 
        n_parents=4, 
        n_offspring=8, 
        elitism=True
    )

    # 3. GA-LLAMEA Improved (concept-level crossover, binary rewards, calibrated D-TS)
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

    # 4. EoH
    eoh_method = EoH(
        llm,
        budget=budget,
        name="EoH",
        pop_size=4
    )

    methods = [GA_LLaMEA_Baseline]
    
    # Generate a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/COMPARISON-PROMPTS_{timestamp}"
    logger = ExperimentLogger(experiment_dir)
    
    print(f"Starting Comparison Experiment: GA-LLAMEA only. Results will be saved to {experiment_dir}")
    experiment = MA_BBOB_Experiment(
        methods=methods, 
        runs=num_runs, 
        seeds=seeds, 
        dims=[5], 
        budget_factor=2000, 
        budget=budget, 
        eval_timeout=300, 
        show_stdout=True, 
        exp_logger=logger
    ) 
    experiment()

    # --- IOH Data Generation ---
    # Re-evaluate best solutions with IOH logger to produce .dat files
    # needed for EAF/ECDF analysis via iohinspector.
    print("\n--- Generating IOH data for best solutions ---")
    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    try:
        exp_data = logger.get_data()
        if exp_data.empty:
            print("No experiment data found, skipping IOH generation.")
        else:
            # Create a fresh MA_BBOB problem for test evaluation
            problem = MA_BBOB(dims=[5], budget_factor=2000)
            problem._ensure_env()

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
                    problem.test(solution, ioh_dir=ioh_dir)
                    print(f"    IOH data written for {solution.name}")
                except Exception as e:
                    print(f"    Failed to generate IOH data for {solution.name}: {e}")

            problem.cleanup()
            print(f"IOH data saved to: {os.path.abspath(ioh_dir)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during IOH data generation: {e}")
