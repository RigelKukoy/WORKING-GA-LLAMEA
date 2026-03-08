"""
GA-LLAMEA Ablation Study: Crossover and Refine Operator
=========================================================

This script runs an ablation study to compare:
1. Baseline LLaMEA with 4 prompts (random_new, simplify, refine, dynamic_crossover with 3 inspirations)
2. GA-LLAMEA with 4 arms (random_new, simplify, refine_weakness, crossover with 3 inspirations)
"""

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA_Method
from iohblade.methods.llamea import LLaMEA
import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import random

class DynamicCrossoverPrompt:
    """
    A dynamic string-like object that injects the current LLAMEA population
    into the prompt string at runtime.
    """
    def __init__(self, method_wrapper, num_inspirations=3):
        self.method_wrapper = method_wrapper
        self.num_inspirations = num_inspirations

    def __str__(self):
        # Access the underlying LLaMEA instance
        llamea_instance = getattr(self.method_wrapper, 'llamea_instance', None)
        if not llamea_instance or not llamea_instance.population:
            # Fallback if no population is available yet (e.g., very early in the run)
            return "Create an improved algorithm by redesigning the working algorithm. Write a clean implementation from scratch."

        # Filter out invalid solutions (no name/description/fitness)
        valid_pop = [p for p in llamea_instance.population if p.name and p.description and p.fitness is not None and not np.isinf(p.fitness)]
        
        if len(valid_pop) == 0:
            return "Create an improved algorithm by redesigning the working algorithm. Write a clean implementation from scratch."

        # Randomly select inspirations (or take all available up to num_inspirations)
        num_to_select = min(self.num_inspirations, len(valid_pop))
        inspirations = random.sample(valid_pop, num_to_select)

        # Build the dynamic string
        insp_texts = []
        for i, insp in enumerate(inspirations):
            desc = f"\nStrategy: {insp.description}" if insp.description else ""
            insp_texts.append(f'Alternative approach {i+1} for inspiration: "{insp.name}" (fitness: {insp.fitness:.4f}){desc}')
            
        inspirations_str = "\n".join(insp_texts)
        
        if len(inspirations) == 1:
            insp_names = f'"{inspirations[0].name}"'
            concept_text = "what strategic concept it might use that could address a weakness"
        else:
            insp_names = ", ".join([f'"{insp.name}"' for insp in inspirations[:-1]]) + f' and "{inspirations[-1].name}"'
            concept_text = "what strategic concepts they might use that could address weaknesses"
            
        return f"""Create an improved algorithm by redesigning the working algorithm above.
{inspirations_str}

Draw inspiration from the alternative approach{"es" if len(inspirations) > 1 else ""} {insp_names} — think about {concept_text} in the working algorithm.
Write a clean implementation from scratch."""

if __name__ == "__main__":
    load_dotenv()

    # LLM Configuration
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file.")
    
    ai_model = "gemini-2.5-flash"

    llm = Gemini_LLM(api_key, ai_model)
    
    # Experiment Configuration
    budget = 100  # LLM queries per run
    num_runs = 5  # Multiple runs for statistical significance
    seeds = [0 + i for i in range(num_runs)]  # Seeds: [0, 1, 2, 3, 4]

    print("=" * 80)
    print("GA-LLAMEA Refine Ablation Study")
    print("=" * 80)
    print(f"Budget: {budget} LLM queries per run")
    print(f"Runs: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {ai_model}")
    print()

    # Method 1. LLaMEA with 4 prompts (random_new, simplify, refine, dynamic crossover 3 inspirations)
    Baseline_LLaMEA = LLaMEA(
        llm=llm,
        budget=budget,
        name="Baseline-LLaMEA-4Prompts",
        mutation_prompts=None, # Will set this below
        n_parents=4,
        n_offspring=8,
        elitism=True
    )
    dynamic_crossover_prompt = DynamicCrossoverPrompt(Baseline_LLaMEA, num_inspirations=3)
    Baseline_LLaMEA.kwargs['mutation_prompts'] = [
        "Generate a new algorithm that is different from the algorithms you have tried before.",
        "Refine and simplify the selected algorithm to improve it.",
        "Refine the strategy of the selected solution to improve it.",
        dynamic_crossover_prompt
    ]
    print("✓ Configured Baseline-LLaMEA-4Prompts")
    print("  Prompts: random_new, simplify, refine, dynamic_crossover (3 inspirations)")
    print()

    # Method 2. GA-LLAMEA with crossover (3 arms) + refine operator
    GA_LLaMEA_WithRefine = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLAMEA-4Arms-WithRefine",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.9,
        tau_max=0.1,
        arm_names=["simplify", "crossover", "random_new", "refine_weakness"],
        num_crossover_inspirations=3,
        use_init_prompt_for_random_new=False
    )
    print("✓ Configured GA-LLAMEA-4Arms-WithRefine")
    print("  Arms: simplify, crossover, random_new, refine_weakness")
    print("  num_inspirations: 3")
    print()

    methods = [Baseline_LLaMEA, GA_LLaMEA_WithRefine]
    
    # Generate a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/ABLATION-PROMPTS-CROSSOVER_{timestamp}"
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
            test_instances = list(range(20, 120))  # 50 test instances
            
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
    print("3. Compare performance across the 4 configurations to see the impact of Crossover and Prompts.")
    print()
