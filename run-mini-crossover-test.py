from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.problems import MA_BBOB
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
        print("\n\n>>> CALLED DYNAMIC CROSSOVER PROMPT <<<\n\n")
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
    api_key = os.getenv("GEMINI_API_KEY")
    llm = Gemini_LLM(api_key, "gemini-2.0-flash")
    
    budget = 10
    
    Baseline_LLaMEA = LLaMEA(
        llm=llm,
        budget=budget,
        name="Baseline-LLaMEA-DynamicCrossover",
        mutation_prompts=None,
        n_parents=2,
        n_offspring=4,
        elitism=True
    )
    
    # 50% chance to pick crossover to force it to trigger quickly
    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.", 
        DynamicCrossoverPrompt(Baseline_LLaMEA, num_inspirations=2),
        DynamicCrossoverPrompt(Baseline_LLaMEA, num_inspirations=2)
    ]
    Baseline_LLaMEA.kwargs['mutation_prompts'] = mutation_prompts

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/MINI-TEST_{timestamp}"
    logger = ExperimentLogger(experiment_dir)
    
    experiment = MA_BBOB_Experiment(
        methods=[Baseline_LLaMEA],
        runs=1,
        seeds=[42],
        dims=[5],
        budget_factor=100,
        budget=budget,
        eval_timeout=60,
        show_stdout=True,
        exp_logger=logger,
    )
    experiment()
