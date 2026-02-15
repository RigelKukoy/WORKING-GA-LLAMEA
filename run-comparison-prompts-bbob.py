"""
BBOB Comparison Experiment Script

This script runs a comparison of different LLM-driven optimization methods on the
BBOB (Black-Box Optimization Benchmarking) test suite.

Methods compared:
- LLaMEA with different mutation prompts
- GA-LLaMEA (improved version with concept-level crossover)
- EoH (Evolution of Heuristics)

Output:
- Results are saved to results/COMPARISON-BBOB_{timestamp}/
- IOH data for ECDF/EAF analysis is saved to ioh/ subdirectory
- Visualization can be done using visualize_comparison_prompts_bbob.ipynb

Usage:
    python run-comparison-prompts-bbob.py

Requirements:
    - Set GEMINI_API_KEY or OPENAI_API_KEY in environment or .env file
"""

import os
import sys
from datetime import datetime

import ioh
import numpy as np
from dotenv import load_dotenv

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import EoH, GA_LLaMEA_Method, LLaMEA
from iohblade.problems import BBOB_SBOX


def create_methods(llm, budget: int) -> list:
    """
    Create all methods for comparison.
    
    Args:
        llm: The LLM instance to use
        budget: Number of algorithm evaluations per run
        
    Returns:
        List of method instances
    """
    # 1. LLaMEA with Prompt 5 ("Refine", "New", "Simplify")
    mutation_prompts5 = [
        "Refine the strategy of the selected solution to improve it.",
        "Generate a new algorithm that is different from the algorithms you have tried before.",
        "Refine and simplify the selected algorithm to improve it.",
    ]
    LLaMEA_Prompt5 = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA-Prompt5",
        mutation_prompts=mutation_prompts5,
        n_parents=4,
        n_offspring=8,
        elitism=True,
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
        elitism=True,
    )

    # 3. LLaMEA with Adaptive Mutation (no fixed prompts)
    LLaMEA_Adaptive = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA-Adaptive",
        mutation_prompts=None,
        adaptive_mutation=True,
        n_parents=4,
        n_offspring=8,
        elitism=True,
    )

    # 4. GA-LLaMEA Improved (concept-level crossover, binary rewards, calibrated D-TS)
    GA_LLaMEA_Improved = GA_LLaMEA_Method(
        llm,
        budget=budget,
        name="GA-LLaMEA-Improved",
        n_parents=4,
        n_offspring=8,
        elitism=True,
        discount=0.9,
        tau_max=0.1,
    )

    # 5. EoH (Evolution of Heuristics)
    eoh_method = EoH(
        llm,
        budget=budget,
        name="EoH",
        pop_size=4,
    )

    return [
        LLaMEA_Prompt5,
        LLaMEA_Prompt1_Mod,
        LLaMEA_Adaptive,
        GA_LLaMEA_Improved,
        eoh_method,
    ]


def create_bbob_problem(
    training_fids: list = None,
    test_fids: list = None,
    dims: list = None,
    budget_factor: int = 2000,
    eval_timeout: int = 600,
    logger: ExperimentLogger = None,
) -> BBOB_SBOX:
    """
    Create a BBOB problem instance with proper configuration.
    
    Args:
        training_fids: List of function IDs for training (default: 10 diverse functions)
        test_fids: List of function IDs for testing (uses same as training if None)
        dims: List of dimensionalities (default: [5])
        budget_factor: Multiplier for evaluation budget (budget = budget_factor * dim)
        eval_timeout: Timeout for each algorithm evaluation in seconds
        logger: ExperimentLogger for IOH data directory
        
    Returns:
        BBOB_SBOX problem instance
    """
    if training_fids is None:
        # Default: 10 diverse functions covering all BBOB function groups
        training_fids = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]
    
    if test_fids is None:
        test_fids = training_fids  # Same functions, different instances
    
    if dims is None:
        dims = [5]  # Default dimensionality
    
    # Training: instances 1-5, Test: instances 6-15
    training_instances = [(f, i) for f in training_fids for i in range(1, 6)]
    test_instances = [(f, i) for f in test_fids for i in range(6, 16)]
    
    # IOH data directory
    ioh_dir = ""
    if logger is not None:
        ioh_dir = os.path.join(logger.dirname, "ioh")
        os.makedirs(ioh_dir, exist_ok=True)
    
    return BBOB_SBOX(
        training_instances=training_instances,
        test_instances=test_instances,
        dims=dims,
        budget_factor=budget_factor,
        eval_timeout=eval_timeout,
        name="BBOB",
        problem_type=ioh.ProblemClass.BBOB,  # Use standard BBOB (not SBOX)
        full_ioh_log=True,  # Enable full IOH logging for ECDF/EAF
        ioh_dir=ioh_dir,
    )


def run_experiment(
    methods: list = None,
    num_runs: int = 2,
    budget: int = 30,
    dims: list = None,
    budget_factor: int = 2000,
    show_stdout: bool = False,
    n_jobs: int = 1,
) -> str:
    """
    Run the BBOB comparison experiment.
    
    Args:
        methods: List of method names to include (None = all methods)
        num_runs: Number of independent runs per method
        budget: Number of algorithm evaluations per run
        dims: Problem dimensionalities
        budget_factor: Function evaluation budget = budget_factor * dim
        show_stdout: Whether to show stdout during execution
        n_jobs: Number of parallel jobs
        
    Returns:
        Path to the experiment results directory
    """
    load_dotenv()
    
    # Setup LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        ai_model = "gemini-2.5-flash"
        llm = Gemini_LLM(api_key, ai_model)
        print(f"Using Gemini LLM: {ai_model}")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")
        ai_model = "gpt-4o"
        llm = OpenAI_LLM(api_key, ai_model)
        print(f"Using OpenAI LLM: {ai_model}")
    
    # Generate seeds
    seeds = list(range(num_runs))
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/COMPARISON-BBOB_{timestamp}"
    logger = ExperimentLogger(experiment_dir)
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"Number of runs: {num_runs}")
    print(f"Budget (algorithm evaluations): {budget}")
    print(f"Seeds: {seeds}")
    
    # Create methods
    all_methods = create_methods(llm, budget)
    
    if methods is not None:
        all_methods = [m for m in all_methods if m.name in methods]
        print(f"Running methods: {[m.name for m in all_methods]}")
    else:
        print(f"Running all methods: {[m.name for m in all_methods]}")
    
    if len(all_methods) == 0:
        raise ValueError("No methods selected. Check method names.")
    
    # Create problem
    problem = create_bbob_problem(
        dims=dims,
        budget_factor=budget_factor,
        logger=logger,
    )
    
    print(f"Problem: {problem.name}")
    print(f"Dimensions: {problem.dims}")
    print(f"Training instances: {len(problem.training_instances)}")
    print(f"Test instances: {len(problem.test_instances)}")
    print(f"Budget factor: {problem.budget_factor}")
    print(f"IOH logging enabled: {problem.full_ioh_log}")
    print()
    
    # Run experiment
    print("=" * 60)
    print("Starting BBOB Comparison Experiment")
    print("=" * 60)
    
    experiment = Experiment(
        methods=all_methods,
        problems=[problem],
        runs=num_runs,
        seeds=seeds,
        budget=budget,
        show_stdout=show_stdout,
        log_stdout=True,
        exp_logger=logger,
        n_jobs=n_jobs,
    )
    
    experiment()
    
    print()
    print("=" * 60)
    print("Experiment Complete!")
    print(f"Results saved to: {os.path.abspath(experiment_dir)}")
    print()
    print("To visualize results, open:")
    print("  examples/visualize_comparison_prompts_bbob.ipynb")
    print("=" * 60)
    
    return experiment_dir


if __name__ == "__main__":
    # Configuration - modify these for your experiment
    config = {
        # Method selection (None = all methods, or list of names)
        "methods": ["GA-LLaMEA-Improved"],  # Only GA-LLaMEA for now
        
        # Number of independent runs
        "num_runs": 5,
        
        # Algorithm evaluation budget (number of algorithms generated per run)
        "budget": 30,
        
        # Problem dimensions
        "dims": [5],
        
        # Function evaluation budget = budget_factor * dim
        "budget_factor": 2000,
        
        # Show stdout during execution
        "show_stdout": True,
        
        # Number of parallel jobs (careful with API rate limits)
        "n_jobs": 1,
    }
    
    # Run the experiment
    experiment_dir = run_experiment(**config)
