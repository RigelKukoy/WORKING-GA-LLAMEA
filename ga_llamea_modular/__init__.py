"""
GA-LLAMEA Package: Modular Implementation for BLADE Integration
================================================================

A modular, well-documented implementation of GA-LLAMEA (LLaMEA with Discounted 
Thompson Sampling) designed for seamless integration with BLADE.

Package Structure:
    ga_llamea_modular/
    ├── __init__.py          # This file - Package exports
    ├── README.md            # Comprehensive integration guide
    ├── bandit.py            # Discounted Thompson Sampling (D-TS) bandit
    ├── operators.py         # Genetic operators (simplify, crossover, random_new, refine_weakness)
    ├── core.py              # Main GA_LLaMEA class
    ├── interfaces.py        # Protocol definitions for BLADE compatibility
    └── utils.py             # Helper functions

Quick Start:
    from ga_llamea_modular import GA_LLaMEA

    from iohblade.llm import LLM
    from iohblade.solution import Solution
    from iohblade.problems import MA_BBOB

    llm = LLM(model="gemini-2.0-flash")
    method = GA_LLaMEA(llm=llm, budget=100, solution_class=Solution)
    problem = MA_BBOB(function_id=1, dimension=5, instance=1)
    best = method(problem)

For detailed integration instructions, see README.md in this package.
"""

# Core class
from .core import GA_LLaMEA

# Bandit components
from .bandit import DiscountedThompsonSampler, ArmState

# Operators
from .operators import SimplifyOperator, CrossoverOperator, RandomNewOperator, WeaknessRefinementOperator

# Protocol interfaces
from .interfaces import LLMProtocol, SolutionProtocol, ProblemProtocol

# Utilities
from .utils import calculate_reward, extract_code, extract_description, validate_code

__version__ = "1.0.0"
__all__ = [
    # Core
    "GA_LLaMEA",
    # Bandit
    "DiscountedThompsonSampler",
    "ArmState",
    # Operators  
    "SimplifyOperator",
    "CrossoverOperator", 
    "RandomNewOperator",
    "WeaknessRefinementOperator",
    # Interfaces
    "LLMProtocol",
    "SolutionProtocol",
    "ProblemProtocol",
    # Utils
    "calculate_reward",
    "extract_code",
    "extract_description",
    "validate_code",
]
