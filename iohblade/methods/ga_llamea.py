"""
GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for Adaptive Operator Selection
====================================================================================

This module provides a wrapper around the ga_llamea_modular.GA_LLaMEA class
to integrate it seamlessly with BLADE's Method interface and experiment system.

GA-LLAMEA extends LLaMEA by using a Discounted Thompson Sampling (D-TS) bandit
to adaptively select between three genetic operators:
    - Mutation: Refine a single parent algorithm
    - Crossover: Combine two parent algorithms
    - Random New: Generate a completely new algorithm

The bandit learns which operator works best over time, adapting to non-stationary
reward distributions through exponential discounting.
"""

from typing import Any

from ga_llamea_modular import GA_LLaMEA as GA_LLaMEA_Algorithm

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution


class GA_LLaMEA_Method(Method):
    """
    BLADE wrapper for GA-LLAMEA (LLaMEA with Discounted Thompson Sampling).

    This method extends LLaMEA by using a multi-armed bandit to adaptively
    select between mutation, crossover, and random new operators based on
    their observed rewards (fitness improvements).
    """

    def __init__(
        self,
        llm: LLM,
        budget: int,
        name: str = "GA-LLAMEA",
        n_parents: int = 4,
        n_offspring: int = 16,
        elitism: bool = True,
        discount: float = 0.9,
        tau_max: float = 1.0,
        **kwargs,
    ):
        """
        Initialize GA-LLAMEA method.

        Args:
            llm: BLADE's LLM instance for code generation.
            budget: Total number of LLM queries allowed.
            name: Method name for logging and identification.
            n_parents: Population size (μ). Default 4.
            n_offspring: Offspring generated per generation (λ). Default 16.
            elitism: Selection strategy. True = (μ+λ), False = (μ,λ)
            discount: D-TS discount factor γ ∈ (0, 1]. Default 0.9.
            tau_max: Maximum posterior uncertainty for D-TS.
            **kwargs: Additional arguments passed to GA_LLaMEA.
        """
        super().__init__(llm, budget, name)
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.elitism = elitism
        self.discount = discount
        self.tau_max = tau_max
        self.kwargs = kwargs

        # Store the last instance for access to bandit statistics
        self._last_instance: GA_LLaMEA_Algorithm | None = None

    def __call__(self, problem: Problem) -> Solution:
        """
        Execute GA-LLAMEA evolution on the given problem.

        Args:
            problem: BLADE Problem instance to optimize (e.g., MA_BBOB).

        Returns:
            Solution: Best solution found during evolution.
        """
        # Create GA-LLAMEA instance with BLADE's Solution class
        self._last_instance = GA_LLaMEA_Algorithm(
            llm=self.llm,
            budget=self.budget,
            solution_class=Solution,
            name=self.name,
            n_parents=self.n_parents,
            n_offspring=self.n_offspring,
            elitism=self.elitism,
            discount=self.discount,
            tau_max=self.tau_max,
            **self.kwargs,
        )

        # Run the evolution and return the best solution
        best_solution = self._last_instance(problem)
        return best_solution

    def get_bandit_state(self) -> dict:
        """Get the current state of the D-TS bandit."""
        if self._last_instance is not None and hasattr(self._last_instance, "bandit"):
            return self._last_instance.bandit.get_state_snapshot()
        return {}

    def get_arm_history(self) -> list:
        """Get the arm selection history for post-hoc analysis.
        
        Returns a list of dicts with keys 'eval' (LLM call index) and
        'operator' (arm name: 'init', 'mutation', 'crossover', 'random_new').
        """
        if self._last_instance is not None:
            return self._last_instance.arm_history
        return []

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for logging."""
        result = {
            "method_name": self.name,
            "budget": self.budget,
            "n_parents": self.n_parents,
            "n_offspring": self.n_offspring,
            "elitism": self.elitism,
            "discount": self.discount,
            "tau_max": self.tau_max,
            "method_type": "GA-LLAMEA",
        }
        result.update(self.kwargs)

        bandit_state = self.get_bandit_state()
        if bandit_state:
            result["bandit_state"] = bandit_state

        return result
