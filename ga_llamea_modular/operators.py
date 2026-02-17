"""
Genetic Operators for GA-LLAMEA
===============================

This module implements the four genetic operators used in GA-LLAMEA:
    1. SimplifyOperator: Simplify and improve a parent algorithm
    2. CrossoverOperator: Combine two parent algorithms
    3. RandomNewOperator: Generate a completely new algorithm (minimal skeleton)
    4. WeaknessRefinementOperator: Improve robustness using per-instance AOCC data
    
The simplify operator matches LLAMEA's proven Prompt5 instructions
verbatim, while crossover is GA-LLAMEA's unique addition for hybridization.
The weakness refinement operator uses per-instance performance data to guide
the LLM toward more robust algorithm designs.

OPERATOR DESIGN PHILOSOPHY:
    Each operator generates a prompt that is sent to the LLM. The prompt
    structure follows a pattern of:
        - Task description (what to optimize)
        - Population history (what's been tried before)
        - Parent code(s) (what to improve/combine)
        - Specific instruction (what this operator does)
        - Output format (how to structure the response)

BLADE INTEGRATION:
    - Uses problem.task_prompt, problem.example_prompt, problem.format_prompt
    - These are provided by BLADE problems like MA_BBOB
    - The LLM is queried through the protocol interface (works with BLADE's LLM)

CUSTOMIZATION:
    To modify how prompts are generated, you can subclass these operators
    and override the build_prompt() method. This allows experimenting with
    different prompting strategies.

USAGE:
    # Typically used internally by GA_LLaMEA, but can be used directly:
    from ga_llamea_modular.operators import SimplifyOperator
    
    simplify = SimplifyOperator()
    prompt = simplify.build_prompt(problem, population, parent)
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .interfaces import ProblemProtocol


class BaseOperator(ABC):
    """Abstract base class for genetic operators.
    
    Each operator is responsible for generating a prompt that instructs
    the LLM to produce a new algorithm based on the operator's strategy.
    
    Subclasses must implement:
        - build_prompt(): Generate the LLM prompt
        - name: Return the operator's name for logging
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the operator name for logging and bandit tracking."""
        pass
    
    @abstractmethod
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        """Build the prompt for this operator.
        
        Args:
            problem: The optimization problem (provides task/format prompts)
            population: Current population for history context
            parent: Primary parent solution (for mutation)
            parent2: Secondary parent solution (for crossover)
            
        Returns:
            str: Complete prompt to send to LLM
        """
        pass
    
    def _get_task_prompt(self, problem: ProblemProtocol) -> str:
        """Get the standardized task prompt matching LLAMEA's format.
        
        This uses the problem's built-in prompts (task_prompt, example_prompt)
        which are defined by BLADE problems like MA_BBOB. This ensures
        GA-LLAMEA uses the exact same prompts as LLAMEA.
        """
        # Use BLADE's task_prompt (same as LLAMEA)
        task_prompt = getattr(problem, 'task_prompt', '')
        example_prompt = getattr(problem, 'example_prompt', '')
        
        # Build prompt using BLADE's prompts (matching LLAMEA structure)
        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        
        return f"""{role_prompt}
{task_prompt}
{example_prompt}"""
    
    def _get_population_history(self, population: List[Any]) -> str:
        """Get a summary of previously generated algorithms.
        
        This provides the LLM with context about what's been tried before,
        helping it avoid duplicates and learn from past successes/failures.
        """
        if not population:
            return ""
        
        # Sort by fitness (best first)
        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)
        
        history = "List of previously generated algorithm names with mean AOCC score:\n"
        for sol in sorted_pop:
            name = sol.name if sol.name else "Unknown"
            fitness = sol.fitness if sol.fitness is not None else 0.0
            history += f"- {name}: {fitness:.4f}\n"
        return history


class SimplifyOperator(BaseOperator):
    """Simplify operator: Simplify and improve a single parent algorithm.
    
    This operator takes a single parent algorithm and asks the LLM to
    simplify it while improving performance. Simpler algorithms tend to
    be more robust and less prone to bugs, making this operator valuable
    for reducing the failure rate of generated code.
    
    Matches LLAMEA's proven Prompt5 "Simplify" instruction verbatim.
    
    When to use:
        - When algorithms are becoming overly complex
        - To reduce error rate from bloated hybrid code
        - When you want cleaner, more maintainable solutions
    
    Prompt structure:
        1. Task description + example
        2. Population history
        3. Selected parent code + fitness
        4. "Refine and simplify the selected algorithm to improve it."
        5. Output format
    """
    
    @property
    def name(self) -> str:
        return "simplify"
    
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        """Build prompt for simplify operator.
        
        Args:
            problem: Optimization problem
            population: Current population
            parent: Parent solution to simplify (required)
            parent2: Not used
            
        Returns:
            Complete simplify prompt
        """
        if parent is None:
            raise ValueError("Simplify requires a parent solution")
        
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        algo_details = f"""
Selected algorithm to simplify and improve:
Name: {parent.name}
Fitness: {parent.fitness:.4f}
Code:
```python
{parent.code}
```
"""
        instruction = "Refine and simplify the selected algorithm to improve it."
        
        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"


class CrossoverOperator(BaseOperator):
    """Crossover operator: Guided Concept Transfer between two parent algorithms.
    
    This operator uses a concept-level crossover strategy inspired by EoH but
    unique to GA-LLAMEA. Instead of giving the LLM two full code implementations
    to merge (which produces Frankenstein code), it:
    
    1. Shows the better parent's FULL CODE as the working reference
    2. Shows the weaker parent's NAME and DESCRIPTION only (no code)
    3. Asks the LLM to redesign the first algorithm by incorporating the
       strategic concept from the second — writing code from scratch
    
    This "Guided Concept Transfer" approach avoids code-merging failures because:
    - The LLM only sees one code implementation
    - The second parent is described abstractly, forcing the LLM to understand
      the concept and implement it fresh rather than copy-pasting
    - Explicit instruction to write from scratch prevents concatenation
    
    When to use:
        - When you have multiple good algorithms with different strategies
        - To explore combinations of successful approaches
        - For diversity in the search space
    
    Prompt structure:
        1. Task description + example
        2. Population history
        3. Better parent's full code + weaker parent's name/description
        4. Instruction: redesign using the concept, write from scratch
        5. Output format
    """
    
    @property
    def name(self) -> str:
        return "crossover"
    
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        """Build prompt for crossover operator (Guided Concept Transfer).
        
        Args:
            problem: Optimization problem
            population: Current population
            parent: First parent (higher fitness, provides full code)
            parent2: Second parent (provides name/description only as inspiration)
            
        Returns:
            Complete crossover prompt
        """
        if parent is None or parent2 is None:
            raise ValueError("Crossover requires two parent solutions")
        
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        # Only show the better parent's full code
        # The weaker parent is described abstractly to prevent code-merging
        parent2_description = ""
        if hasattr(parent2, 'description') and parent2.description:
            parent2_description = f"\nStrategy: {parent2.description}"
        
        algo_details = f"""
Working Algorithm (fitness: {parent.fitness:.4f}):
```python
{parent.code}
```

Alternative approach for inspiration: "{parent2.name}" (fitness: {parent2.fitness:.4f}){parent2_description}
"""
        instruction = f"""Create an improved algorithm by redesigning the working algorithm above.
Draw inspiration from the alternative approach "{parent2.name}" — think about what strategic concept it might use that could address a weakness in the working algorithm.
Write a clean implementation from scratch. Do not copy-paste code.
Use only numpy. Keep it simple and functional."""
        
        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"


class RandomNewOperator(BaseOperator):
    """Random new operator: Generate a completely new algorithm.
    
    This operator asks the LLM to create a novel algorithm from scratch,
    different from what's been tried before. This maintains exploration
    and can discover entirely new approaches.
    
    To reduce the high failure rate (~40%) from structural errors (missing
    __call__, wrong return types, array shape mismatches), the prompt includes
    the best existing algorithm's code as a structural reference when available.
    The LLM is explicitly told to use a DIFFERENT strategy — the reference is
    only for correct code structure and formatting.
    
    When to use:
        - To escape local optima
        - When existing algorithms have converged
        - For exploration of new algorithm families
    
    Prompt structure:
        1. Task description + example
        2. Population history (to avoid repetition)
        3. Working reference code for structure (when available)
        4. "Generate a completely new approach"
        5. Output format
    """
    
    @property
    def name(self) -> str:
        return "random_new"
    
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        """Build prompt for random new operator.
        
        Args:
            problem: Optimization problem
            population: Current population (for context on what to avoid)
            parent: Not used
            parent2: Not used
            
        Returns:
            Complete random new prompt
        """
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        is_init = kwargs.get("is_init", False)

        if is_init:
            instruction = ""
            history = "" # No history for initialization
            # Avoid extra newlines when instruction and history are empty
            return f"{task_prompt}\n\n{problem.format_prompt}"
        
        # Provide a minimal structural skeleton to reduce -inf failures from
        # formatting errors, WITHOUT revealing any algorithmic strategy.
        # This prevents DE monoculture by not showing the best solution's code.
        reference = ""
        if population:
            reference = """
For correct code structure, follow this template:
```python
import numpy as np

class YourAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb  # lower bounds (numpy array)
        ub = func.bounds.ub  # upper bounds (numpy array)
        f_opt = np.inf
        x_opt = None
        eval_count = 0

        # Your optimization logic here
        # Use func(x) to evaluate a candidate x (numpy array of shape (dim,))
        # Track eval_count and stop when eval_count >= self.budget

        return f_opt, x_opt
```
Use a DIFFERENT strategy from the algorithms listed above. This template is only for correct structure and formatting.
"""
        
        instruction = "Generate a new algorithm that is different from the algorithms you have tried before."
        
        return f"{task_prompt}\n\n{history}\n{reference}\n{instruction}\n\n{problem.format_prompt}"


class WeaknessRefinementOperator(BaseOperator):
    """Weakness refinement operator: Improve robustness across problem instances.
    
    This operator shows the parent algorithm's code alongside its **per-instance
    AOCC scores** (already stored in ``metadata["aucs"]``), sorted worst-to-best,
    with weak instances labeled.  It asks the LLM to analyse what optimisation
    landscapes cause failures and redesign the algorithm for robustness.
    
    The bottom quartile of instances is labelled WEAK and the top quartile
    STRONG, giving the LLM concrete evidence about where the algorithm
    underperforms.
    
    When to use:
        - When the population has good average fitness but high variance
        - To break through plateaus caused by algorithms that overfit to
          certain function types
        - To encourage robustness over specialisation
    
    Prompt structure:
        1. Task description + example
        2. Population history
        3. Parent code + per-instance AOCC scores with WEAK/STRONG labels
        4. Instruction to diagnose failures and redesign for robustness
        5. Output format
    """
    
    @property
    def name(self) -> str:
        return "refine_weakness"
    
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        """Build prompt for weakness refinement operator.
        
        Args:
            problem: Optimization problem
            population: Current population
            parent: Parent solution to refine (required, must have metadata["aucs"])
            parent2: Not used
            
        Returns:
            Complete weakness refinement prompt
        """
        if parent is None:
            raise ValueError("Weakness refinement requires a parent solution")
        
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        # Build per-instance performance breakdown
        aucs = parent.metadata.get("aucs", [])
        if aucs:
            # Sort by score and label bottom/top quartile
            indexed_aucs = sorted(enumerate(aucs), key=lambda x: x[1])
            n = len(indexed_aucs)
            weak_cutoff = n // 4
            strong_cutoff = n - n // 4
            
            perf_lines = []
            for rank, (idx, score) in enumerate(indexed_aucs):
                if rank < weak_cutoff:
                    label = " (WEAK)"
                elif rank >= strong_cutoff:
                    label = " (STRONG)"
                else:
                    label = ""
                perf_lines.append(f"- Instance {idx}: {score:.4f}{label}")
            
            perf_section = "\n".join(perf_lines)
        else:
            perf_section = "Per-instance data not available."
        
        algo_details = f"""
Algorithm to improve (mean AOCC: {parent.fitness:.4f}):
```python
{parent.code}
```

Per-instance performance (AOCC scores, sorted worst to best):
{perf_section}
"""
        
        instruction = """This algorithm performs well on average but poorly on some problem instances.
Analyze what types of optimization landscapes or function properties might
cause these failures. Redesign the algorithm to be more robust across all
instances while maintaining its strengths on the ones it already handles well."""
        
        return f"{task_prompt}\n\n{history}\n{algo_details}\n\n{instruction}\n\n{problem.format_prompt}"


# Factory dictionary for easy operator instantiation
OPERATORS = {
    "simplify": SimplifyOperator,
    "crossover": CrossoverOperator,
    "random_new": RandomNewOperator,
    "refine_weakness": WeaknessRefinementOperator,
}


def get_operator(name: str) -> BaseOperator:
    """Factory function to get an operator instance by name.
    
    Args:
        name: Operator name ("simplify", "crossover", or "random_new")
        
    Returns:
        Operator instance
        
    Raises:
        ValueError: If operator name is not recognized
    """
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator: {name}. Must be one of: {list(OPERATORS.keys())}")
    return OPERATORS[name]()
