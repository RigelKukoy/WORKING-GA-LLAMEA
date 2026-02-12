"""
Genetic Operators for GA-LLAMEA
===============================

This module implements the three genetic operators used in GA-LLAMEA:
    1. SimplifyOperator: Simplify and improve a parent algorithm
    2. CrossoverOperator: Combine two parent algorithms
    3. RandomNewOperator: Generate a completely new algorithm
    
The simplify operator matches LLAMEA's proven Prompt5 instructions
verbatim, while crossover is GA-LLAMEA's unique addition for hybridization.

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
        
        # Include best solution as structural reference to reduce -inf failures
        reference = ""
        if population:
            best = max(population, key=lambda s: s.fitness)
            if best.code:
                reference = f"""
For reference, here is a working algorithm with correct structure and format:
```python
{best.code}
```
Your new algorithm must use a DIFFERENT strategy from the above. Use it only as a reference for correct code structure and formatting.
"""
        
        instruction = "Generate a new algorithm that is different from the algorithms you have tried before."
        
        return f"{task_prompt}\n\n{history}\n{reference}\n{instruction}\n\n{problem.format_prompt}"


# Factory dictionary for easy operator instantiation
OPERATORS = {
    "simplify": SimplifyOperator,
    "crossover": CrossoverOperator,
    "random_new": RandomNewOperator,
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
