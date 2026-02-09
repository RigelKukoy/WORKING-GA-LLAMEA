"""
Utility Functions
=================

Helper functions used across the GA-LLAMEA package.

Functions:
    calculate_reward: Compute reward for bandit update
    extract_code: Extract Python code from LLM response
    extract_description: Extract algorithm description from LLM response
    create_solution: Factory function for creating solution objects

BLADE INTEGRATION NOTES:
    - extract_code(): Uses same regex pattern as BLADE's LLM.extract_algorithm_code()
    - create_solution(): Creates solutions compatible with BLADE's Solution class
"""

import re
from typing import Optional, Any, Tuple, Type


def calculate_reward(parent_score: float, child_score: float, is_valid: bool) -> float:
    """Calculate normalized reward for bandit update.
    
    The reward function is designed to encourage improvement over the
    *parent* (not the global best), giving the bandit a meaningful signal:
    - Invalid solutions (errors) get 0 reward
    - Solutions worse than or equal to parent get 0 reward
    - Solutions better than parent get reward normalized to [0, 1]
    
    The normalization ensures the bandit receives clean signals regardless
    of the absolute fitness scale.
    
    Args:
        parent_score: Baseline fitness of the parent (or median for random_new).
                      This should be the *parent's* fitness, NOT the global best.
        child_score: New solution's fitness after evaluation.
        is_valid: Whether the solution is valid (no errors during evaluation)
        
    Returns:
        float: Reward value in [0.0, 1.0]
        
    Example:
        >>> calculate_reward(0.5, 0.7, True)   # Improvement over parent
        0.4
        >>> calculate_reward(0.7, 0.5, True)   # Worse than parent
        0.0
        >>> calculate_reward(0.5, 0.9, False)  # Invalid
        0.0
        >>> calculate_reward(0.0, 0.3, True)   # Parent had 0 fitness
        0.3
    """
    if not is_valid:
        return 0.0  # Invalid code gets zero reward
    if child_score <= parent_score:
        return 0.0  # No improvement over parent
    # Normalized improvement as a fraction, capped at 1.0
    if parent_score <= 0:
        return min(1.0, child_score)
    return min(1.0, (child_score - parent_score) / max(0.01, parent_score))


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response.
    
    Uses the same pattern as BLADE's LLM.extract_algorithm_code() for consistency.
    
    Extraction priority:
        1. Code in ```python ... ``` blocks
        2. Code in ``` ... ``` blocks
        3. Raw response if it contains 'def ' or 'class '
    
    Args:
        response: LLM response text
        
    Returns:
        str or None: Extracted code, or None if not found
        
    Example:
        >>> response = '''Here is my algorithm:
        ... ```python
        ... class MyOptimizer:
        ...     def __call__(self, f):
        ...         pass
        ... ```
        ... '''
        >>> code = extract_code(response)
        >>> 'class MyOptimizer' in code
        True
    """
    # Pattern matches BLADE's LLM.extract_algorithm_code()
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: return raw response if it looks like code
    if "def " in response or "class " in response:
        return response.strip()
    
    return None


def extract_description(response: str) -> str:
    """Extract algorithm description from LLM response.
    
    Looks for "# Description: ..." or "# Name: ..." patterns, which
    match the output format requested in prompts.
    
    Args:
        response: LLM response text
        
    Returns:
        str: Extracted description, or empty string if not found
        
    Example:
        >>> response = "# Description: Adaptive differential evolution\\n```python..."
        >>> extract_description(response)
        'Adaptive differential evolution'
    """
    match = re.search(
        r"#\s*(?:Description|Name):\s*(.*?)(?:\n|$)", 
        response, 
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_classname(code: str) -> str:
    """Extract the Python class name from generated code.
    
    This is important for BLADE's problem evaluation, which expects
    the solution to have a 'name' attribute containing the class name.
    
    Args:
        code: Python code string
        
    Returns:
        str: Class name or empty string if not found
        
    Example:
        >>> code = "class AdaptiveEvolution:\\n    def __call__(self, f):..."
        >>> extract_classname(code)
        'AdaptiveEvolution'
    """
    try:
        match = re.search(r"class\s*(\w*)(?:\(\w*\))?:", code, re.IGNORECASE)
        if match:
            return match.group(1)
    except Exception:
        pass
    return ""


# Modules that are NOT available in the evaluation sandbox.
# Generated code using these will always fail, wasting budget.
FORBIDDEN_IMPORTS = [
    "scipy.optimize",
    "cma",
    "sobol_seq",
    "torch",
    "tensorflow",
    "sklearn",
    "deap",
    "platypus",
    "pygmo",
    "optuna",
    "nevergrad",
]


def validate_code(code: str) -> Tuple[bool, str]:
    """Quick static validation of generated code before expensive evaluation.
    
    Performs lightweight checks to catch common issues that would cause
    the subprocess evaluation to fail, saving budget:
    
    1. Checks for forbidden imports (modules not in the sandbox)
    2. Verifies a class definition exists
    3. Verifies a __call__ method exists
    4. Tries compile() to catch syntax errors
    
    Args:
        code: Python code string from LLM response
        
    Returns:
        Tuple of (is_valid, error_message).
        is_valid is True if code passes all checks.
        error_message describes the first failure found, or "" if valid.
        
    Example:
        >>> code = "import cma\\nclass Foo:\\n    def __call__(self, f): pass"
        >>> validate_code(code)
        (False, "Forbidden import: 'cma' is not available in the evaluation environment")
        >>> code = "import numpy as np\\nclass Foo:\\n    def __call__(self, f): pass"
        >>> validate_code(code)
        (True, "")
    """
    if not code or not code.strip():
        return False, "Empty code"
    
    # 1. Check for forbidden imports
    for forbidden in FORBIDDEN_IMPORTS:
        # Match "import cma", "from cma import ...", "from scipy.optimize import ..."
        patterns = [
            rf"^\s*import\s+{re.escape(forbidden)}\b",
            rf"^\s*from\s+{re.escape(forbidden)}\b",
        ]
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE):
                return False, f"Forbidden import: '{forbidden}' is not available in the evaluation environment"
    
    # 2. Check for class definition
    if not re.search(r"class\s+\w+", code):
        return False, "No class definition found. Code must define a class."
    
    # 3. Check for __call__ method
    if "__call__" not in code:
        return False, "No __call__ method found. The class must implement __call__(self, func)."
    
    # 4. Try to compile (catches syntax errors)
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    return True, ""


class DefaultSolution:
    """Default minimal Solution implementation for standalone use.
    
    This class is used when no solution_class is provided to GA_LLaMEA.
    It provides the minimum attributes needed for the algorithm to work.
    
    For BLADE integration, always pass solution_class=Solution from iohblade
    to get full compatibility with BLADE's logging and evaluation system.
    
    Attributes:
        code (str): Algorithm code
        fitness (float): Fitness score
        error (str): Error message from evaluation
        name (str): Algorithm/class name
        metadata (dict): Additional metadata storage
        description (str): Algorithm description
        parent_ids (list): UUIDs of parent solutions (for CEG)
        operator (str): Operator that created this solution
        generation (int): Generation number
    """
    
    def __init__(self, code: str = "", **kwargs):
        self.code = code
        self.fitness = 0.0
        self.error = ""
        self.name = ""
        self.metadata = {}
        self.description = ""
        self.parent_ids = []
        self.operator = None
        self.generation = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_solution(
    solution_class: Type[Any],
    code: str = "",
    fitness: float = -float('inf'),
    parent_ids: Optional[list] = None,
    operator: Optional[str] = None,
    generation: int = 0,
) -> Any:
    """Factory function to create a Solution instance.
    
    Ensures the created solution has all required attributes for GA-LLAMEA
    and BLADE compatibility.
    
    Args:
        solution_class: Class to instantiate (BLADE's Solution or DefaultSolution)
        code: Algorithm code string
        fitness: Initial fitness value
        parent_ids: List of parent solution UUIDs (for CEG lineage tracking)
        operator: Name of the operator that created this solution (e.g. "mutation")
        generation: Generation number this solution belongs to
        
    Returns:
        Solution instance with all required attributes
    """
    sol = solution_class(code=code)
    sol.fitness = fitness
    
    # Set lineage fields for CEG (Code Evolution Graph) support
    if parent_ids is not None:
        sol.parent_ids = parent_ids
    elif not hasattr(sol, 'parent_ids'):
        sol.parent_ids = []
    if operator is not None:
        sol.operator = operator
    elif not hasattr(sol, 'operator'):
        sol.operator = None
    sol.generation = generation
    
    # Ensure required attributes exist
    if not hasattr(sol, 'metadata'):
        sol.metadata = {}
    if not hasattr(sol, 'error'):
        sol.error = ""
    if not hasattr(sol, 'name') or not sol.name:
        sol.name = extract_classname(code)
    if not hasattr(sol, 'description'):
        sol.description = ""
        
    return sol
