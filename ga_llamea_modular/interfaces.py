"""
Protocol Interfaces for BLADE Compatibility
============================================

This module defines Protocol-based interfaces that allow GA-LLAMEA to work with 
BLADE without any modifications to BLADE's codebase.

HOW IT WORKS:
    Python Protocols (PEP 544) define structural subtyping. Instead of requiring
    explicit inheritance, any class that has the required methods/attributes
    automatically satisfies the protocol.

    BLADE's classes (LLM, Solution, Problem) naturally satisfy these protocols,
    enabling zero-change integration.

BLADE COMPATIBILITY:
    - LLMProtocol: BLADE's LLM.query() method matches this signature
    - SolutionProtocol: BLADE's Solution class has all required attributes
    - ProblemProtocol: BLADE's Problem class has evaluate() and prompt attributes

USAGE:
    Just pass BLADE classes directly - no adapters needed:
    
    from iohblade.llm import LLM
    from iohblade.solution import Solution
    from iohblade.problems import MA_BBOB
    
    llm = LLM(...)  # Satisfies LLMProtocol automatically
    solution = Solution(...)  # Satisfies SolutionProtocol automatically
    problem = MA_BBOB(...)  # Satisfies ProblemProtocol automatically
"""

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interface.
    
    BLADE's LLM class satisfies this protocol automatically.
    
    Required method:
        query(prompt): Send message(s) to LLM and get response text
    
    BLADE Implementation:
        BLADE's LLM.query() accepts a list of message dicts:
        [{"role": "user", "content": "..."}]
        
        GA-LLAMEA passes messages in this format, ensuring compatibility.
    """
    
    def query(self, prompt: str) -> str:
        """Query the LLM with a prompt and return the response.
        
        Args:
            prompt: Can be a string or list of message dicts
            
        Returns:
            str: The LLM's response text
        """
        ...


@runtime_checkable
class SolutionProtocol(Protocol):
    """Protocol for Solution interface.
    
    BLADE's Solution class satisfies this protocol automatically.
    
    Required attributes:
        code (str): The algorithm code
        fitness (float): Fitness score after evaluation
        error (str): Error message if evaluation failed
        name (str): Algorithm/class name
        metadata (Dict): Additional metadata storage
    
    BLADE Implementation:
        BLADE's Solution class in iohblade/solution.py has all these
        attributes plus additional ones (id, description, feedback, etc.)
    """
    
    code: str
    fitness: float
    error: str
    name: str
    metadata: Dict[str, Any]


@runtime_checkable
class ProblemProtocol(Protocol):
    """Protocol for Problem interface.
    
    BLADE's Problem class satisfies this protocol automatically.
    
    Required attributes:
        llm_call_counter (int): Tracks LLM API calls
        task_prompt (str): Description of the optimization task
        format_prompt (str): Instructions for output format
        example_prompt (str): Example code for the task
    
    Required method:
        evaluate(solution): Evaluate a solution and update its fitness
    
    BLADE Implementation:
        BLADE's Problem class in iohblade/problem.py provides evaluate()
        which runs the solution code in a subprocess and computes fitness.
        
        The prompts (task_prompt, format_prompt, example_prompt) are set
        by problem subclasses like MA_BBOB.
    """
    
    llm_call_counter: int
    task_prompt: str
    format_prompt: str
    
    def evaluate(self, solution: Any) -> None:
        """Evaluate a solution and update its fitness.
        
        BLADE's implementation runs the solution code in a subprocess,
        executes optimization, and computes AOCC (Area Over Convergence Curve).
        
        Args:
            solution: Solution object to evaluate (modifies fitness in-place)
        """
        ...


# Type aliases for explicit typing
LLM = LLMProtocol
Solution = SolutionProtocol
Problem = ProblemProtocol
