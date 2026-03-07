"""Verify the fix: test problem(solution) with the new direct subprocess approach."""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from iohblade.problems import MA_BBOB
    from iohblade.solution import Solution

    print("Creating MA_BBOB problem (dims=[5], budget_factor=800, eval_timeout=60)...")
    problem = MA_BBOB(
        dims=[5],
        budget_factor=800,
        training_instances=list(range(0, 5)),  # Only 5 instances for quick test
        test_instances=list(range(20, 25)),
        eval_timeout=60,
    )

    print("Setting up venv (this takes ~2 min for pip install)...")
    t0 = time.time()
    problem._ensure_env()
    print(f"  Venv ready in {time.time()-t0:.1f}s")

    # Test with RandomSearch
    random_search_code = '''
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
        return self.f_opt, self.x_opt
'''

    solution = Solution(
        code=random_search_code,
        name="RandomSearch",
        description="Simple random search",
        generation=0,
    )

    print("\nEvaluating RandomSearch through problem(solution)...")
    t0 = time.time()
    result = problem(solution)
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"Result:")
    print(f"  Time:     {elapsed:.2f}s")
    print(f"  Fitness:  {result.fitness}")
    print(f"  Error:    {result.error}")
    print(f"  Feedback: {str(result.feedback)[:200]}")
    print(f"{'='*60}")
    
    if result.fitness != -np.inf and not result.error:
        print("\n✓ FIX VERIFIED: Algorithm evaluated successfully!")
    elif result.error:
        print(f"\n✗ ERROR: {result.error}")
    else:
        print("\n✗ UNEXPECTED: fitness is -inf but no error")

    problem.cleanup()
    print("Done!")
