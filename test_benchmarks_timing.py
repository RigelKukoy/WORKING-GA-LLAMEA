import time
import traceback
from iohblade.solution import Solution

def test_automl():
    print("Testing AutoML (Task 31)...")
    try:
        from iohblade.problems.automl import AutoML
        
        prob = AutoML(openml_task_id=31)
        code = """
import numpy as np
class DummyAutoMLAlgorithm:
    def __init__(self, X, y, **kwargs):
        pass
    def __call__(self, X):
        return np.zeros(len(X))
"""
        sol = Solution(code=code, name="DummyAutoMLAlgorithm")
        t0 = time.time()
        res = prob.evaluate(sol)
        t1 = time.time()
        print(f"  Result: {res.fitness} ({res.feedback})")
        print(f"  Time taken: {t1 - t0:.2f} seconds.")
    except Exception as e:
        print(f"  Failed: {e}")
        traceback.print_exc()

def test_photonics(problem_type):
    print(f"Testing Photonics ({problem_type})...")
    try:
        from iohblade.problems.photonics import Photonics
        prob = Photonics(logger=None, problem_type=problem_type, seeds=1, budget_factor=10) # Reduced budget factor & seeds for test
        code = """
import numpy as np
class DummyPhotonicsAlgorithm:
    def __init__(self, budget, dim):
        self.budget = min(budget, 100) # Small limit for test
        self.dim = dim
    def __call__(self, func):
        best_f = float('inf')
        best_x = None
        for _ in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        return best_f, best_x
        """
        sol = Solution(code=code, name="DummyPhotonicsAlgorithm")
        t0 = time.time()
        res = prob.evaluate(sol)
        t1 = time.time()
        print(f"  Result: {res.fitness} ({res.feedback})")
        print(f"  Time taken: {t1 - t0:.2f} seconds.")
    except Exception as e:
        print(f"  Failed: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    test_automl()
    print("-" * 40)
    test_photonics("bragg")
    print("-" * 40)
    test_photonics("ellipsometry")
    print("-" * 40)
    test_photonics("photovoltaic")
