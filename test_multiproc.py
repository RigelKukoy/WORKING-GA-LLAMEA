"""Test pickle size and multiprocessing overhead for MA_BBOB."""
import time
import sys
import os
import cloudpickle
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iohblade.problems import MA_BBOB
from iohblade.solution import Solution

# Create the problem
problem = MA_BBOB(
    dims=[5],
    budget_factor=800,
    training_instances=list(range(0, 20)),
    test_instances=list(range(20, 40)),
    eval_timeout=1200,
)

# Test 1: How big is the pickled problem?
print("TEST 1: Pickle size")
t0 = time.time()
data = cloudpickle.dumps(problem)
t1 = time.time()
print(f"  Pickle size: {len(data) / 1024:.1f} KB")
print(f"  Pickle time: {t1-t0:.2f}s")

t0 = time.time()
problem_restored = cloudpickle.loads(data)
t1 = time.time()
print(f"  Unpickle time: {t1-t0:.2f}s")

# Test 2: Ensure env  
print("\nTEST 2: _ensure_env")
t0 = time.time()
problem._ensure_env()
t1 = time.time()
print(f"  _ensure_env: {t1-t0:.2f}s")
print(f"  _env_path: {problem._env_path}")

# Test 3: Pickle WITH env path
t0 = time.time()
data2 = cloudpickle.dumps(problem)
t1 = time.time()
print(f"\n  Pickle size (with env): {len(data2) / 1024:.1f} KB")
print(f"  Pickle time (with env): {t1-t0:.2f}s")

# Test 4: Full __call__ with multiprocessing
print("\nTEST 3: Full problem(solution) call with multiprocessing")

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

t0 = time.time()
result = problem(solution)
t1 = time.time()
print(f"  Full problem(solution) time: {t1-t0:.2f}s")
print(f"  Fitness: {result.fitness}")
print(f"  Error: {result.error}")
print(f"  Feedback: {result.feedback}")

problem.cleanup()
print("\nDone!")
