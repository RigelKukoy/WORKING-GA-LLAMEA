"""
Quick test to understand where time is spent during evaluation.
Tests:
1. How long does a simple RandomSearch take on MA_BBOB (direct, no venv)?
2. How long does the venv creation take?
3. How long does a simple algorithm take through the full subprocess pipeline?
"""
import time
import os
import sys
import numpy as np
import ioh
from ioh import logger as ioh_logger

# Add the project to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iohblade.problems import MA_BBOB
from iohblade.solution import Solution
from iohblade.utils import aoc_logger, correct_aoc, OverBudgetException

# ---- Test 1: Direct evaluation of Random Search (NO subprocess, NO venv) ----
print("=" * 70)
print("TEST 1: Direct RandomSearch evaluation (no venv, no subprocess)")
print("=" * 70)

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

# Simulate what mabbob.py evaluate() does
namespace = {"np": np}
exec(random_search_code, namespace)

dims = [5]
budget_factor = 800
training_instances = list(range(0, 20))

# Load MA_BBOB data
import pandas as pd
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems")
weights = pd.read_csv(os.path.join(base_path, "mabbob", "weights.csv"), index_col=0)
iids = pd.read_csv(os.path.join(base_path, "mabbob", "iids.csv"), index_col=0)
opt_locs = pd.read_csv(os.path.join(base_path, "mabbob", "opt_locs.csv"), index_col=0)

start = time.time()
aucs = []
for dim in dims:
    for idx in training_instances:
        budget = budget_factor * dim
        f_new = ioh.problem.ManyAffine(
            xopt=np.array(opt_locs.iloc[idx])[:dim],
            weights=np.array(weights.iloc[idx]),
            instances=np.array(iids.iloc[idx], dtype=int),
            n_variables=dim,
        )
        f_new.set_id(100)
        f_new.set_instance(idx)

        l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        f_new.attach_logger(l2)

        try:
            algorithm = namespace["RandomSearch"](budget=budget, dim=dim)
            algorithm(f_new)
        except OverBudgetException:
            aucs.append(0)
            break

        aucs.append(correct_aoc(f_new, l2, budget))
        l2.reset(f_new)
        f_new.reset()

elapsed = time.time() - start
print(f"Direct eval of RandomSearch (20 instances, dim=5, budget_factor=800): {elapsed:.2f}s")
print(f"Mean AUC: {np.mean(aucs):.4f}")
print()

# ---- Test 2: Time the venv creation ----
print("=" * 70)
print("TEST 2: Virtual environment creation time")
print("=" * 70)

problem = MA_BBOB(
    dims=[5],
    budget_factor=800,
    training_instances=list(range(0, 20)),
    test_instances=list(range(20, 40)),
)

start = time.time()
problem._ensure_env()
elapsed = time.time() - start
print(f"Venv creation + pip install deps: {elapsed:.2f}s")
print(f"Venv path: {problem._env_path}")
print(f"Python bin: {problem._python_bin}")
print()

# ---- Test 3: Full pipeline evaluation (subprocess + venv) ----
print("=" * 70)
print("TEST 3: Full pipeline evaluation (subprocess + venv)")
print("=" * 70)

solution = Solution(
    code=random_search_code,
    name="RandomSearch",
    description="Simple random search",
    generation=0,
)

start = time.time()
result = problem(solution)
elapsed = time.time() - start
print(f"Full pipeline eval (subprocess + venv): {elapsed:.2f}s")
print(f"Fitness: {result.fitness}")
print(f"Feedback: {result.feedback}")
print(f"Error: {result.error}")
print()

# Cleanup
problem.cleanup()
print("Done!")
