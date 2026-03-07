"""
Direct timing test - evaluates algorithms WITHOUT subprocess/venv to isolate compute time.
Uses the same MA_BBOB evaluation logic from mabbob.py but runs directly in this process.
"""
import time
import os
import sys
import numpy as np
import ioh
from ioh import logger as ioh_logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iohblade.utils import aoc_logger, correct_aoc, OverBudgetException

import pandas as pd
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems")
weights = pd.read_csv(os.path.join(base_path, "mabbob", "weights.csv"), index_col=0)
iids = pd.read_csv(os.path.join(base_path, "mabbob", "iids.csv"), index_col=0)
opt_locs = pd.read_csv(os.path.join(base_path, "mabbob", "opt_locs.csv"), index_col=0)

def run_evaluation(algorithm_code, algorithm_name, dims=[5], budget_factor=800, training_instances=list(range(0, 20))):
    namespace = {"np": np}
    exec(algorithm_code, namespace)
    
    # Small test run (same as mabbob.py line 147-154)
    try:
        from ioh import get_problem
        l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        problem = get_problem(11, 1, 2)
        problem.attach_logger(l2_temp)
        algorithm = namespace[algorithm_name](budget=100, dim=2)
        algorithm(problem)
    except OverBudgetException:
        pass
    except Exception as e:
        print(f"  Small test run FAILED: {e}")
        return None
    
    aucs = []
    for dim in dims:
        for idx_i, idx in enumerate(training_instances):
            t0 = time.time()
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
                algorithm = namespace[algorithm_name](budget=budget, dim=dim)
                algorithm(f_new)
            except OverBudgetException:
                aucs.append(0)
                break
            except Exception as e:
                print(f"  Instance {idx} FAILED: {e}")
                return None

            aucs.append(correct_aoc(f_new, l2, budget))
            t1 = time.time()
            print(f"  Instance {idx_i}/{len(training_instances)} (dim={dim}): {t1-t0:.2f}s  auc={aucs[-1]:.4f}")
            l2.reset(f_new)
            f_new.reset()
    
    return np.mean(aucs)


# ==== Test 1: Simple RandomSearch ====
print("=" * 70)
print("TEST 1: RandomSearch (baseline - should be fast)")
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

start = time.time()
result = run_evaluation(random_search_code, "RandomSearch")
elapsed = time.time() - start
print(f"\nTotal: {elapsed:.2f}s, Mean AUC: {result}")
print()

# ==== Test 2: A DE-style algorithm (similar to what the LLM generates) ====
print("=" * 70)
print("TEST 2: DE-style algorithm (pop_size=20, similar to LLM output)")
print("=" * 70)

de_code = '''
import numpy as np

class SimpleDEAlgorithm:
    def __init__(self, budget=10000, dim=10, pop_size_factor=4, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(4, pop_size_factor * dim)
        self.F = F
        self.CR = CR

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        evals = self.pop_size
        
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)].copy()
        
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                trial = np.copy(pop[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < f_opt:
                        f_opt = f_trial
                        x_opt = trial.copy()
        return f_opt, x_opt
'''

start = time.time()
result = run_evaluation(de_code, "SimpleDEAlgorithm")
elapsed = time.time() - start
print(f"\nTotal: {elapsed:.2f}s, Mean AUC: {result}")
print()

# ==== Test 3: Large population swarm (pop=30, like the LLM generates) ====
print("=" * 70)
print("TEST 3: Large population swarm (pop_size=30)")
print("=" * 70)

swarm_code = '''
import numpy as np

class LargeSwarm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        evals = self.pop_size
        pbest = pop.copy()
        pbest_f = fitness.copy()
        gbest = pop[np.argmin(fitness)].copy()
        gbest_f = np.min(fitness)
        
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                direction = 1.5 * r1 * (pbest[i] - pop[i]) + 1.5 * r2 * (gbest - pop[i])
                new_x = pop[i] + direction + 0.1 * np.random.randn(self.dim)
                new_x = np.clip(new_x, lb, ub)
                f_new = func(new_x)
                evals += 1
                if f_new < pbest_f[i]:
                    pbest[i] = new_x
                    pbest_f[i] = f_new
                if f_new < gbest_f:
                    gbest = new_x.copy()
                    gbest_f = f_new
                pop[i] = new_x
        return gbest_f, gbest
'''

start = time.time()
result = run_evaluation(swarm_code, "LargeSwarm")
elapsed = time.time() - start
print(f"\nTotal: {elapsed:.2f}s, Mean AUC: {result}")
print()

print("=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
