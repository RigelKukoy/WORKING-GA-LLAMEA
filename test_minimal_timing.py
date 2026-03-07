"""Minimal test: just time a loop of 4000 function evals (budget_factor=800 * dim=5)."""
import time
import numpy as np

print("Importing ioh...")
t0 = time.time()
import ioh
from ioh import get_problem, logger as ioh_logger
print(f"ioh imported in {time.time()-t0:.2f}s")

# Get a simple BBOB problem
problem = get_problem(11, 1, 5)  # function 11, instance 1, dim 5
budget = 4000  # 800 * 5

# Test 1: Simple loop calling func
print(f"\nRunning {budget} evaluations on BBOB f11 dim=5...")
t0 = time.time()
lb = problem.bounds.lb
ub = problem.bounds.ub
print(f"  bounds.lb type={type(lb)}, shape={np.array(lb).shape}, value={lb}")
print(f"  bounds.ub type={type(ub)}, shape={np.array(ub).shape}, value={ub}")

best_f = np.inf
for i in range(budget):
    x = np.random.uniform(lb, ub)
    f = problem(x)
    if f < best_f:
        best_f = f

elapsed = time.time() - t0
print(f"  {budget} evals in {elapsed:.2f}s ({budget/elapsed:.0f} evals/sec)")
print(f"  best_f = {best_f}")
problem.reset()

# Test 2: 20 instances, same as training set
print(f"\nRunning 20 instances * {budget} evals each...")
import pandas as pd
import os
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems")
weights_df = pd.read_csv(os.path.join(base_path, "mabbob", "weights.csv"), index_col=0)
iids_df = pd.read_csv(os.path.join(base_path, "mabbob", "iids.csv"), index_col=0)
opt_locs_df = pd.read_csv(os.path.join(base_path, "mabbob", "opt_locs.csv"), index_col=0)

dim = 5
total_start = time.time()
for idx in range(20):
    t0 = time.time()
    f_new = ioh.problem.ManyAffine(
        xopt=np.array(opt_locs_df.iloc[idx])[:dim],
        weights=np.array(weights_df.iloc[idx]),
        instances=np.array(iids_df.iloc[idx], dtype=int),
        n_variables=dim,
    )
    lb = f_new.bounds.lb
    ub = f_new.bounds.ub
    
    best_f = np.inf
    for i in range(budget):
        x = np.random.uniform(lb, ub)
        f = f_new(x)
        if f < best_f:
            best_f = f
    
    elapsed = time.time() - t0
    print(f"  Instance {idx}: {elapsed:.2f}s  best_f={best_f:.4f}")
    f_new.reset()

total_elapsed = time.time() - total_start
print(f"\nTotal for 20 instances: {total_elapsed:.2f}s")
print(f"That's {20 * budget} total function evaluations")
print("Done!")
