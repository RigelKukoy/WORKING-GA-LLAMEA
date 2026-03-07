"""Extract and test the actual generated algorithms from the latest run."""
import json
import os
import time
import sys
import numpy as np
import ioh
from ioh import logger as ioh_logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iohblade.utils import aoc_logger, correct_aoc, OverBudgetException

import pandas as pd
base_path = os.path.join(os.path.dirname(__file__), "iohblade", "problems")
weights_df = pd.read_csv(os.path.join(base_path, "mabbob", "weights.csv"), index_col=0)
iids_df = pd.read_csv(os.path.join(base_path, "mabbob", "iids.csv"), index_col=0)
opt_locs_df = pd.read_csv(os.path.join(base_path, "mabbob", "opt_locs.csv"), index_col=0)

base = r"results\COMPARISON-PROMPTS_20260307_121534"
logfile = os.path.join(base, "run-GA-LLAMEA-InitPrompt-MA_BBOB-0", "log.jsonl")

with open(logfile) as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        
        # Only test ones that timed out (not code errors)
        if "timed out" not in str(entry.get("error", "")):
            continue
        
        name = entry["name"]
        code = entry["code"]
        
        print(f"\n{'='*70}")
        print(f"Testing algorithm [{i}]: {name}")
        print(f"{'='*70}")
        
        namespace = {"np": np}
        try:
            exec(code, namespace)
        except Exception as e:
            print(f"  EXEC FAILED: {e}")
            continue
        
        # Test 1: Small test run (same as mabbob.py does)
        print(f"  Step 1: Small test run (dim=2, budget=100)...")
        t0 = time.time()
        try:
            from ioh import get_problem
            l2_temp = aoc_logger(100, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
            problem = get_problem(11, 1, 2)
            problem.attach_logger(l2_temp)
            algorithm = namespace[name](budget=100, dim=2)
            algorithm(problem)
            print(f"    OK in {time.time()-t0:.2f}s")
        except OverBudgetException:
            print(f"    OverBudget (OK) in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"    FAILED in {time.time()-t0:.2f}s: {e}")
            continue
        
        # Test 2: Single instance evaluation (dim=5, budget=4000)
        print(f"  Step 2: Single instance (dim=5, budget=4000)...")
        dim = 5
        budget = 800 * dim
        idx = 0
        
        f_new = ioh.problem.ManyAffine(
            xopt=np.array(opt_locs_df.iloc[idx])[:dim],
            weights=np.array(weights_df.iloc[idx]),
            instances=np.array(iids_df.iloc[idx], dtype=int),
            n_variables=dim,
        )
        f_new.set_id(100)
        f_new.set_instance(idx)
        l2 = aoc_logger(budget, upper=1e2, triggers=[ioh_logger.trigger.ALWAYS])
        f_new.attach_logger(l2)
        
        t0 = time.time()
        try:
            algorithm = namespace[name](budget=budget, dim=dim)
            # Add a hard timeout of 30 seconds
            import signal
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(30)
            algorithm(f_new)
            elapsed = time.time() - t0
            auc = correct_aoc(f_new, l2, budget)
            print(f"    OK in {elapsed:.2f}s  AUC={auc:.4f}")
        except OverBudgetException:
            elapsed = time.time() - t0
            print(f"    OverBudget in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    FAILED in {elapsed:.2f}s: {e}")
        
        f_new.reset()
        
        # Only test one algorithm to save time
        print(f"\n  NOTE: Only tested first timed-out algorithm. Check if it's slow or fast.")
        break

print("\nDone!")
