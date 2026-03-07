"""Test venv creation and subprocess evaluation timing."""
import time
import os
import sys
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: How long does venv creation take?
print("=" * 70)
print("TEST 1: Virtual environment creation speed")
print("=" * 70)

t0 = time.time()
import virtualenv
t1 = time.time()
print(f"  Import virtualenv: {t1-t0:.2f}s")

env_dir = tempfile.mkdtemp(prefix="blade_test_env_")
env_path = Path(env_dir)
print(f"  Creating venv at: {env_path}")

t0 = time.time()
virtualenv.cli_run([env_dir])
t1 = time.time()
print(f"  Venv creation: {t1-t0:.2f}s")

python_bin = env_path / ("Scripts" if os.name == "nt" else "bin") / "python"
print(f"  Python bin: {python_bin}")
print(f"  Exists: {python_bin.exists()}")

# Test 2: How long does pip install take?
print("\n" + "=" * 70)
print("TEST 2: Installing dependencies into venv")
print("=" * 70)

deps = [
    "numpy",
    "cloudpickle>=3.1.0,<4",
    "joblib>=1.4.2,<2",
    "pandas==2.2.3",
    "ioh==0.3.22",
    "configspace==1.2.1",
    "smac==2.3.1",
]

t0 = time.time()
result = subprocess.run(
    [str(python_bin), "-m", "pip", "install", *deps],
    capture_output=True,
    text=True,
)
t1 = time.time()
print(f"  pip install time: {t1-t0:.2f}s")
print(f"  Return code: {result.returncode}")
if result.returncode != 0:
    print(f"  STDERR: {result.stderr[:500]}")
else:
    print(f"  Success! (stdout last 200 chars): ...{result.stdout[-200:]}")

# Test 3: Can we run a simple script in the venv?
print("\n" + "=" * 70)
print("TEST 3: Running a simple Python script in venv")
print("=" * 70)

test_script = env_path / "test_run.py"
test_script.write_text("import numpy as np; print('numpy version:', np.__version__); print('OK')")

t0 = time.time()
result = subprocess.run(
    [str(python_bin), str(test_script)],
    capture_output=True,
    text=True,
    timeout=60,
)
t1 = time.time()
print(f"  Simple script time: {t1-t0:.2f}s")
print(f"  stdout: {result.stdout.strip()}")
if result.returncode != 0:
    print(f"  STDERR: {result.stderr[:500]}")

# Test 4: Run the actual evaluation script pattern
print("\n" + "=" * 70)  
print("TEST 4: Running MA_BBOB evaluation in subprocess (like problem.py does)")
print("=" * 70)

# Set PYTHONPATH so iohblade can be found
repo_root = Path(__file__).resolve().parent
env = os.environ.copy()
env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")

eval_script = env_path / "test_mabbob_eval.py"
eval_script.write_text(f'''
import sys
import time
import numpy as np
sys.path.insert(0, r"{repo_root}")

print("Starting eval in subprocess...")
t0 = time.time()

import ioh
from ioh import logger as ioh_logger
print(f"  ioh import: {{time.time()-t0:.2f}}s")

import pandas as pd
base_path = r"{os.path.join(repo_root, 'iohblade', 'problems')}"
weights = pd.read_csv(base_path + "/mabbob/weights.csv", index_col=0)
iids = pd.read_csv(base_path + "/mabbob/iids.csv", index_col=0)
opt_locs = pd.read_csv(base_path + "/mabbob/opt_locs.csv", index_col=0)
print(f"  Data loaded: {{time.time()-t0:.2f}}s")

# Simulate RandomSearch
dim = 5
budget = 800 * dim
for idx in range(20):
    f_new = ioh.problem.ManyAffine(
        xopt=np.array(opt_locs.iloc[idx])[:dim],
        weights=np.array(weights.iloc[idx]),
        instances=np.array(iids.iloc[idx], dtype=int),
        n_variables=dim,
    )
    best_f = np.inf
    lb, ub = f_new.bounds.lb, f_new.bounds.ub
    for i in range(budget):
        x = np.random.uniform(lb, ub)
        f = f_new(x)
        if f < best_f:
            best_f = f
    f_new.reset()

total = time.time() - t0
print(f"  Total eval in subprocess: {{total:.2f}}s")
print("SUBPROCESS_DONE")
''')

t0 = time.time()
proc = subprocess.Popen(
    [str(python_bin), str(eval_script)],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
try:
    stdout, stderr = proc.communicate(timeout=300)
    t1 = time.time()
    print(f"  Total wall time: {t1-t0:.2f}s")
    print(f"  stdout: {stdout.strip()}")
    if stderr:
        print(f"  stderr (last 300 chars): ...{stderr[-300:]}")
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()
    print(f"  TIMED OUT after 300s!")
    print(f"  stdout: {stdout.strip()}")
    if stderr:
        print(f"  stderr (last 300 chars): ...{stderr[-300:]}")

# Cleanup
import shutil
shutil.rmtree(env_path, ignore_errors=True)
print("\nDone! Cleaned up venv.")
