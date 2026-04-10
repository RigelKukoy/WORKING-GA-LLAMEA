import time
import traceback
import os
import shutil
from iohblade.solution import Solution

def test_kerneltuner():
    print("Testing KernelTuner...")
    try:
        from iohblade.problems.kerneltuner import Kerneltuner
        
        # Ensure benchmark_hub dataset is in the right place
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, "benchmark_hub_data")
        
        if not os.path.exists(data_dir):
            print(f"Copying benchmark_hub data to {data_dir}...")
            shutil.copytree("/tmp/benchmark_hub", data_dir)
            print("Data copied successfully.")
            
        prob = Kerneltuner(
            logger=None, 
            kernels=["gemm"], 
            gpus=["A100"], 
            budget=10 # Very small budget for quick test
        )
        code = """
import numpy as np
import random
class DummyKernelTunerAlgorithm(OptAlg):
    def __init__(self, budget=5000):
        self.param = None

    def __call__(self, func, searchspace):
        self.budget = searchspace.size
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()

        self.f_opt = np.inf
        self.x_opt = None
        
        # Just grab random sample
        pop = self.generate_population(10)
        
        for p in pop:
            try:
                f = func(p)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = p
            except BaseException:
                pass
                
        return self.x_opt, self.f_opt

    def generate_population(self, pop_size=10):
        pop = list(list(p) for p in self.searchspace.get_random_sample(pop_size))
        return pop
"""
        sol = Solution(code=code, name="DummyKernelTunerAlgorithm")
        t0 = time.time()
        res = prob.evaluate(sol)
        t1 = time.time()
        print(f"  Result: {res.fitness} ({res.feedback})")
        print(f"  Time taken: {t1 - t0:.2f} seconds.")
    except Exception as e:
        print(f"  Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_kerneltuner()
