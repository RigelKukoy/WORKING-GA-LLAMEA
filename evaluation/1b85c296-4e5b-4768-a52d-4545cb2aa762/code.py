
import os
import numpy as np
import random
import re
import json
import time
import traceback
import math

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner.strategies.wrapper import OptAlg


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


