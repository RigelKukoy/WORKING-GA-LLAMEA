import sys
import os
sys.path.insert(0, r"c:\Users\Kukoy\Documents\Experiment-GA\WORKING-GA-LLAMEA")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    plot_convergence,
    plot_experiment_CEG,
    plot_boxplot_fitness_hue,
    plot_boxplot_fitness,
    fitness_table,
)
EXPERIMENT_DIRS = [
    '../results/CROSSOVER-ABLATION',
    '../results/EoH',
    '../results/GA-LLAMEA-4-INIT-100',
    '../results/GA-LLAMEA-8-INIT-100',
]

logger = ExperimentLogger(EXPERIMENT_DIRS[0], True)
for extra_dir in EXPERIMENT_DIRS[1:]:
    logger.add_read_dir(extra_dir)

print('Fitness Boxplot (grouped by method)')
plot_boxplot_fitness(logger)
