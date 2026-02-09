
import json
import os

source_notebook = "examples/visualize_comparison.ipynb"
target_notebook = "examples/visualize_prompt_comparison.ipynb"
target_dir = "../results/COMPARISON-PROMPTS"

with open(source_notebook, 'r') as f:
    nb = json.load(f)

# 1. Update Header
nb['cells'][0]['source'] = [
    "# LLAMEA vs GA-LLAMEA: Prompt Comparison\n",
    "\n",
    f"Visualization of results from `{target_dir}`.\\n",
    "Comparison of:\n",
    "- LLaMEA (Prompt 5: Refine, New, Simplify)\n",
    "- LLaMEA (Prompt 1 Modified: Refine or redesign)\n",
    "- GA-LLAMEA (Baseline)\n",
    "\n",
    "This includes:\n",
    "- Convergence Plots\n",
    "- CEG Plots\n",
    "- Boxplots\n",
    "- Fitness Tables\n",
    "- Behavior Metrics (with code projection)\n"
]

# 2. Update Imports and Data Loading
nb['cells'][1]['source'] = [
    "from iohblade.loggers import ExperimentLogger\n",
    "from iohblade.plots import plot_convergence, plot_experiment_CEG, plot_boxplot_fitness_hue, plot_boxplot_fitness, fitness_table\n",
    "import os\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    f"# Load Results\n",
    f"logger = ExperimentLogger('{target_dir}', True)\n",
    "methods, problems = logger.get_methods_problems()\n",
    "print(\"Methods:\", methods)\n",
    "print(\"Problems:\", problems)"
]

# 3. Remove old data loading cells
del nb['cells'][2] 
del nb['cells'][2]

# 4. Clean up remaining cells (replace logger references and labels)
new_cells = []
# Keep 0 (header) and 1 (imports/load)
new_cells.append(nb['cells'][0])
new_cells.append(nb['cells'][1])

# Iterate from index 2
i = 2
while i < len(nb['cells']):
    cell = nb['cells'][i]
    
    # Keep helper functions (metrics)
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def safe_compute_metrics" in source_str or "def display_metrics" in source_str:
            new_cells.append(cell)
            i += 1
            continue

    if cell['cell_type'] == 'markdown':
        new_cells.append(cell)
        i += 1
        continue
        
    if cell['cell_type'] == 'code':
        # Replace logger and labels
        new_source = []
        for line in cell['source']:
            if "logger_mabbob" in line:
                line = line.replace("logger_mabbob", "logger")
            
            if "MA-BBOB" in line:
                line = line.replace("MA-BBOB", "Prompt Comparison Results")
            
            # Skip lines related to the "second" logger if present (legacy cleanup)
            if "logger_ga" in line:
                 # In original comparison notebook, we had code blocks for logger_ga.
                 # We want to skip those blocks entirely if they are purely duplicates for the second dataset.
                 # But here we are iterating cell by cell.
                 # If this line is the MAIN part of the cell, we might be keeping a cell we want to drop?
                 # Actually, better strategy:
                 # The original notebook had pairs: Plot MABBOB, Plot GA.
                 # We only need ONE plot command since `logger` now contains all methods.
                 pass
            
            new_source.append(line)
        
        # Check if this cell was intended for the "second" dataset (GA-LLAMEA in original)
        # If so, we skip it because the first cell (modified to use `logger`) now covers everything.
        source_joined = "".join(cell['source'])
        if "logger_ga" in source_joined:
            # Skip this cell entirely
            i += 1
            continue
            
        cell['source'] = new_source
        new_cells.append(cell)
        i += 1

nb['cells'] = new_cells

with open(target_notebook, 'w') as f:
    json.dump(nb, f, indent=4)

print(f"Created {target_notebook}")
