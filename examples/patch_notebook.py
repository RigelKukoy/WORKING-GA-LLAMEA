import json
import os

notebook_path = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_3arm_vs_4arm_fixed.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

convergence_source = [
    "# Plot convergence curves using iohblade (consistent with other notebooks)\n",
    "if len(df_experiment) > 0 and len(methods) > 0:\n",
    "    print('Convergence Plot')\n",
    "    try:\n",
    "        from iohblade.experiments import ExperimentLogger\n",
    "        from iohblade.plots import plot_convergence\n",
    "        logger = ExperimentLogger(EXPERIMENT_DIR)\n",
    "        plot_convergence(logger, metric='fitness', save=False)\n",
    "    except Exception as e:\n",
    "        print(f'Error plotting convergence: {e}')\n",
    "else:\n",
    "    print('No data available for convergence plot')"
]

arm_selection_source = [
    "# Analyze longitudinal arm selection patterns\n",
    "def analyze_longitudinal_arm_selection(experiment_dir, methods):\n",
    "    \"\"\"Analyze arm selection patterns per generation from experiment logs\"\"\"\n",
    "    arm_data = {}\n",
    "    for method in methods:\n",
    "        run_dirs = glob.glob(os.path.join(experiment_dir, f'run-{method}-*'))\n",
    "        # Track operators counts per generation across all runs\n",
    "        gen_operator_counts = {}\n",
    "        for run_dir in run_dirs:\n",
    "            log_file = os.path.join(run_dir, 'log.jsonl')\n",
    "            if os.path.exists(log_file):\n",
    "                with open(log_file, 'r', encoding='utf-8') as f:\n",
    "                    for i, line in enumerate(f):\n",
    "                        try:\n",
    "                            entry = json.loads(line.strip())\n",
    "                            operator = entry.get('operator')\n",
    "                            if operator and operator != 'init':\n",
    "                                if i not in gen_operator_counts:\n",
    "                                    gen_operator_counts[i] = {'total': 0}\n",
    "                                gen_operator_counts[i][operator] = gen_operator_counts[i].get(operator, 0) + 1\n",
    "                                gen_operator_counts[i]['total'] += 1\n",
    "                        except json.JSONDecodeError:\n",
    "                            continue\n",
    "        \n",
    "        method_probs = {}\n",
    "        if gen_operator_counts:\n",
    "            max_gen = max(gen_operator_counts.keys())\n",
    "            all_ops = set()\n",
    "            for gen_counts in gen_operator_counts.values():\n",
    "                for k in gen_counts.keys():\n",
    "                    if k != 'total':\n",
    "                        all_ops.add(k)\n",
    "            \n",
    "            for op in all_ops:\n",
    "                method_probs[op] = []\n",
    "                \n",
    "            for i in range(max_gen + 1):\n",
    "                if i in gen_operator_counts and gen_operator_counts[i]['total'] > 0:\n",
    "                    total = gen_operator_counts[i]['total']\n",
    "                    for op in all_ops:\n",
    "                        count = gen_operator_counts[i].get(op, 0)\n",
    "                        method_probs[op].append((count / total) * 100)\n",
    "                else:\n",
    "                    # Fill with previous value if possible, else 0\n",
    "                    for op in all_ops:\n",
    "                        if len(method_probs[op]) > 0:\n",
    "                            method_probs[op].append(method_probs[op][-1])\n",
    "                        else:\n",
    "                            method_probs[op].append(0)\n",
    "        arm_data[method] = method_probs\n",
    "    return arm_data\n",
    "\n",
    "def plot_longitudinal_arm_selection(arm_data):\n",
    "    \"\"\"Plot arm selection probability over generations\"\"\"\n",
    "    fig, axes = plt.subplots(1, len(arm_data), figsize=(15, 6))\n",
    "    if len(arm_data) == 1:\n",
    "        axes = [axes]\n",
    "    \n",
    "    for i, (method, method_probs) in enumerate(arm_data.items()):\n",
    "        if not method_probs:\n",
    "            continue\n",
    "        for op, probs in method_probs.items():\n",
    "            axes[i].plot(range(len(probs)), probs, label=op, linewidth=2)\n",
    "        axes[i].set_title(f'{method} Longitudinal Operator Selection')\n",
    "        axes[i].set_xlabel('Generation / Evaluation')\n",
    "        axes[i].set_ylabel('Selection Probability (%)')\n",
    "        axes[i].legend()\n",
    "        axes[i].grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "if len(df_experiment) > 0:\n",
    "    print('GA-LLAMEA Longitudinal Arm Selection Analysis')\n",
    "    try:\n",
    "        arm_data = analyze_longitudinal_arm_selection(EXPERIMENT_DIR, methods)\n",
    "        if any(arm_data.values()):\n",
    "            plot_longitudinal_arm_selection(arm_data)\n",
    "        else:\n",
    "            print('No arm selection data found')\n",
    "    except Exception as e:\n",
    "        print(f'Error analyzing arm selection: {e}')\n",
    "else:\n",
    "    print('No data available for arm selection analysis')\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update Convergence Plot cell
        if "plot_convergence_custom" in source and "# Plot convergence curves" in source:
            cell['source'] = convergence_source

        # 2. Update Arm Selection Plot cell
        if "analyze_arm_selection(" in source and "plot_arm_selection(" in source:
            cell['source'] = arm_selection_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Patched visualize_3arm_vs_4arm_fixed.ipynb")
