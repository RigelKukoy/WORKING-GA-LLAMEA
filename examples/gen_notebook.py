"""
gen_notebook.py
Generates behaviour_profile_and_correlation.ipynb using nbformat.
Replicates the two-step methodology from:
  van Stein et al. (2025) Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery

Key point: uses ALL generated algorithms (~500 per method), not just the best one.
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

# ---------------------------------------------------------------------------
# Cell definitions (in order)
# ---------------------------------------------------------------------------

cells = []

# ── MARKDOWN CELL 1 (title) ──────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "# Behaviour Space Analysis -- GA-LLaMEA Experiments\n"
    "\n"
    "Replicates the two-step methodology from:\n"
    "> van Stein et al. (2025) Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery\n"
    "\n"
    "## Methodology (matching the paper)\n"
    "\n"
    "Data scope: ALL generated algorithms from each experiment (~500 per method).\n"
    "Each algorithm is a unique piece of code from the LLaMEA/EoH search.\n"
    "The AOCC fitness for each algorithm comes directly from the experiment log.\n"
    "\n"
    "Step 1 -- Correlation heatmap (feature selection):\n"
    "Compute Pearson r between all 11 behaviour metrics + normalised AOCC.\n"
    "Identify redundant pairs (|r| > 0.7) and drop one from each pair.\n"
    "The paper selected: Expl %, Conv-rate, Delta fitness, Success %, No-imp streak.\n"
    "\n"
    "Step 2 -- Parallel coordinates (gold standard profile):\n"
    "Plot all generated algorithms as polylines coloured by AOCC quartile.\n"
    "Q4 (dark red) lines reveal the behaviour profile of successful algorithms.\n"
    "\n"
    "---\n"
    "Experiments:\n"
    "| Method | Experiment Dir |\n"
    "|--------|----------------|\n"
    "| EoH | ../results/EoH |\n"
    "| Baseline-LLaMEA | ../results/CROSSOVER-ABLATION/baseline-LLAMEA |\n"
    "| GA-LLAMEA-8-INIT-100 | ../results/GA-LLAMEA-8-INIT-100 |\n"
    "\n"
    "IOH trajectory data is stored in ../results/ioh-all/{method}/{algo_id}/.\n"
    "Runs are skipped if the folder already exists so execution is resumable."
))

# ── MARKDOWN CELL 2 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 1. Setup"))

# ── CODE CELL 1 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "import os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from pandas.plotting import parallel_coordinates\n"
    "from IPython.display import display\n"
    "from tqdm.std import tqdm\n"
    "import polars as pl\n"
    "\n"
    "import ioh\n"
    "from ioh import logger as ioh_logger\n"
    "import iohinspector\n"
    "import iohblade as _iohblade\n"
    "from iohblade.loggers import ExperimentLogger\n"
    "from iohblade.behaviour_metrics import compute_behavior_metrics\n"
    "from iohblade.utils import budget_logger\n"
    "from iohblade import OverBudgetException\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'figure.figsize': (14, 9),\n"
    "    'font.size': 12,\n"
    "    'axes.grid': True,\n"
    "    'grid.alpha': 0.3,\n"
    "})\n"
    "sns.set_palette('colorblind')\n"
    "print('All imports successful.')"
))

# ── MARKDOWN CELL 3 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 2. Configuration"))

# ── CODE CELL 2 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "EXPERIMENT_DIRS = [\n"
    "    '../results/EoH',\n"
    "    '../results/CROSSOVER-ABLATION/baseline-LLAMEA',\n"
    "    '../results/GA-LLAMEA-8-INIT-100',\n"
    "]\n"
    "IOH_ALL_DIR = '../results/ioh-all'\n"
    "\n"
    "# Number of MA_BBOB instances and repetitions per algorithm for trajectory data.\n"
    "# The paper uses 5 training instances per BBOB function x 10 functions = 50 runs.\n"
    "# Reduce N_INSTANCES or N_REPS to speed up; increase for more statistical power.\n"
    "N_INSTANCES = 5   # instances 100..104\n"
    "N_REPS      = 2   # repetitions per instance\n"
    "\n"
    "DIM    = 5\n"
    "BUDGET = 2000 * DIM  # 10 000\n"
    "F_MIN, F_MAX = 1e-8, 1e2\n"
    "\n"
    "ALL_FEATS = [\n"
    "    'avg_nearest_neighbor_distance',\n"
    "    'dispersion',\n"
    "    'avg_exploration_pct',\n"
    "    'avg_distance_to_best',\n"
    "    'intensification_ratio',\n"
    "    'avg_exploitation_pct',\n"
    "    'average_convergence_rate',\n"
    "    'avg_improvement',\n"
    "    'success_rate',\n"
    "    'longest_no_improvement_streak',\n"
    "    'last_improvement_fraction',\n"
    "]\n"
    "\n"
    "STN_FEATS = [\n"
    "    'avg_exploration_pct',\n"
    "    'average_convergence_rate',\n"
    "    'avg_improvement',\n"
    "    'success_rate',\n"
    "    'longest_no_improvement_streak',\n"
    "]\n"
    "\n"
    "NICE_NAMES = {\n"
    "    'avg_nearest_neighbor_distance': 'NN-dist',\n"
    "    'dispersion':                    'Disp',\n"
    "    'avg_exploration_pct':           'Expl %',\n"
    "    'avg_distance_to_best':          'Dist->best',\n"
    "    'intensification_ratio':         'Inten-ratio',\n"
    "    'avg_exploitation_pct':          'Explt %',\n"
    "    'average_convergence_rate':      'Conv-rate',\n"
    "    'avg_improvement':               'Delta fitness',\n"
    "    'success_rate':                  'Success %',\n"
    "    'longest_no_improvement_streak': 'No-imp streak',\n"
    "    'last_improvement_fraction':     'Last-imp frac',\n"
    "    'aocc_norm':                     'Norm. fitness',\n"
    "}\n"
    "\n"
    "METHOD_COLORS = {\n"
    "    'EoH':                  '#E69F00',\n"
    "    'Baseline-LLaMEA':      '#56B4E9',\n"
    "    'GA-LLAMEA-8-INIT-100': '#009E73',\n"
    "}\n"
    "\n"
    "QUARTILE_LABELS = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']\n"
    "QUART_CAT = pd.CategoricalDtype(categories=QUARTILE_LABELS, ordered=True)\n"
    "REDUNDANCY_THRESHOLD = 0.7\n"
    "\n"
    "print('Configuration ready.')"
))

# ── MARKDOWN CELL 4 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 3. Load All Generated Algorithms\n"
    "\n"
    "Loads every algorithm produced across all LLaMEA/EoH runs (~500 per method).\n"
    "The fitness column is the AOCC score from the experiment log -- the same metric\n"
    "the paper uses as the performance signal."
))

# ── CODE CELL 3 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "exp_logger = ExperimentLogger(EXPERIMENT_DIRS[0], True)\n"
    "for d in EXPERIMENT_DIRS[1:]:\n"
    "    exp_logger.add_read_dir(d)\n"
    "\n"
    "methods, problems = exp_logger.get_methods_problems()\n"
    "print(f'Methods ({len(methods)}): {sorted(methods)}')\n"
    "\n"
    "all_algos = exp_logger.get_problem_data('MA_BBOB')\n"
    "all_algos.replace([-np.inf], 0, inplace=True)\n"
    "all_algos.fillna(0, inplace=True)\n"
    "all_algos = all_algos[all_algos['code'].notna() & (all_algos['code'] != '')].copy()\n"
    "\n"
    "print(f'Total algorithms: {len(all_algos)}')\n"
    "print()\n"
    "print(all_algos.groupby('method_name').size().rename('num_algorithms'))\n"
    "print()\n"
    "print('AOCC range per method:')\n"
    "print(all_algos.groupby('method_name')['fitness'].agg(['min','mean','max']).round(4))"
))

# ── MARKDOWN CELL 5 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 4. Load MA_BBOB Configuration"))

# ── CODE CELL 4 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "MABBOB_DIR = os.path.join(os.path.dirname(_iohblade.__file__), 'problems', 'mabbob')\n"
    "weights  = pd.read_csv(os.path.join(MABBOB_DIR, 'weights.csv'),  index_col=0)\n"
    "iids_df  = pd.read_csv(os.path.join(MABBOB_DIR, 'iids.csv'),     index_col=0)\n"
    "opt_locs = pd.read_csv(os.path.join(MABBOB_DIR, 'opt_locs.csv'), index_col=0)\n"
    "print(f'MA_BBOB config loaded from: {MABBOB_DIR}')"
))

# ── MARKDOWN CELL 6 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 5. Generate IOH Trajectory Data for All Algorithms\n"
    "\n"
    "Runs each algorithm on N_INSTANCES MA_BBOB instances x N_REPS repetitions\n"
    "and saves the trajectory to IOH_ALL_DIR/{method}/{algo_id}/.\n"
    "\n"
    "Skips any algorithm whose output folder already exists -- so you can\n"
    "interrupt and re-run this cell without losing progress.\n"
    "\n"
    "With N_INSTANCES=5, N_REPS=2 (~10 runs per algorithm x 1498 algorithms)\n"
    "expect 30-90 minutes depending on algorithm complexity."
))

# ── CODE CELL 5 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "import shutil\n"
    "\n"
    "os.makedirs(IOH_ALL_DIR, exist_ok=True)\n"
    "skipped = completed = errors = 0\n"
    "\n"
    "for method in sorted(methods):\n"
    "    method_algos = all_algos[all_algos['method_name'] == method]\n"
    "    method_dir   = os.path.join(IOH_ALL_DIR, method)\n"
    "    os.makedirs(method_dir, exist_ok=True)\n"
    "    print(f'\\n[{method}] -- {len(method_algos)} algorithms')\n"
    "\n"
    "    for _, row in tqdm(method_algos.iterrows(), total=len(method_algos), desc=method):\n"
    "        algo_id   = str(row['id'])\n"
    "        algo_name = str(row['name'])\n"
    "        alg_code  = str(row['code'])\n"
    "        out_dir   = os.path.join(method_dir, algo_id)\n"
    "\n"
    "        if os.path.isdir(out_dir):\n"
    "            skipped += 1\n"
    "            continue\n"
    "\n"
    "        try:\n"
    "            analyzer = ioh_logger.Analyzer(\n"
    "                triggers=[ioh_logger.trigger.ALWAYS],\n"
    "                folder_name=out_dir,\n"
    "                algorithm_name=algo_id,\n"
    "                store_positions=True,\n"
    "            )\n"
    "            bl = budget_logger(budget=BUDGET)\n"
    "            l1 = ioh_logger.Combine([bl, analyzer])\n"
    "\n"
    "            for iid in range(100, 100 + N_INSTANCES):\n"
    "                problem = ioh.problem.ManyAffine(\n"
    "                    xopt=np.array(opt_locs.iloc[iid])[:DIM],\n"
    "                    weights=np.array(weights.iloc[iid]),\n"
    "                    instances=np.array(iids_df.iloc[iid], dtype=int),\n"
    "                    n_variables=DIM,\n"
    "                )\n"
    "                problem.set_id(100)\n"
    "                problem.set_instance(iid)\n"
    "                problem.attach_logger(l1)\n"
    "\n"
    "                for rep in range(N_REPS):\n"
    "                    np.random.seed(rep)\n"
    "                    try:\n"
    "                        safe_globals = {'np': np}\n"
    "                        local_env    = {}\n"
    "                        exec(alg_code, safe_globals, local_env)\n"
    "                        algorithm = local_env[algo_name](budget=BUDGET, dim=DIM)\n"
    "                        algorithm(problem)\n"
    "                    except OverBudgetException:\n"
    "                        pass\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                    problem.reset()\n"
    "\n"
    "            completed += 1\n"
    "\n"
    "        except Exception as e:\n"
    "            errors += 1\n"
    "            if os.path.isdir(out_dir):\n"
    "                shutil.rmtree(out_dir)\n"
    "\n"
    "print(f'\\nDone. Completed: {completed}, Skipped: {skipped}, Errors: {errors}')"
))

# ── MARKDOWN CELL 7 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 6. Compute Behaviour Metrics per Algorithm\n"
    "\n"
    "Loads the trajectory for each algorithm and computes behaviour metrics.\n"
    "Metrics are aggregated (median) over all (instance, run_id) pairs,\n"
    "giving one behaviour vector per algorithm -- matching the paper's approach\n"
    "where each algorithm is one point in behaviour space.\n"
    "\n"
    "AOCC fitness is taken directly from the experiment log."
))

# ── CODE CELL 6 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "behaviour_records = []\n"
    "\n"
    "for method in sorted(methods):\n"
    "    method_algos = all_algos[all_algos['method_name'] == method]\n"
    "    method_dir   = os.path.join(IOH_ALL_DIR, method)\n"
    "    print(f'\\n  Loading: {method}')\n"
    "\n"
    "    for _, row in tqdm(method_algos.iterrows(), total=len(method_algos), desc=method):\n"
    "        algo_id = str(row['id'])\n"
    "        out_dir = os.path.join(method_dir, algo_id)\n"
    "\n"
    "        if not os.path.isdir(out_dir):\n"
    "            continue\n"
    "\n"
    "        try:\n"
    "            m = iohinspector.DataManager()\n"
    "            m.add_folder(out_dir)\n"
    "            df_traj = m.load(monotonic=False, include_meta_data=True).to_pandas()\n"
    "        except Exception:\n"
    "            continue\n"
    "\n"
    "        if df_traj.empty:\n"
    "            continue\n"
    "\n"
    "        run_metrics = []\n"
    "        for (inst, run), grp in df_traj.groupby(['instance', 'run_id']):\n"
    "            grp = grp.sort_values('evaluations').reset_index(drop=True)\n"
    "            if len(grp) < 10:\n"
    "                continue\n"
    "            try:\n"
    "                m_dict = compute_behavior_metrics(grp)\n"
    "            except Exception:\n"
    "                m_dict = {f: np.nan for f in ALL_FEATS}\n"
    "            run_metrics.append(m_dict)\n"
    "\n"
    "        if not run_metrics:\n"
    "            continue\n"
    "\n"
    "        # Median over all runs -- one point per algorithm (matching the paper)\n"
    "        agg = pd.DataFrame(run_metrics)[ALL_FEATS].median().to_dict()\n"
    "        agg['algo_id']     = algo_id\n"
    "        agg['method_name'] = method\n"
    "        agg['aocc']        = float(row['fitness'])\n"
    "        agg['generation']  = int(row['generation']) if pd.notna(row.get('generation')) else -1\n"
    "        behaviour_records.append(agg)\n"
    "\n"
    "df_beh = pd.DataFrame(behaviour_records)\n"
    "df_beh.replace([np.inf, -np.inf], np.nan, inplace=True)\n"
    "df_beh.fillna(0, inplace=True)\n"
    "\n"
    "print(f'\\nBehaviour DataFrame: {df_beh.shape}')\n"
    "print(df_beh.groupby('method_name').size().rename('algorithms_with_data'))\n"
    "df_beh[ALL_FEATS + ['aocc']].describe().round(4)"
))

# ── MARKDOWN CELL 8 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 7. Normalise AOCC Fitness\n"
    "\n"
    "The paper normalises AOCC to [0, 1] per method (Section 4.3)."
))

# ── CODE CELL 7 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "df = df_beh.copy()\n"
    "\n"
    "df['aocc_norm'] = (\n"
    "    df.groupby('method_name')['aocc']\n"
    "      .transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12))\n"
    ")\n"
    "\n"
    "print('Normalised AOCC per method:')\n"
    "print(df.groupby('method_name')[['aocc','aocc_norm']]\n"
    "        .agg(['min','mean','max']).round(4))"
))

# ── MARKDOWN CELL 9 ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 8. Step 1 -- Correlation Heatmap (Feature Selection)\n"
    "\n"
    "Replicates Figure 3 from the paper.\n"
    "\n"
    "This is a data-cleaning step, not a method comparison.\n"
    "It verifies the metrics are complementary and flags redundant pairs:\n"
    "- Expl % and Explt % (perfectly inversely correlated by design)\n"
    "- Expl % with Dist->best\n"
    "- Explt % with Inten-ratio\n"
    "- Last-imp frac with No-imp streak\n"
    "\n"
    "Only one feature from each pair is kept for the behaviour profile."
))

# ── CODE CELL 8 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "corr_cols = ALL_FEATS + ['aocc_norm']\n"
    "corr = df[corr_cols].corr()\n"
    "corr_plot = corr.rename(columns=NICE_NAMES, index=NICE_NAMES)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(16, 13))\n"
    "sns.heatmap(\n"
    "    corr_plot,\n"
    "    cmap='coolwarm', center=0, vmin=-1, vmax=1,\n"
    "    square=True, linewidths=0.5,\n"
    "    annot=True, fmt='.1f', annot_kws={'size': 9},\n"
    "    cbar_kws=dict(label='Pearson r'),\n"
    "    ax=ax,\n"
    ")\n"
    "ax.set_title('Behaviour metrics\\ncorrelation matrix', fontsize=14, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── MARKDOWN CELL 10 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "### 8a. Identify Redundant Feature Pairs\n"
    "\n"
    "Flags pairs with |r| >= threshold and recommends which to drop."
))

# ── CODE CELL 9 ──────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "feat_corr    = df[ALL_FEATS].corr()\n"
    "fitness_corr = df[ALL_FEATS + ['aocc_norm']].corr()['aocc_norm'].drop('aocc_norm').abs()\n"
    "\n"
    "redundant_pairs = []\n"
    "for i, f1 in enumerate(ALL_FEATS):\n"
    "    for j, f2 in enumerate(ALL_FEATS):\n"
    "        if j <= i:\n"
    "            continue\n"
    "        r = feat_corr.loc[f1, f2]\n"
    "        if abs(r) >= REDUNDANCY_THRESHOLD:\n"
    "            keep = f1 if fitness_corr[f1] >= fitness_corr[f2] else f2\n"
    "            drop = f2 if keep == f1 else f1\n"
    "            redundant_pairs.append({\n"
    "                'Feature A':        NICE_NAMES.get(f1, f1),\n"
    "                'Feature B':        NICE_NAMES.get(f2, f2),\n"
    "                'Pearson r':        round(r, 2),\n"
    "                'Recommended keep': NICE_NAMES.get(keep, keep),\n"
    "                'Recommended drop': NICE_NAMES.get(drop, drop),\n"
    "            })\n"
    "\n"
    "rp_df = pd.DataFrame(redundant_pairs)\n"
    "print(f'Redundant pairs (|r| >= {REDUNDANCY_THRESHOLD}):')\n"
    "display(rp_df)\n"
    "\n"
    "inv     = {v: k for k, v in NICE_NAMES.items()}\n"
    "to_drop = {inv.get(r['Recommended drop'], r['Recommended drop']) for r in redundant_pairs}\n"
    "selected_feats = [f for f in ALL_FEATS if f not in to_drop]\n"
    "\n"
    "print(f'\\nSelected non-redundant features ({len(selected_feats)}):')\n"
    "for f in selected_feats:\n"
    "    print(f'  {NICE_NAMES.get(f, f)}')\n"
    "\n"
    "print(\"\\nPaper's 5 selected features (for reference):\")\n"
    "for f in STN_FEATS:\n"
    "    print(f'  {NICE_NAMES.get(f, f)}')"
))

# ── MARKDOWN CELL 11 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 9. Step 2 -- Behaviour Profile (All Algorithms, Quartile Colour)\n"
    "\n"
    "Replicates Figure 4 (left panel) from the paper.\n"
    "\n"
    "Each polyline = one generated algorithm (aggregated over instances).\n"
    "Colour = AOCC quartile: Q1 blue = poor performers, Q4 dark red = best.\n"
    "The dark-red cluster is the gold standard behaviour profile.\n"
    "\n"
    "The paper found Q4 algorithms show: low NN-dist, average Disp,\n"
    "low Expl %, and focused exploitation behaviour."
))

# ── CODE CELL 10 ─────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "pc = df[ALL_FEATS + ['aocc_norm']].copy()\n"
    "\n"
    "for feat in ALL_FEATS:\n"
    "    lo, hi = pc[feat].min(), pc[feat].max()\n"
    "    pc[feat] = (pc[feat] - lo) / (hi - lo + 1e-12)\n"
    "\n"
    "pc['fitness_group'] = pd.qcut(\n"
    "    pc['aocc_norm'], 4, labels=QUARTILE_LABELS, duplicates='drop'\n"
    ").astype(QUART_CAT)\n"
    "pc = pc.sort_values('fitness_group', key=lambda s: s.cat.codes)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(18, 9))\n"
    "parallel_coordinates(\n"
    "    pc.rename(columns=NICE_NAMES), 'fitness_group',\n"
    "    ax=ax, alpha=0.2, linewidth=0.8, colormap='seismic',\n"
    ")\n"
    "ax.set_title(\n"
    "    'Behaviour profile over all generated algorithms\\n'\n"
    "    '(coloured by normalised AOCC quartile -- replicates Fig. 4 left)',\n"
    "    fontsize=13, fontweight='bold',\n"
    ")\n"
    "ax.set_ylabel('Scaled feature value')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── MARKDOWN CELL 12 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "### 9a. Top-100 Algorithms per Method\n"
    "\n"
    "Adapted from Figure 4 (right panel): top-100 per method instead of per function.\n"
    "Each line = one algorithm, coloured by method."
))

# ── CODE CELL 11 ─────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "TOP_K = 100\n"
    "method_list = sorted(df['method_name'].unique())\n"
    "\n"
    "pieces = []\n"
    "for method in method_list:\n"
    "    sub   = df[df['method_name'] == method].copy()\n"
    "    topk  = sub.nlargest(min(TOP_K, len(sub)), 'aocc')\n"
    "    frame = topk[ALL_FEATS + ['aocc_norm']].copy()\n"
    "    frame['method'] = method\n"
    "    pieces.append(frame)\n"
    "\n"
    "top_df = pd.concat(pieces, ignore_index=True)\n"
    "for feat in ALL_FEATS:\n"
    "    lo, hi = top_df[feat].min(), top_df[feat].max()\n"
    "    top_df[feat] = (top_df[feat] - lo) / (hi - lo + 1e-12)\n"
    "\n"
    "cmap   = plt.get_cmap('tab10')\n"
    "colors = [cmap(i) for i in range(len(method_list))]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(18, 9))\n"
    "parallel_coordinates(\n"
    "    top_df.rename(columns=NICE_NAMES), 'method',\n"
    "    ax=ax, alpha=0.6, linewidth=1.1, color=colors,\n"
    ")\n"
    "ax.set_title(\n"
    "    f'Top-{TOP_K} behaviour profiles per method\\n'\n"
    "    '(each line = one algorithm -- adapted from Fig. 4 right)',\n"
    "    fontsize=13, fontweight='bold',\n"
    ")\n"
    "ax.set_ylabel('Scaled feature value')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── MARKDOWN CELL 13 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "### 9b. Per-Method Behaviour Profile\n"
    "\n"
    "One plot per method, each algorithm coloured by its AOCC quartile within that method.\n"
    "Compare the Q4 (dark red) profile against the universal gold standard from Section 9."
))

# ── CODE CELL 12 ─────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, len(method_list),\n"
    "                          figsize=(18 * len(method_list) // 3, 9),\n"
    "                          sharey=False)\n"
    "if len(method_list) == 1:\n"
    "    axes = [axes]\n"
    "\n"
    "for ax, method in zip(axes, method_list):\n"
    "    sub = df[df['method_name'] == method][ALL_FEATS + ['aocc_norm']].copy()\n"
    "    for feat in ALL_FEATS:\n"
    "        lo, hi = sub[feat].min(), sub[feat].max()\n"
    "        sub[feat] = (sub[feat] - lo) / (hi - lo + 1e-12)\n"
    "\n"
    "    sub['fitness_group'] = pd.qcut(\n"
    "        sub['aocc_norm'], 4, labels=QUARTILE_LABELS, duplicates='drop'\n"
    "    ).astype(QUART_CAT)\n"
    "    sub = sub.sort_values('fitness_group', key=lambda s: s.cat.codes)\n"
    "\n"
    "    parallel_coordinates(\n"
    "        sub.rename(columns=NICE_NAMES), 'fitness_group',\n"
    "        ax=ax, alpha=0.3, linewidth=0.9, colormap='seismic',\n"
    "    )\n"
    "    ax.set_title(method, fontsize=11, fontweight='bold')\n"
    "    ax.set_ylabel('Scaled feature value')\n"
    "    ax.tick_params(axis='x', rotation=50)\n"
    "\n"
    "fig.suptitle('Behaviour Profile per Method -- coloured by AOCC quartile',\n"
    "             fontsize=13, fontweight='bold', y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── MARKDOWN CELL 14 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 10. Diagnosing Method Differences via the Gold Standard\n"
    "\n"
    "The paper's key analytical move (Sections 4.3 and 5):\n"
    "1. Read the Q4 median profile from Section 9 -- the gold standard.\n"
    "2. Compare each method's median algorithm behaviour against that standard.\n"
    "\n"
    "Example from the paper: LLaMEA-2's poor performance was attributed to its\n"
    "low success rate and high stagnation -- an overly exploratory profile."
))

# ── CODE CELL 13 ─────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "q4_threshold  = df['aocc_norm'].quantile(0.75)\n"
    "gold_standard = df[df['aocc_norm'] >= q4_threshold][ALL_FEATS].median().rename('Q4 Gold Standard')\n"
    "\n"
    "method_medians = df.groupby('method_name')[ALL_FEATS + ['aocc']].median()\n"
    "\n"
    "combined = method_medians[ALL_FEATS].T.copy()\n"
    "combined['Q4 Gold Standard'] = gold_standard\n"
    "combined.index = [NICE_NAMES.get(f, f) for f in combined.index]\n"
    "combined = combined.round(4)\n"
    "\n"
    "print('Median behaviour metrics per method vs. Q4 Gold Standard')\n"
    "display(\n"
    "    combined.style\n"
    "    .background_gradient(cmap='RdYlGn', axis=1)\n"
    "    .set_table_styles([\n"
    "        {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'),\n"
    "                                      ('font-weight', 'bold'), ('text-align', 'center')]},\n"
    "        {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '5px 10px')]},\n"
    "    ])\n"
    "    .format('{:.4f}')\n"
    ")\n"
    "\n"
    "print('\\nMedian AOCC per method:')\n"
    "print(method_medians[['aocc']].rename(columns={'aocc': 'Median AOCC'}).round(4))"
))

# ── MARKDOWN CELL 15 ────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 11. Behaviour Profile -- 5 Non-Redundant Features (Paper STN Metrics)\n"
    "\n"
    "Parallel coordinates using only the 5 least-correlated features from Section 8a,\n"
    "matching the paper's STN analysis: Expl %, Conv-rate, Delta fitness, Success %, No-imp streak."
))

# ── CODE CELL 14 ─────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "pc_stn = df[STN_FEATS + ['aocc_norm']].copy()\n"
    "\n"
    "for feat in STN_FEATS:\n"
    "    lo, hi = pc_stn[feat].min(), pc_stn[feat].max()\n"
    "    pc_stn[feat] = (pc_stn[feat] - lo) / (hi - lo + 1e-12)\n"
    "\n"
    "pc_stn['fitness_group'] = pd.qcut(\n"
    "    pc_stn['aocc_norm'], 4, labels=QUARTILE_LABELS, duplicates='drop'\n"
    ").astype(QUART_CAT)\n"
    "pc_stn = pc_stn.sort_values('fitness_group', key=lambda s: s.cat.codes)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 8))\n"
    "parallel_coordinates(\n"
    "    pc_stn.rename(columns=NICE_NAMES), 'fitness_group',\n"
    "    ax=ax, alpha=0.3, linewidth=1.0, colormap='seismic',\n"
    ")\n"
    "ax.set_title(\n"
    "    'Behaviour Profile -- 5 Non-Redundant Features\\n'\n"
    "    '(paper STN metrics, coloured by AOCC quartile)',\n"
    "    fontsize=13, fontweight='bold',\n"
    ")\n"
    "ax.set_ylabel('Scaled feature value')\n"
    "ax.tick_params(axis='x', rotation=20)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── MARKDOWN CELL 16 (summary) ───────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## 12. Summary\n"
    "\n"
    "| Step | What it does | Paper Figure |\n"
    "|------|--------------|--------------||\n"
    "| Section 8 Correlation heatmap | Feature selection -- identify redundant pairs | Fig. 3 |\n"
    "| Section 9 Parallel coords (all algorithms) | Gold standard behaviour profile | Fig. 4 left |\n"
    "| Section 9a Top-100 per method | Best algorithms behaviour by method | Fig. 4 right |\n"
    "| Section 9b Per-method parallel coords | Diagnose within-method variation | -- |\n"
    "| Section 10 Gold standard table | Quantify deviation from ideal profile | Sections 4.3/5 |\n"
    "| Section 11 STN-feature profile | 5-feature view matching paper STN analysis | Fig. 4 + Section 3.3 |"
))

# ---------------------------------------------------------------------------
# Build and write the notebook
# ---------------------------------------------------------------------------

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.10.0',
}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'behaviour_profile_and_correlation.ipynb')

# Write
with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    nbformat.write(nb, fh)

# Validate by reading back
with open(OUT_PATH, 'r', encoding='utf-8') as fh:
    nb_check = nbformat.read(fh, as_version=4)

print(f'Notebook written and validated: {len(nb_check.cells)} cells')
