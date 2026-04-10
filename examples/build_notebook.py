import json
import uuid

def make_id():
    return uuid.uuid4().hex[:8]

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "id": make_id(),
        "metadata": {},
        "source": source_lines,
    }

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": make_id(),
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

cells = []

# CELL 0 - markdown
cells.append(md_cell([
    "# IOH Benchmark, Behaviour Metrics & Elo Rating\n",
    "\n",
    "Generates IOH data for the **best algorithm** from each of the three experiments used in\n",
    "`Final-Comparison-Visualization`, then computes:\n",
    "\n",
    "- **ECDF / AOCC** \u2013 empirical cumulative distribution and area-over-convergence-curve\n",
    "- **Behaviour Metrics** \u2013 per-run exploration/exploitation profile\n",
    "- **Elo Rating** \u2013 tournament-style ranking across algorithms\n",
    "\n",
    "**Experiments (Methods)**:\n",
    "| Method | Experiment Dir |\n",
    "|--------|----------------|\n",
    "| `EoH` | `../results/EoH` |\n",
    "| `Baseline-LLaMEA` | `../results/CROSSOVER-ABLATION/baseline-LLAMEA` |\n",
    "| `GA-LLAMEA-8-INIT-100` | `../results/GA-LLAMEA-8-INIT-100` |\n",
    "\n",
    "**Benchmark**: MA_BBOB, dim=5, budget=10\u202f000, instances 100\u2013149, 5 repetitions each.",
]))

# CELL 1 - markdown
cells.append(md_cell([
    "## 1. Setup",
]))

# CELL 2 - code
cells.append(code_cell([
    "import os\n",
    "import re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pandas.plotting import parallel_coordinates\n",
    "from tqdm.std import tqdm\n",
    "\n",
    "import ioh\n",
    "from ioh import logger as ioh_logger\n",
    "import iohinspector\n",
    "import polars as pl\n",
    "\n",
    "from iohblade.loggers import ExperimentLogger\n",
    "from iohblade.behaviour_metrics import compute_behavior_metrics\n",
    "from iohblade.utils import budget_logger\n",
    "from iohblade import OverBudgetException\n",
    "\n",
    "plt.rcParams.update({\n",
    "    'figure.figsize': (12, 8),\n",
    "    'font.size': 12,\n",
    "    'axes.grid': True,\n",
    "    'grid.alpha': 0.3,\n",
    "})\n",
    "sns.set_palette('colorblind')\n",
    "print('All imports successful.')",
]))

# CELL 3 - markdown
cells.append(md_cell([
    "## 2. Load Experiment Data",
]))

# CELL 4 - code
cells.append(code_cell([
    "EXPERIMENT_DIRS = [\n",
    "    '../results/EoH',\n",
    "    '../results/CROSSOVER-ABLATION/baseline-LLAMEA',\n",
    "    '../results/GA-LLAMEA-8-INIT-100',\n",
    "]\n",
    "\n",
    "# Where to write generated IOH data (one sub-folder per method)\n",
    "IOH_OUTPUT_DIR = '../results/ioh-best'\n",
    "\n",
    "# MA_BBOB configuration files (shipped with iohblade)\n",
    "import iohblade as _iohblade\n",
    "MABBOB_DIR = os.path.join(os.path.dirname(_iohblade.__file__), 'problems', 'mabbob')\n",
    "\n",
    "weights  = pd.read_csv(os.path.join(MABBOB_DIR, 'weights.csv'),  index_col=0)\n",
    "iids_df  = pd.read_csv(os.path.join(MABBOB_DIR, 'iids.csv'),     index_col=0)\n",
    "opt_locs = pd.read_csv(os.path.join(MABBOB_DIR, 'opt_locs.csv'), index_col=0)\n",
    "print(f'MA_BBOB data loaded from: {MABBOB_DIR}')\n",
    "print(f'weights={weights.shape}, iids={iids_df.shape}, opt_locs={opt_locs.shape}')",
]))

# CELL 5 - code
cells.append(code_cell([
    "exp_logger = ExperimentLogger(EXPERIMENT_DIRS[0], True)\n",
    "for extra_dir in EXPERIMENT_DIRS[1:]:\n",
    "    exp_logger.add_read_dir(extra_dir)\n",
    "\n",
    "methods, problems = exp_logger.get_methods_problems()\n",
    "print(f'Methods ({len(methods)}): {sorted(methods)}')\n",
    "print(f'Problems ({len(problems)}): {problems}')",
]))

# CELL 6 - markdown
cells.append(md_cell([
    "## 3. Identify Best Algorithm Per Method",
]))

# CELL 7 - code
cells.append(code_cell([
    "data = exp_logger.get_problem_data('MA_BBOB')\n",
    "data.replace([-np.inf], 0, inplace=True)\n",
    "data.fillna(0, inplace=True)\n",
    "\n",
    "best_rows = data.loc[data.groupby('method_name')['fitness'].idxmax()].copy()\n",
    "print('Best algorithms per method:')\n",
    "print(best_rows[['method_name', 'name', 'fitness', 'seed']].to_string(index=False))",
]))

# CELL 8 - markdown
cells.append(md_cell([
    "## 4. Generate IOH Benchmark Data\n",
    "\n",
    "Evaluates each best algorithm on MA_BBOB instances 100\u2013149 (dim=5, budget=10\u202f000, 5 reps).  \n",
    "Results are written to `../results/ioh-best/{method_name}/`.  \n",
    "**Skip if the folder already exists.**",
]))

# CELL 9 - code
cells.append(code_cell([
    "os.makedirs(IOH_OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "for _, row in best_rows.iterrows():\n",
    "    method_name    = row['method_name']\n",
    "    algorithm_name = row['name']\n",
    "    alg_code       = row['code']\n",
    "    out_dir        = os.path.join(IOH_OUTPUT_DIR, method_name)\n",
    "\n",
    "    if os.path.isdir(out_dir):\n",
    "        print(f'[SKIP] {method_name} \u2014 IOH data already exists at {out_dir}')\n",
    "        continue\n",
    "\n",
    "    print(f'\\nBenchmarking [{method_name}] \u2192 {algorithm_name}')\n",
    "\n",
    "    dim    = 5\n",
    "    budget = 2000 * dim  # 10 000\n",
    "\n",
    "    analyzer = ioh_logger.Analyzer(\n",
    "        triggers=[ioh_logger.trigger.ALWAYS],\n",
    "        folder_name=out_dir,\n",
    "        algorithm_name=method_name,\n",
    "        store_positions=True,\n",
    "    )\n",
    "    bl  = budget_logger(budget=budget)\n",
    "    l1  = ioh_logger.Combine([bl, analyzer])\n",
    "\n",
    "    for iid in tqdm(range(100, 150), desc=method_name):\n",
    "        problem = ioh.problem.ManyAffine(\n",
    "            xopt=np.array(opt_locs.iloc[iid])[:dim],\n",
    "            weights=np.array(weights.iloc[iid]),\n",
    "            instances=np.array(iids_df.iloc[iid], dtype=int),\n",
    "            n_variables=dim,\n",
    "        )\n",
    "        problem.set_id(100)\n",
    "        problem.set_instance(iid)\n",
    "        problem.attach_logger(l1)\n",
    "\n",
    "        for rep in range(5):\n",
    "            np.random.seed(rep)\n",
    "            try:\n",
    "                safe_globals = {'np': np}\n",
    "                local_env    = {}\n",
    "                exec(alg_code, safe_globals, local_env)\n",
    "                algorithm = local_env[algorithm_name](budget=budget, dim=dim)\n",
    "                algorithm(problem)\n",
    "            except OverBudgetException:\n",
    "                pass\n",
    "            except Exception:\n",
    "                pass\n",
    "            problem.reset()\n",
    "\n",
    "    print(f'  \u2192 Saved to {out_dir}')\n",
    "\n",
    "print('\\nIOH data generation complete.')",
]))

# CELL 10 - markdown
cells.append(md_cell([
    "## 5. Load IOH Data & ECDF / AOCC",
]))

# CELL 11 - code
cells.append(code_cell([
    "manager = iohinspector.DataManager()\n",
    "for method in sorted(methods):\n",
    "    d = os.path.join(IOH_OUTPUT_DIR, method)\n",
    "    if os.path.isdir(d):\n",
    "        manager.add_folder(d)\n",
    "        print(f'  Loaded: {d}')\n",
    "    else:\n",
    "        print(f'  [MISSING] {d}')\n",
    "\n",
    "df_ioh = manager.load(monotonic=True, include_meta_data=True)\n",
    "print(f'\\nTotal rows (monotonic): {len(df_ioh)}')\n",
    "print(f'Algorithms : {df_ioh[\"algorithm_name\"].unique().to_list()}')\n",
    "print(f'Dimensions : {df_ioh[\"dimension\"].unique().to_list()}')",
]))

# CELL 12 - code (FIX comment)
cells.append(code_cell([
    "# FIX: plot_ecdf creates its own fig/ax; passing ax= triggers an UnboundLocalError\n",
    "#      in the library. Use the returned ax to customise instead.\n",
    "ax, _ = iohinspector.plot_ecdf(\n",
    "    df_ioh.filter(pl.col('dimension') == 5),\n",
    "    f_max=100, f_min=1e-8, scale_eval_log=True,\n",
    ")\n",
    "ax.figure.set_size_inches(12, 8)\n",
    "ax.set_title('ECDF \u2014 MA_BBOB (dim=5)', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 13 - code (AOCC table)
cells.append(code_cell([
    "# AOCC table\n",
    "df_eaf = iohinspector.transform_fval(df_ioh, 1e-8, 1e2)\n",
    "aocc   = iohinspector.get_aocc(\n",
    "    df_eaf.filter(pl.col('dimension') == 5),\n",
    "    10000,\n",
    "    free_vars=['algorithm_name'],\n",
    ")\n",
    "print('AOCC per algorithm (dim=5):')\n",
    "print(aocc)",
]))

# CELL 14 - markdown
cells.append(md_cell([
    "## 6. Behaviour Metrics\n",
    "\n",
    "Loads the full (non-monotonic) trajectory data with positions and computes behaviour\n",
    "metrics per `(instance, run_id)` group, then aggregates to method level.",
]))

# CELL 15 - code
cells.append(code_cell([
    "BEHAVIOUR_FEATS = [\n",
    "    'avg_nearest_neighbor_distance',\n",
    "    'dispersion',\n",
    "    'avg_exploration_pct',\n",
    "    'avg_distance_to_best',\n",
    "    'intensification_ratio',\n",
    "    'avg_exploitation_pct',\n",
    "    'average_convergence_rate',\n",
    "    'avg_improvement',\n",
    "    'success_rate',\n",
    "    'longest_no_improvement_streak',\n",
    "    'last_improvement_fraction',\n",
    "]\n",
    "\n",
    "NICE_NAMES = {\n",
    "    'avg_nearest_neighbor_distance': 'NN-dist',\n",
    "    'dispersion':                    'Disp',\n",
    "    'avg_exploration_pct':           'Expl %',\n",
    "    'avg_distance_to_best':          'Dist\u2192best',\n",
    "    'intensification_ratio':         'Inten-ratio',\n",
    "    'avg_exploitation_pct':          'Explt %',\n",
    "    'average_convergence_rate':      'Conv-rate',\n",
    "    'avg_improvement':               '\u0394 fitness',\n",
    "    'success_rate':                  'Success %',\n",
    "    'longest_no_improvement_streak': 'No-imp streak',\n",
    "    'last_improvement_fraction':     'Last-imp frac',\n",
    "}\n",
    "\n",
    "print('Behaviour feature list ready.')",
]))

# CELL 16 - code (FIX comment)
cells.append(code_cell([
    "best_algo_map = dict(zip(best_rows['method_name'], best_rows['name']))\n",
    "\n",
    "records = []\n",
    "\n",
    "for method in sorted(methods):\n",
    "    out_dir = os.path.join(IOH_OUTPUT_DIR, method)\n",
    "    if not os.path.isdir(out_dir):\n",
    "        print(f'[SKIP] No IOH data for {method}')\n",
    "        continue\n",
    "\n",
    "    algo_name = best_algo_map.get(method, '?')\n",
    "    print(f'  {method} (best algorithm: {algo_name})')\n",
    "\n",
    "    m = iohinspector.DataManager()\n",
    "    m.add_folder(out_dir)\n",
    "    df_m = m.load(monotonic=False, include_meta_data=True).to_pandas()\n",
    "\n",
    "    x_cols = [c for c in df_m.columns if c.startswith('x')]\n",
    "    has_positions = len(x_cols) > 0\n",
    "\n",
    "    grouped = df_m.groupby(['instance', 'run_id'])\n",
    "    print(f'    -> {grouped.ngroups} runs ({\"positions available\" if has_positions else \"NO positions\"})')\n",
    "\n",
    "    for (inst, run), grp in tqdm(grouped, desc=method, leave=False):\n",
    "        grp = grp.sort_values('evaluations').reset_index(drop=True)\n",
    "        if len(grp) < 10:\n",
    "            continue\n",
    "        try:\n",
    "            metrics = compute_behavior_metrics(grp)\n",
    "        except Exception:\n",
    "            metrics = {f: np.nan for f in BEHAVIOUR_FEATS}\n",
    "        metrics['method_name'] = method\n",
    "        metrics['instance']    = inst\n",
    "        metrics['run_id']      = run\n",
    "        records.append(metrics)\n",
    "\n",
    "df_metrics = pd.DataFrame(records)\n",
    "keep_cols = BEHAVIOUR_FEATS + ['method_name', 'instance', 'run_id']\n",
    "df_metrics = df_metrics[[c for c in keep_cols if c in df_metrics.columns]]\n",
    "df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)\n",
    "\n",
    "print(f'\\nBehaviour metrics DataFrame: {df_metrics.shape}')\n",
    "print(f'Methods present: {df_metrics[\"method_name\"].unique().tolist()}')\n",
    "df_metrics.head()",
]))

# CELL 17 - markdown
cells.append(md_cell([
    "### 6a. Per-Method Aggregated Behaviour Metrics (Bar Chart)",
]))

# CELL 18 - code
cells.append(code_cell([
    "# Aggregate: median per method\n",
    "df_agg = (\n",
    "    df_metrics.groupby('method_name')[BEHAVIOUR_FEATS]\n",
    "    .median()\n",
    "    .reset_index()\n",
    ")\n",
    "\n",
    "n_feats   = len(BEHAVIOUR_FEATS)\n",
    "n_methods = len(df_agg)\n",
    "fig, axes = plt.subplots(3, 4, figsize=(18, 12))\n",
    "axes = axes.flatten()\n",
    "\n",
    "palette = sns.color_palette('colorblind', n_methods)\n",
    "\n",
    "for i, feat in enumerate(BEHAVIOUR_FEATS):\n",
    "    ax = axes[i]\n",
    "    vals = df_agg[feat].values\n",
    "    ax.bar(df_agg['method_name'], vals, color=palette)\n",
    "    ax.set_title(NICE_NAMES.get(feat, feat), fontsize=10, fontweight='bold')\n",
    "    ax.set_xticks(range(n_methods))\n",
    "    ax.set_xticklabels(df_agg['method_name'], rotation=30, ha='right', fontsize=8)\n",
    "    ax.set_ylabel('Median')\n",
    "\n",
    "for j in range(n_feats, len(axes)):\n",
    "    axes[j].set_visible(False)\n",
    "\n",
    "fig.suptitle('Behaviour Metrics \u2014 Median per Method (MA_BBOB, dim=5)', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 19 - markdown
cells.append(md_cell([
    "### 6b. Parallel Coordinates \u2014 Behaviour Profile",
]))

# CELL 20 - code
cells.append(code_cell([
    "# Min-max scale features so all axes are 0-1\n",
    "pc = df_metrics[BEHAVIOUR_FEATS + ['method_name']].dropna().copy()\n",
    "\n",
    "for feat in BEHAVIOUR_FEATS:\n",
    "    col_min, col_max = pc[feat].min(), pc[feat].max()\n",
    "    if col_max > col_min:\n",
    "        pc[feat] = (pc[feat] - col_min) / (col_max - col_min)\n",
    "    else:\n",
    "        pc[feat] = 0.0\n",
    "\n",
    "pc_renamed = pc.rename(columns=NICE_NAMES)\n",
    "method_col = 'method_name'\n",
    "\n",
    "color_map = dict(zip(sorted(methods), [f'C{i}' for i in range(len(methods))]))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(16, 8))\n",
    "parallel_coordinates(\n",
    "    pc_renamed, method_col,\n",
    "    ax=ax, alpha=0.3, linewidth=0.8,\n",
    "    color=[color_map[m] for m in pc[method_col]],\n",
    ")\n",
    "ax.set_title('Behaviour Profile \u2014 Parallel Coordinates (scaled 0-1)', fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Scaled feature value')\n",
    "ax.tick_params(axis='x', rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 21 - markdown
cells.append(md_cell([
    "### 6c. Correlation Heatmap",
]))

# CELL 22 - code
cells.append(code_cell([
    "corr = df_metrics[BEHAVIOUR_FEATS].corr()\n",
    "corr_renamed = corr.rename(columns=NICE_NAMES, index=NICE_NAMES)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 12))\n",
    "sns.heatmap(\n",
    "    corr_renamed,\n",
    "    cmap='coolwarm', center=0, vmin=-1, vmax=1,\n",
    "    square=True, linewidths=0.5,\n",
    "    cbar_kws=dict(label='Pearson r'),\n",
    "    ax=ax,\n",
    ")\n",
    "ax.set_title('Behaviour Metrics \u2014 Correlation Matrix', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 23 - markdown
cells.append(md_cell([
    "### 6d. Exploration vs Exploitation per Method (Violin)",
]))

# CELL 24 - code
cells.append(code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "for ax, feat, title in zip(\n",
    "    axes,\n",
    "    ['avg_exploration_pct', 'avg_exploitation_pct'],\n",
    "    ['Exploration %', 'Exploitation %'],\n",
    "):\n",
    "    sns.violinplot(\n",
    "        data=df_metrics, x='method_name', y=feat,\n",
    "        palette='colorblind', inner='box', ax=ax,\n",
    "    )\n",
    "    ax.set_title(title, fontsize=12, fontweight='bold')\n",
    "    ax.set_xlabel('')\n",
    "    ax.tick_params(axis='x', rotation=30)\n",
    "\n",
    "fig.suptitle('Exploration / Exploitation Distribution per Method', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 25 - markdown
cells.append(md_cell([
    "### 6e. No-Improvement Streak & Success Rate",
]))

# CELL 26 - code
cells.append(code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "for ax, feat, title in zip(\n",
    "    axes,\n",
    "    ['longest_no_improvement_streak', 'success_rate'],\n",
    "    ['Longest No-Improvement Streak', 'Success Rate'],\n",
    "):\n",
    "    sns.boxplot(\n",
    "        data=df_metrics, x='method_name', y=feat,\n",
    "        palette='colorblind', ax=ax,\n",
    "    )\n",
    "    ax.set_title(title, fontsize=12, fontweight='bold')\n",
    "    ax.set_xlabel('')\n",
    "    ax.tick_params(axis='x', rotation=30)\n",
    "\n",
    "fig.suptitle('Stagnation & Success per Method', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 27 - markdown
cells.append(md_cell([
    "## 7. Elo Rating (Tournament Ranking)\n",
    "\n",
    "Simulates 100\u202f000 rounds of pairwise tournament matches using the full IOH run data.\n",
    "A higher Elo rating means the algorithm consistently outperforms its opponents.",
]))

# CELL 28 - code
cells.append(code_cell([
    "# Reload monotonic data (needed for tournament ranking)\n",
    "manager_elo = iohinspector.DataManager()\n",
    "for method in sorted(methods):\n",
    "    d = os.path.join(IOH_OUTPUT_DIR, method)\n",
    "    if os.path.isdir(d):\n",
    "        manager_elo.add_folder(d)\n",
    "\n",
    "df_elo_input = manager_elo.load(monotonic=True, include_meta_data=True)\n",
    "print(f'Loaded {len(df_elo_input)} rows for Elo computation')\n",
    "print(f'Algorithms: {df_elo_input[\"algorithm_name\"].unique().to_list()}')",
]))

# CELL 29 - code
cells.append(code_cell([
    "dt_elo = iohinspector.get_tournament_ratings(df_elo_input, nrounds=100000)\n",
    "dt_elo['Rating']    = pd.to_numeric(dt_elo['Rating'],    errors='coerce')\n",
    "dt_elo['Deviation'] = pd.to_numeric(dt_elo['Deviation'], errors='coerce').fillna(0)\n",
    "\n",
    "print('Elo Ratings (sorted, best first):')\n",
    "print(dt_elo.sort_values('Rating', ascending=False).to_string(index=False))",
]))

# CELL 30 - code
cells.append(code_cell([
    "dt_elo_sorted = dt_elo.sort_values('algorithm_name').reset_index(drop=True)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 7))\n",
    "sns.pointplot(\n",
    "    data=dt_elo_sorted, x='algorithm_name', y='Rating',\n",
    "    linestyle='none', ax=ax, color='steelblue',\n",
    ")\n",
    "ax.errorbar(\n",
    "    dt_elo_sorted['algorithm_name'],\n",
    "    dt_elo_sorted['Rating'],\n",
    "    yerr=dt_elo_sorted['Deviation'],\n",
    "    fmt='o', color='steelblue', alpha=0.8, capsize=7, elinewidth=2.5,\n",
    ")\n",
    "ax.tick_params(axis='x', rotation=30)\n",
    "ax.set_xlabel('')\n",
    "ax.set_title('Tournament Ranking (Elo Rating) \u2014 MA_BBOB (dim=5)', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 31 - code
cells.append(code_cell([
    "# Bar chart version \u2014 sorted by rating\n",
    "dt_elo_ranked = dt_elo.sort_values('Rating', ascending=True).reset_index(drop=True)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "colors = sns.color_palette('colorblind', len(dt_elo_ranked))\n",
    "ax.barh(\n",
    "    dt_elo_ranked['algorithm_name'],\n",
    "    dt_elo_ranked['Rating'],\n",
    "    xerr=dt_elo_ranked['Deviation'],\n",
    "    color=colors, alpha=0.85, capsize=4, ecolor='gray',\n",
    ")\n",
    "ax.set_xlabel('Elo Rating')\n",
    "ax.set_title('Algorithm Ranking by Elo Score', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

# CELL 32 - markdown
cells.append(md_cell([
    "## 8. Summary Table",
]))

# CELL 33 - code (FIX comment)
cells.append(code_cell([
    "# Combine AOCC + Elo into one summary\n",
    "# FIX: aocc is already a pandas DataFrame (get_aocc has return_as_pandas=True by default)\n",
    "aocc_pd = aocc.rename(columns={'algorithm_name': 'Method', 'AOCC': 'AOCC (median)'})\n",
    "elo_pd  = dt_elo.rename(columns={'algorithm_name': 'Method'})[['Method', 'Rating', 'Deviation']]\n",
    "elo_pd.columns = ['Method', 'Elo Rating', 'Elo Deviation']\n",
    "\n",
    "summary = aocc_pd.merge(elo_pd, on='Method', how='outer').sort_values('Elo Rating', ascending=False)\n",
    "summary[['AOCC (median)', 'Elo Rating', 'Elo Deviation']] = summary[\n",
    "    ['AOCC (median)', 'Elo Rating', 'Elo Deviation']\n",
    "].round(4)\n",
    "print('Summary \u2014 AOCC & Elo Ratings:')\n",
    "print(summary.to_string(index=False))",
]))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

output_path = r"c:\Users\Kukoy\Documents\Experiment-GA\WORKING-GA-LLAMEA\examples\IOH-Behaviour-Elo-Analysis-Fixed.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)}")
