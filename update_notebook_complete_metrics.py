
import json

notebook_paths = ["examples/visualize_comparison.ipynb", "examples/visualize_ga_llamea_metrics.ipynb"]

new_source = [
    "from iohblade.behaviour_metrics import compute_behavior_metrics, average_convergence_rate, improvement_statistics, longest_no_improvement_streak, last_improvement_fraction\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "import numpy as np\n",
    "\n",
    "def safe_compute_metrics(df):\n",
    "    # Ensure required columns exist\n",
    "    if 'raw_y' not in df.columns and 'fitness' in df.columns:\n",
    "        df['raw_y'] = df['fitness']\n",
    "    if 'evaluations' not in df.columns:\n",
    "        df = df.reset_index(drop=True)\n",
    "        df['evaluations'] = df.index + 1\n",
    "\n",
    "    # Check for x columns\n",
    "    x_cols = [c for c in df.columns if c.startswith(\"x\")]\n",
    "    \n",
    "    # If no x columns but we have code, project code to latent space\n",
    "    if not x_cols and 'code' in df.columns:\n",
    "        try:\n",
    "            # 1. Vectorize code\n",
    "            # Use TfidfVectorizer to capture semantic similarity of keywords/structure\n",
    "            vectorizer = TfidfVectorizer(max_features=1000)\n",
    "            # Ensure string type and handle missing\n",
    "            code_data = df['code'].fillna('').astype(str)\n",
    "            if len(code_data) > 1:\n",
    "                X_tfidf = vectorizer.fit_transform(code_data)\n",
    "                \n",
    "                # 2. Reduce dimensionality (Project to \"Coordinate Space\")\n",
    "                # Use 10 dimensions or less if fewer samples. SVD requires n_components < n_samples\n",
    "                n_components = min(10, len(df) - 1, X_tfidf.shape[1] - 1)\n",
    "                if n_components > 1:\n",
    "                    svd = TruncatedSVD(n_components=n_components, random_state=42)\n",
    "                    X_reduced = svd.fit_transform(X_tfidf)\n",
    "                    \n",
    "                    # 3. Assign as x coordinates\n",
    "                    for i in range(n_components):\n",
    "                        df[f'x{i}'] = X_reduced[:, i]\n",
    "                    \n",
    "                    x_cols = [f'x{i}' for i in range(n_components)]\n",
    "        except Exception as e:\n",
    "            # print(f\"Warning: Code projection failed: {e}\")\n",
    "            pass\n",
    "\n",
    "    if x_cols:\n",
    "        # Some metrics might fail if bounds are expected but random X has arbitrary range\n",
    "        # We let compute_behavior_metrics handle defaults (it infers bounds from data min/max if not provided)\n",
    "        return compute_behavior_metrics(df)\n",
    "    \n",
    "    # Compute limit metrics (only y-based) if projection failed or no code\n",
    "    avg_imp, success_rate = improvement_statistics(df)\n",
    "    metrics = {\n",
    "        \"average_convergence_rate\": average_convergence_rate(df),\n",
    "        \"avg_improvement\": avg_imp,\n",
    "        \"success_rate\": success_rate,\n",
    "        \"longest_no_improvement_streak\": longest_no_improvement_streak(df),\n",
    "        \"last_improvement_fraction\": last_improvement_fraction(df),\n",
    "        \"avg_nearest_neighbor_distance\": float(\"nan\"),\n",
    "        \"dispersion\": float(\"nan\"),\n",
    "        \"avg_exploration_pct\": float(\"nan\"),\n",
    "        \"avg_distance_to_best\": float(\"nan\"),\n",
    "        \"intensification_ratio\": float(\"nan\"),\n",
    "        \"avg_exploitation_pct\": float(\"nan\"),\n",
    "    }\n",
    "    return metrics\n",
    "\n",
    "def display_metrics(logger, name):\n",
    "    print(f\"--- {name} Behavior Metrics ---\")\n",
    "    methods, problems = logger.get_methods_problems()\n",
    "    all_metrics = []\n",
    "\n",
    "    for problem in problems:\n",
    "        try:\n",
    "            df_problem = logger.get_problem_data(problem)\n",
    "            # Group by method and seed (run)\n",
    "            grouped = df_problem.groupby([\"method_name\", \"seed\"])\n",
    "            \n",
    "            for (method, seed), gdf in grouped:\n",
    "                gdf = gdf.copy()\n",
    "                m = safe_compute_metrics(gdf)\n",
    "                m['method'] = method\n",
    "                m['problem'] = problem\n",
    "                m['seed'] = seed\n",
    "                all_metrics.append(m)\n",
    "        except Exception as e:\n",
    "            # print(f\"Error processing problem {problem}: {e}\")\n",
    "            pass\n",
    "\n",
    "    if not all_metrics:\n",
    "        print(\"No metrics computed.\")\n",
    "        return\n",
    "\n",
    "    df_metrics = pd.DataFrame(all_metrics)\n",
    "    # Aggregate by method\n",
    "    df_agg = df_metrics.groupby(\"method\").mean(numeric_only=True)\n",
    "    return df_agg"
]

for nb_path in notebook_paths:
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
        
        # Find cell with safe_compute_metrics
        found = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source_str = "".join(cell['source'])
                if "def safe_compute_metrics" in source_str:
                    cell['source'] = new_source
                    found = True
                
                # Update data loading to use new path
                if "logger_mabbob = ExperimentLogger" in source_str or "logger_ga = ExperimentLogger" in source_str:
                    cell['source'] = [
                        "# Load Comparison Results\n",
                        "logger = ExperimentLogger('../results/COMPARISON-PROMPTS', True)\n",
                        "methods, problems = logger.get_methods_problems()\n",
                        "print(\"Methods found:\", methods)\n",
                        "print(\"Problems found:\", problems)"
                    ]
                    found = True
        
        if found:
            with open(nb_path, 'w') as f:
                json.dump(nb, f, indent=4)
            print(f"Updated {nb_path}")
        else:
            print(f"Could not find function to update in {nb_path}")
            
    except Exception as e:
        print(f"Error updating {nb_path}: {e}")
