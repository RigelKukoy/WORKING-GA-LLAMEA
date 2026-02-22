import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys

sys.path.append(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA')
from iohblade.behaviour_metrics import compute_behavior_metrics

def safe_compute_metrics(df):
    x_cols = [c for c in df.columns if c.startswith('x')]
    if not x_cols and 'code' in df.columns:
        try:
            vectorizer = TfidfVectorizer(max_features=1000)
            code_data = df['code'].fillna('').astype(str)
            if len(code_data) > 1:
                X_tfidf = vectorizer.fit_transform(code_data)
                n_components = min(10, len(df) - 1, X_tfidf.shape[1] - 1)
                if n_components > 1:
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    X_reduced = svd.fit_transform(X_tfidf)
                    for i in range(n_components):
                        df[f'x{i}'] = X_reduced[:, i]
                    x_cols = [f'x{i}' for i in range(n_components)]
        except Exception as e:
            pass
    if not x_cols:
        return None
    try:
        metrics = compute_behavior_metrics(df)
        return metrics
    except Exception as e:
        print(f"Error Type: {type(e).__name__}: {str(e)}")
        print("------- df.dtypes ---------")
        print(df.dtypes)
        print("------- 5 rows of df ---------")
        print(df.head())
        # Let's find exactly which metric fails
        import traceback
        traceback.print_exc()
        return None

EXPERIMENT_DIR = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\results\COMPARISON-PROMPTS'
run_dirs = [d for d in os.listdir(EXPERIMENT_DIR) if d.startswith('run-')]
for run_dir in sorted(run_dirs):
    log_path = os.path.join(EXPERIMENT_DIR, run_dir, 'log.jsonl')
    if not os.path.exists(log_path): continue
    
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                sol = entry.get('solution', entry)
                rows.append({
                    'fitness': sol.get('fitness', None),
                    'code': sol.get('code', ''),
                    'name': sol.get('name', ''),
                    'generation': sol.get('generation', None),
                })
            except json.JSONDecodeError:
                continue

    if not rows: continue
    df_run = pd.DataFrame(rows)
    df_run['fitness'] = pd.to_numeric(df_run['fitness'].astype(str).replace('-inf', '-inf'), errors='coerce')
    df_run = df_run.dropna(subset=['fitness'])
    df_run = df_run.reset_index(drop=True)
    df_run['evaluations'] = df_run.index + 1
    df_run['raw_y'] = df_run['fitness']

    print(f"Testing {run_dir}...")
    metrics = safe_compute_metrics(df_run.copy())
    if metrics: print("  Success!")
    else: print("  Failed.")
    
