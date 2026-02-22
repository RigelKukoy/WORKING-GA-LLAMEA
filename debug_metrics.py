import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys

sys.path.append(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA')
from iohblade.behaviour_metrics import compute_behavior_metrics

rows = []
log_path = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\results\COMPARISON-PROMPTS_20260208_202434\run-EoH-MA_BBOB-0\log.jsonl'
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

df_run = pd.DataFrame(rows)
df_run['fitness'] = pd.to_numeric(df_run['fitness'].astype(str).replace('-inf', '-inf'), errors='coerce')
df_run = df_run.dropna(subset=['fitness'])
df_run = df_run.reset_index(drop=True)
df_run['evaluations'] = df_run.index + 1
df_run['raw_y'] = df_run['fitness']
df = df_run.copy()

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

try:
    metrics = compute_behavior_metrics(df)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("------- df.dtypes ---------")
    print(df.dtypes)
    print("------- sample values ---------")
    for col in df.columns:
        print(f"{col}: type={type(df[col].iloc[0])} val={df[col].iloc[0]}")
