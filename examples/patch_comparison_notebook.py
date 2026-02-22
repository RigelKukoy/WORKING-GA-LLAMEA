import json
import pandas as pd
import numpy as np

# Test if pd.to_numeric parses '-inf' correctly
s = pd.Series(['-inf', 1.5, 'invalid'])
res = pd.to_numeric(s, errors='coerce')
print("Numeric parsing logic test:")
print(res)

notebook_path = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_comparison_prompts.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            if "df_run = df_run.dropna(subset=['fitness'])" in line and not modified:
                # Add line to convert string values to numeric and handle -inf
                new_source.append("    df_run['fitness'] = pd.to_numeric(df_run['fitness'].astype(str).replace('-inf', '-inf'), errors='coerce')\n")
                modified = True
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Patch applied.")
