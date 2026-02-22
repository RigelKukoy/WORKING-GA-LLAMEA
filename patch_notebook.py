import json

with open(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_3arm_vs_4arm_fixed.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_source = [
    "# Custom convergence plot with padding to 100 generations\n",
    "if len(df_experiment) > 0 and len(methods) > 0:\n",
    "    print('Convergence Plot')\n",
    "    try:\n",
    "        all_convergence_data = []\n",
    "        for run_name, data in convergence_data.items():\n",
    "            method = data['method']\n",
    "            run_conv = data['convergence']\n",
    "            if not run_conv:\n",
    "                continue\n",
    "            gen_dict = {item['generation']: item['best_so_far'] for item in run_conv}\n",
    "            current_best = float('-inf')\n",
    "            for gen in range(100):\n",
    "                if gen in gen_dict:\n",
    "                    current_best = gen_dict[gen]\n",
    "                if current_best != float('-inf'):\n",
    "                    all_convergence_data.append({\n",
    "                        'Generation': gen,\n",
    "                        'Best Fitness (AOCC)': current_best,\n",
    "                        'Method': method,\n",
    "                        'Run': run_name\n",
    "                    })\n",
    "        df_conv = pd.DataFrame(all_convergence_data)\n",
    "        if len(df_conv) > 0:\n",
    "            plt.figure(figsize=(14, 8))\n",
    "            ax = sns.lineplot(data=df_conv, x='Generation', y='Best Fitness (AOCC)', hue='Method', palette={'GA-LLAMEA-3arm': 'blue', 'GA-LLAMEA-4arm': 'red'})\n",
    "            plt.title('Convergence Comparison: 3ARM vs 4ARM')\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "        else:\n",
    "            print('No valid data.')\n",
    "    except Exception as e:\n",
    "        import traceback\n",
    "        traceback.print_exc()\n",
    "        print(f'Error plotting convergence: {e}')\n",
    "else:\n",
    "    print('No data available for convergence plot')"
]

# Find the corresponding cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        if len(cell['source']) > 0 and 'Plot convergence curves' in cell['source'][0]:
            print('Found cell, replacing source')
            cell['source'] = new_source
            break

with open(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_3arm_vs_4arm_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Saved Jupyter Notebook")
