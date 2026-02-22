import json

file_path = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_3arm_vs_4arm_fixed.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if "best_fitness = float('inf')" in line:
                line = line.replace("float('inf')", "float('-inf')")
            if "if fitness < best_fitness:" in line:
                line = line.replace("if fitness < best_fitness:", "if fitness > best_fitness:")
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook fixed.")
