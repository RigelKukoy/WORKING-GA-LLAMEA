import nbformat
import glob
import os

def patch_nb(filename):
    print(f'Patching {filename}')
    with open(filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    changed = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            src = cell.source
            if "df_metrics.groupby('method').agg(['mean', 'std'])" in src:
                src = src.replace(
                    "df_metrics.groupby('method').agg(['mean', 'std'])",
                    "df_metrics.groupby('method')[metric_cols].agg(['mean', 'std'])"
                )
                cell.source = src
                changed = True
                
            if "df_run = pd.DataFrame(rows)" in src:
                # Ensure fitness is converted correctly
                if "to_numeric" not in src:
                    src = src.replace(
                        "df_run = pd.DataFrame(rows)",
                        "df_run = pd.DataFrame(rows)\n    df_run['fitness'] = pd.to_numeric(df_run['fitness'].astype(str).replace('-inf', '-inf'), errors='coerce')"
                    )
                    cell.source = src
                    changed = True
                
                # Check for raw_y assignment and astype float
                if "df_run['raw_y'] = df_run['fitness']" in src and ".astype(float)" not in src:
                    src = src.replace("df_run['raw_y'] = df_run['fitness']", "df_run['raw_y'] = df_run['fitness'].astype(float)")
                    cell.source = src
                    changed = True

    if changed:
        with open(filename, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print('Saved changes.')
    else:
        print('No changes needed.')

if __name__ == '__main__':
    patch_nb(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_3arm_vs_4arm.ipynb')
    patch_nb(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_comparison_prompts.ipynb')
