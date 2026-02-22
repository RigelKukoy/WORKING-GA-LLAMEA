import nbformat
import sys

def fix_notebook(filepath):
    print(f"Fixing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            if "df_metrics.groupby('method').agg(['mean', 'std'])" in source:
                source = source.replace(
                    "df_metrics.groupby('method').agg(['mean', 'std'])",
                    "df_metrics.groupby('method')[[c for c in df_metrics.columns if c not in ('method', 'seed', 'run_dir')]].agg(['mean', 'std'])"
                )
                cell.source = source
                modified = True
            
            if "df_run['fitness'].astype(str).replace('-inf', '-inf')" in source:
                # User's weird cast line - let's make it robust and print errors
                new_lines = []
                for line in source.split('\n'):
                    if "df_run['fitness'].astype(str).replace('-inf', '-inf')" in line:
                        new_lines.append("    df_run['fitness'] = pd.to_numeric(df_run['fitness'], errors='coerce')")
                    elif "df_run['raw_y'] = df_run['fitness']" in line:
                        new_lines.append("    df_run['raw_y'] = df_run['fitness'].astype(float)")
                    elif "except Exception as e:" in line:
                        new_lines.append("    except Exception as e:")
                        new_lines.append("        import traceback")
                        new_lines.append("        traceback.print_exc()")
                    else:
                        new_lines.append(line)
                cell.source = '\n'.join(new_lines)
                modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Updated!")
    else:
        print("No changes needed or found.")

if __name__ == '__main__':
    fix_notebook(r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\examples\visualize_comparison_prompts.ipynb')
