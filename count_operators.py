import os
import json
from collections import Counter

EXPERIMENT_DIR = r'c:\Users\User\Desktop\GA-LLAMEA\WORKING-GA-LLAMEA\results\3ARM-VS-4ARM_20260219_022444'

def count_operators():
    run_dirs = [d for d in os.listdir(EXPERIMENT_DIR) if d.startswith('run-GA-LLAMEA')]
    
    for run_dir in sorted(run_dirs):
        log_path = os.path.join(EXPERIMENT_DIR, run_dir, 'log.jsonl')
        if not os.path.exists(log_path): continue
        
        ops = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    op = entry.get('operator')
                    if not op and 'metadata' in entry and isinstance(entry['metadata'], dict):
                        op = entry['metadata'].get('operator', 'unknown')
                    elif not op:
                        # try nested solution
                        sol = entry.get('solution', {})
                        op = sol.get('operator')
                        if not op and 'metadata' in sol and isinstance(sol['metadata'], dict):
                            op = sol['metadata'].get('operator', 'unknown')
                            
                    ops.append(op)
                except Exception as e:
                    pass
        
        c = Counter(ops)
        print(f"Run {run_dir}: {c}")

if __name__ == '__main__':
    count_operators()
