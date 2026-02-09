
from iohblade.loggers import ExperimentLogger
import os

# Mimic the notebook environment
# Notebook is in examples/, data is in results/COMPARISON-PROMPTS
# So path is ../results/COMPARISON-PROMPTS

print(f"Current CWD: {os.getcwd()}")

try:
    logger = ExperimentLogger('../results/COMPARISON-PROMPTS', read=True)
    
    print("First call to get_methods_problems:")
    methods, problems = logger.get_methods_problems()
    print(f"Methods: {methods}")
    print(f"Problems: {problems}")
    
    # Simulate plot_convergence logic
    print("\nSimulating plot_convergence data retrieval:")
    for problem in problems:
        print(f"Retrieving data for problem: {problem}")
        data = logger.get_problem_data(problem_name=problem)
        print(f"Data shape: {data.shape}")
        if data.empty:
            print("WARNING: Data is empty for this problem!")
    
    print("\nSecond call to get_methods_problems (inside plot_convergence essentially):")
    methods2, problems2 = logger.get_methods_problems()
    print(f"Methods: {methods2}")
    print(f"Problems: {problems2}")
    
    if len(problems) > 0 and len(problems2) == 0:
        print("\nISSUE REPRODUCED: Second call returned empty lists!")
    elif len(problems) == 0:
        print("\nISSUE: First call returned empty lists! Path might be wrong.")
    else:
        print("\nBoth calls successful. Issue not reproduced in script.")

except Exception as e:
    print(f"An error occurred: {e}")
