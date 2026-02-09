
from iohblade.loggers import ExperimentLogger
import iohblade
import os

# Create a dummy logger dir for testing
# We'll rely on the existing one since we know it exists: ../results/COMPARISON-PROMPTS
path = '../results/COMPARISON-PROMPTS'

try:
    with open('examples/test_result.txt', 'w') as f:
        f.write(f"iohblade location: {iohblade.__file__}\n")
        f.write(f"ExperimentLogger location: {ExperimentLogger.__module__}\n")
        f.write(f"Original CWD: {os.getcwd()}\n")
        logger = ExperimentLogger(path, read=True)
        
        methods, problems = logger.get_methods_problems()
        f.write(f"Call 1 (Original CWD): Found {len(problems)} problems.\n")
        
        # Change CWD
        new_cwd = os.path.dirname(os.getcwd()) # Go up one level
        os.chdir(new_cwd)
        f.write(f"Changed CWD to: {os.getcwd()}\n")
        
        methods, problems = logger.get_methods_problems()
        f.write(f"Call 2 (New CWD): Found {len(problems)} problems.\n")
        
        if len(problems) == 0:
            f.write("ISSUE REPRODUCED: get_methods_problems failed after CWD change with relative path!\n")
            f.write("FAIL\n")
        else:
            f.write("SUCCESS: Found problems after CWD change.\n")
            f.write("PASS\n")

except Exception as e:
    with open('examples/test_result.txt', 'w') as f:
        f.write(f"An error occurred: {e}\n")
        f.write("FAIL\n")
