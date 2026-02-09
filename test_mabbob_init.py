
from iohblade.problems import MA_BBOB
from iohblade.loggers import ExperimentLogger
import pandas as pd
import os

def test_init():
    print("Testing Pandas version...")
    print(pd.__version__)

    print("\nTesting MA_BBOB init...")
    try:
        prob = MA_BBOB()
        print("MA_BBOB initialized successfully.")
    except Exception as e:
        print(f"MA_BBOB init failed: {e}")
        return

    print("\nTesting ExperimentLogger init...")
    try:
        logger = ExperimentLogger("results/TEST_LOGGER_INIT")
        print("ExperimentLogger initialized successfully.")
    except Exception as e:
        print(f"ExperimentLogger init failed: {e}")

if __name__ == "__main__":
    test_init()
