import sys
import os

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Check if Source directory exists and is importable
sys.path.append(os.path.join(os.getcwd(), 'Source'))

try:
    import utils
    import agent
    import environment
    import sim
    print("Source modules imported successfully.")
except ImportError as e:
    print(f"Source module import failed: {e}")
    sys.exit(1)

print("Environment verification passed.")
