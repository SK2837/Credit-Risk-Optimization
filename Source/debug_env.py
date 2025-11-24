from simulation import SimulationEnv
import numpy as np
import sys
import os

# Add current directory to path so imports work
sys.path.append(os.getcwd())

try:
    print("Initializing environment...")
    env = SimulationEnv()
    print("Environment initialized.")
    
    print("Resetting environment...")
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    if np.isnan(obs).any():
        print("CRITICAL: Initial observation contains NaN!")
    else:
        print("Initial observation is valid.")
    
    print("Taking a step...")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step 1 observation: {obs}")
    
    if np.isnan(obs).any():
        print("CRITICAL: Step 1 observation contains NaN!")
    else:
        print("Step 1 observation is valid.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
