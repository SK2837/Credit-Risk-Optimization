# Credit Risk Optimization with Reinforcement Learning

## What the project does
This repository contains a **research prototype** that shows how a reinforcement‑learning (RL) agent can be used to **dynamically optimise the acceptance threshold** in a credit‑scoring pipeline.

- **Traditional approach**: Choose a static cutoff score (e.g., 65) that maximises profit on a historical test set.
- **RL approach**: Treat each week as a decision step. The agent observes the current acceptance rate, selects a new cutoff, receives the profit (or loss) as a reward, and updates its policy. Over time it learns to adapt the threshold to changing market conditions (e.g., shifts in default rates or credit‑score distributions).

The goal is to demonstrate that a simple Q‑learning agent can **outperform the static optimisation** both in simulated environments and on a small real‑world data set.

## Quick start

```bash
# 1. Install the required Python packages
pip install -r requirements.txt

# 2. Run the experiment (headless mode – plots are saved, not shown)
export MPLBACKEND=Agg
python3 Source/RL_experiment.py
```

The script will:
1. **Create a simulation environment** (`simulation.py`) that mimics loan applications, defaults, repayments, and optional “distortions” (e.g., a sudden increase in default rates).
2. **Train a Q‑learning agent** for 100 episodes (each episode = 114 simulated weeks).
3. **Evaluate** the trained agent on both the original and distorted environments.
4. **Save plots** (learning curves, Q‑value heatmaps, performance comparisons) under the `bookkeeping/` directory.

## Project layout

```
Credit_Risk_Optimization/
├─ README.md                # ← you are reading this file
├─ requirements.txt         # Python dependencies
├─ bookkeeping/            # Generated results and plots (created at runtime)
├─ Source/
│   ├─ agent.py            # Q‑learning agent implementation
│   ├─ model.py            # Feature transformer & SGD models
│   ├─ simulation.py       # Gym‑compatible environment (loan‑business simulator)
│   ├─ manager.py          # Orchestrates training, testing, and visualisation
│   ├─ RL_experiment.py    # Main entry point (converted from the original notebook)
│   └─ ... (utility files) # e.g., environment.py, policy.py, etc.
└─ ... (other artefacts)
```

### Key modules

| Module | Purpose |
|--------|---------|
| **simulation.py** | Implements `SimulationEnv`, a `gym.Env` that produces weekly state vectors (acceptance rate) and rewards (profit). It can be configured with *distortions* to simulate market shifts. |
| **agent.py** | Contains the `Agent` class that holds the Q‑learning logic, interacts with the environment, and stores the learned value‑function model. |
| **model.py** | Provides `FeatureTransformer` (RBF sampler) and `Model`/`EnvironmentModel` (SGD regressors) that approximate the Q‑values for each discrete action. |
| **policy.py** | Implements action‑selection strategies (e.g., Boltzmann‑Q for exploration, greedy for testing). |
| **manager.py** | High‑level wrapper that creates the environment, agent, runs training loops, evaluates performance, and produces plots. |
| **RL_experiment.py** | The script you run; it sets hyper‑parameters, creates a `Manager`, and calls the training/evaluation pipeline. |

## How the RL loop works
1. **State** – a single scalar: the acceptance rate observed in the previous week (0 → 1).
2. **Action** – a discrete threshold value (5, 10, …, 100) that the agent will apply for the next week.
3. **Reward** – the profit earned during the week (interest earned minus defaults).
4. **Transition** – the environment updates its internal loan‑application simulation, applies the chosen threshold, and returns the new state and reward.

The agent uses a **Q‑learning update**:
```
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```
where `α` is the learning rate and `γ` the discount factor. The state is first transformed by an RBF feature map to capture non‑linearity, then a linear SGD model predicts Q‑values for each action.

## Distorted scenarios
After the initial training, the script can *distort* the environment (e.g., increase default rates, shift credit‑score distributions). This tests the agent’s ability to **adapt** without retraining from scratch. The `distortions` dictionary passed to `SimulationEnv` controls these changes.

## Important notes
- The code was originally written for an older `gym` version (0.26). It has been patched to work with the current environment, but if you upgrade to `gymnasium` you may need minor import changes.
- The simulation uses **synthetic data** (dummy values) for privacy; results are illustrative rather than production‑grade.
- Plots are saved automatically; because we set `MPLBACKEND=Agg`, they are not displayed interactively. Open the generated PNG/PDF files in `bookkeeping/` to view them.

## Getting help
If you run into import errors or missing packages, double‑check `requirements.txt` and ensure you are using Python 3.11 (or later). Feel free to open an issue or modify the hyper‑parameters in `RL_experiment.py` to experiment with different learning rates, discount factors, or number of training episodes.

---
**Happy experimenting!** If you need further clarification on any part of the code or want to extend the project, just let me know.
