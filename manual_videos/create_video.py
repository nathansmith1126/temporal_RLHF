import pandas as pd

# Import Minigrid environment
from Minigrid.minigrid.envs.test_envs import TestEnv
from Minigrid.minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions

# Import custom DFA code.
from AUTOMATA.auto_funcs import DFAMonitor, dfa_T1


# Instantiate a door and key minigrid environment.
auto_task = dfa_T1  # Assign task to be a DFA.
auto_reward = 0.1
env = TestEnv( auto_task=auto_task, auto_reward=auto_reward, render_mode="human")
sample_space = env.observation_space  # ???
obs, _ = env.reset()  # ???

# Start manual control.
manual = ManualControl(env, seed=42)
manual.start()

# Save actions string to a CSV file.
trace = pd.DataFrame(env.trace)
trace.to_csv("strings/traj_20.csv", index=False)
