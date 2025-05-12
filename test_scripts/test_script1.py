from __future__ import annotations
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Minigrid.minigrid.envs.test_envs import TestEnv
from Minigrid.minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions
from AUTOMATA.auto_funcs import DFAMonitor, dfa_T1
auto_task = dfa_T1
auto_reward = 0.1 
# Instantiate your custom DoorKeyEnv
env = TestEnv( auto_task=auto_task, auto_reward=auto_reward, render_mode="human")

sample_space = env.observation_space
# for action in Actions:
#     print(f"{action.value}: {action.name}")
obs, _ = env.reset()
# Start manual control interface
manual = ManualControl(env, seed=42)
manual.start()