# register_env.py

import gymnasium as gym
from gymnasium.envs.registration import register
from Minigrid.minigrid.envs.test_envs import TestEnv
from AUTOMATA.auto_funcs import dfa_T1
import register_env 

# Define a wrapper function for environment creation
def make_custom_env():
    return TestEnv(auto_task=dfa_T1, auto_reward=0.1, render_mode="rgb_array")

# Register the environment

register(
    id="MiniGrid-TemporalTestEnv-v0",               # Unique environment ID
    entry_point="Minigrid.minigrid.envs.test_envs:TestEnv",  # Module path to the class
    kwargs={
        "auto_task": dfa_T1,
        "auto_reward": 0.1,
        "render_mode": "rgb_array"
    },
)

print("✅ Custom environment 'MiniGrid-TemporalTestEnv-v0' registered successfully.")
