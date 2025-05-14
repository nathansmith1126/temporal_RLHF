import sys
import os 

# Add parent and rl-starter-files to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# rl_starter_path = os.path.abspath(os.path.join(project_root, "rl-starter-files"))
# sys.path.insert(0, rl_starter_path)

import torch
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from datetime import datetime
from model import ACModel
from utils_RLHF import get_obss_preprocessor, device
from minigrid.wrappers import ImgObsWrapper
from AUTOMATA.auto_funcs import dfa_T1

# ----------------------------
# 🔧 Configuration
# ----------------------------

ENV_NAME = "MiniGrid-UnlockPickup-v0"
# ENV_NAME        = "MiniGrid-TemporalTestEnv-v0"
MODEL_TIMESTAMP = "2025-05-13_12-14-04"  # 📝 Change to your saved timestamp
MODEL_DIR       = rf"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\torch_models\{ENV_NAME}"
MODEL_PATH      = os.path.join(MODEL_DIR, f"model_{MODEL_TIMESTAMP}.pt")

NUM_EPISODES = 10
RENDER = True  # Set to False if running headless

if ENV_NAME in gym.envs.registry:
    print(f"{ENV_NAME} is already registered, no need to register")
else:
    # register environment
    register(
        id="MiniGrid-TemporalTestEnv-v0",               # Unique environment ID
        entry_point="Minigrid.minigrid.envs.test_envs:TestEnv",  # Module path to the class
        kwargs={
            "auto_task": dfa_T1,
            "auto_reward": 0.1,
            "render_mode": "rgb_array"
        },
    )

# ----------------------------
# 🧠 Load environment
# ----------------------------
def make_env():
    env = gym.make(ENV_NAME, render_mode="human" if RENDER else None)
    # env = ImgObsWrapper(env)
    return env

env = make_env()
# Preprocess observations
obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
action_space = env.action_space

# ----------------------------
# 🧠 Load model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acmodel = ACModel(obs_space=obs_space, action_space=action_space, 
                  use_memory=False, use_text=False).to(device)
acmodel.load_state_dict(torch.load(MODEL_PATH, map_location=device))
acmodel.eval()

print(f"✅ Loaded model from {MODEL_PATH}")

# ----------------------------
# 🎮 Run evaluation
# ----------------------------
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if RENDER:
            env.render()

        preprocessed_obs = preprocess_obss([obs], device=device)
        with torch.no_grad():
            dummy_memory = torch.zeros(1, acmodel.memory_size, device=device)
            dist, _, _ = acmodel(preprocessed_obs, dummy_memory)
        action = dist.probs.argmax(dim=1).item()
        print(f"action choice is {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"{reward}")
        done = terminated or truncated
        total_reward += reward

    print(f"🎯 Episode {ep+1} finished with reward: {total_reward}")

env.close()
print("✅ Evaluation complete.")

