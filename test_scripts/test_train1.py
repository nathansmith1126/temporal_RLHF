# 🚀 torch-ac minimal training example
import os
import sys

# Add parent and rl-starter-files to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# rl_starter_path = os.path.abspath(os.path.join(project_root, "rl-starter-files"))
# sys.path.insert(0, rl_starter_path)

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from Minigrid.minigrid.envs import test_envs
from gymnasium.envs.registration import register
import utils_RLHF
from utils_RLHF import get_obss_preprocessor, device
import torch_ac
from torch_ac.algos.ppo import PPOAlgo
from torch_ac.utils.penv import ParallelEnv
from model import ACModel  # Should be your custom model
from AUTOMATA.auto_funcs import dfa_T1
from datetime import datetime

# ----------------------------
# 🏁 Main Training Routine
# ----------------------------
def main(ENV_NAME, NUM_ENVS, TOTAL_FRAMES, 
                          FRAMES_PER_PROC):
    
    # ----------------------------
    # 🧠 Environment factory
    # ----------------------------

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

    def make_env():
        env = gym.make(ENV_NAME)
        # env.observation_space
        # env = ImgObsWrapper(env)
        # env.observation_space
        return env
    
    # Create environments
    env_list = [make_env() for _ in range(NUM_ENVS)]
    envs = ParallelEnv(env_list)
    # sample_space = make_env().observation_space

    # Preprocess observations
    obs_space, preprocess_obss = get_obss_preprocessor(envs.observation_space)

    # Create model
    model = ACModel(
        obs_space=obs_space,
        action_space=envs.action_space,
        use_memory=False,
        use_text=False
    ).to(device)

    # print(model)
    # Create PPO algorithm
    algo = torch_ac.PPOAlgo(
        envs=env_list,
        acmodel=model,
        device=device,
        num_frames_per_proc=FRAMES_PER_PROC,
        discount=0.99,
        lr=0.00025,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=1,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=preprocess_obss
    )

    # Training loop
    num_frames = 0
    update = 0

    while num_frames < TOTAL_FRAMES:
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        num_frames += logs["num_frames"]
        update += 1

        print(
            f"Update {update} | Frames {num_frames} "
            f"| Return: {np.mean(logs['return_per_episode']):.2f} "
            f"| Episodes: {len(logs['return_per_episode'])}"
        )

    print("✅ Training complete!")
    return algo, ENV_NAME



# ----------------------------
# 🔒 Windows-safe entry point
# ----------------------------
if __name__ == "__main__":
    # ENV_NAME = "MiniGrid-TemporalTestEnv-v0"
    ENV_NAME = "MiniGrid-UnlockPickup-v0"
    # ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
    NUM_ENVS = 4
    TOTAL_FRAMES = 1_000_000
    FRAMES_PER_PROC = 128
    save = True
    algo, ENV_NAME = main(ENV_NAME=ENV_NAME, NUM_ENVS=NUM_ENVS, 
                          TOTAL_FRAMES=TOTAL_FRAMES, 
                          FRAMES_PER_PROC=FRAMES_PER_PROC)

    if save:
        # Create directory to save models
        save_dir = os.path.join(r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\torch_models", 
                                ENV_NAME)
        os.makedirs(save_dir, exist_ok=True)

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_file = os.path.join(save_dir, f"model_{timestamp}.pt")

        # Save the model
        torch.save(algo.acmodel.state_dict(), model_file)
        print(f"✅ Model saved to {model_file}")


