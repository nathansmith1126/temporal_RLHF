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
# import utils_RLHF
# from utils_RLHF import get_obss_preprocessor, device
from utils_RLHF.other import device
from utils_RLHF.format import get_obss_preprocessor
from utils_RLHF.misc import load_bts_est, BT_SPEC_Estimator
import torch_ac
from torch_ac.algos.ppo import PPOAlgo
from torch_ac.utils.penv import ParallelEnv
from model import ACModel  # Should be your custom model
from AUTOMATA.auto_funcs import dfa_T1, create_wfa_T1
from datetime import datetime

# ----------------------------
# 🏁 Main Training Routine
# ----------------------------
def main(ENV_NAME, NUM_ENVS, TOTAL_FRAMES,
                          FRAMES_PER_PROC, WFA_indicator, 
                          gae_lambda = 0.95, entropy_coef = 0.05, 
                          max_steps=None):
    
    # ----------------------------
    # 🧠 Environment factory
    # ----------------------------

    if ENV_NAME in gym.envs.registry:
        print(f"{ENV_NAME} is already registered, no need to register")
    else:
        if WFA_indicator:
            "register WFA augmented env"
            # f = 0.99
            # s = 0.4
            # WFA_T1 = create_wfa_T1(f=f, s=s)
            est_name = "bts_est_2025-05-22_10-30-34.pkl"
            est_direc = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"
            full_path = os.path.join(est_direc, est_name)
            BTS_EST = load_bts_est(pickle_path=full_path)
            
            WFA = BTS_EST.learned_WFA
            # register environment
            register(
                id="MiniGrid-TemporalSPWFATestEnv-v0",               # Unique environment ID
                entry_point="Minigrid.minigrid.envs.test_envs:SPWFA_TestEnv",  # Module path to the class
                kwargs={
                    "WFA": WFA,
                    "max_steps": max_steps,
                    "render_mode": "rgb_array"
                },
            )
        else:
            # register dfa environment
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
    env      = env_list[0]
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
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
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

        if update % 40 == 0:
            obs, _ = env.reset()
            preprocessed_obs = preprocess_obss([obs], device=device)
            with torch.no_grad():
                dummy_memory = torch.zeros(1, model.memory_size, device=device)
                dist, _, _ = model(preprocessed_obs, dummy_memory)
                print("🔍 Action distribution:", dist.probs.cpu().numpy())

        

    print("✅ Training complete!")
    return algo, ENV_NAME



# ----------------------------
# 🔒 Windows-safe entry point
# ----------------------------
if __name__ == "__main__":
    WFA_indicator = True

    if WFA_indicator:
        ENV_NAME = "MiniGrid-TemporalSPWFATestEnv-v0"
    else:
        ENV_NAME = "MiniGrid-TemporalTestEnv-v0"
    # ENV_NAME = "MiniGrid-TemporalTestEnv-v0"
    # ENV_NAME = "MiniGrid-UnlockPickup-v0"
    # ENV_NAME = "MiniGrid-DoorKey-8x8-v0"

    # HYPERPAREMETERS
    max_steps = None 
    NUM_ENVS = 4
    TOTAL_FRAMES = 1_000_000
    FRAMES_PER_PROC = 512
    save = True
    entropy_coef = 0.15                                                                  
    gae_lambda   = 0.90
    algo, ENV_NAME = main(ENV_NAME=ENV_NAME, 
                          NUM_ENVS=NUM_ENVS, 
                          TOTAL_FRAMES=TOTAL_FRAMES, 
                          FRAMES_PER_PROC=FRAMES_PER_PROC, 
                          WFA_indicator=WFA_indicator, 
                          entropy_coef=entropy_coef, 
                          gae_lambda=gae_lambda,
                          max_steps=max_steps)

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


