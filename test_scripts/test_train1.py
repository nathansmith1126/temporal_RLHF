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
import torch_ac
from minigrid.wrappers import ImgObsWrapper
from Minigrid.minigrid.envs import test_envs
from gymnasium.envs.registration import register
# import utils_RLHF
# from utils_RLHF import get_obss_preprocessor, device
from utils_RLHF.other import device
from utils_RLHF.format import get_obss_preprocessor
from utils_RLHF.misc import register_special_envs, create_multiroom_env, create_ord_obj_env
from torch_ac.algos.ppo import PPOAlgo
from torch_ac.utils.penv import ParallelEnv
from model import ACModel  # custom model
from AUTOMATA.auto_funcs import dfa_T1
from datetime import datetime
from typing import Optional 

# ----------------------------
# 🏁 Main Training Routine
# ----------------------------
def PPO_train(ENV, NUM_ENVS, TOTAL_FRAMES,
                          FRAMES_PER_PROC, 
                          gae_lambda = 0.95, entropy_coef = 0.05, 
                          save_indicator: Optional[bool] = True, 
                          patience: Optional[int] = 30, 
                          max_steps: Optional[int] = None, 
                          ):
    """
    Runs training routine and saves model to torch_models
    Args:
    ENV - minigrdid environment agent navigates 
    NUM_ENVS [int] - number of parallel environments used during PPO setup
    gae_lambda [float] - learning parameter, decrease to promote exploration
    entropy_coef [float] - learning parameter, increase to promote exploration
    save_indicator [bool] - boolean indicator to save model
    patience [int]  - after patience updates with no improvement the training algorithm 
    max_steps [int] - maximum number of actions an agent takes before environment resets
    """
    # if env_params:
    #     f_reward = env_params["f_reward"]
    #     f_penalty = env_params["f_penalty"]
    #     finish_factor = env_params["finish_factor"]
    #     env_size = env_params["env_size"]
    #     register_special_envs( ENV_NAME=ENV_NAME, 
    #                       f_reward=f_reward, 
    #                       f_penalty=f_penalty, 
    #                       env_size=env_size, 
    #                       finish_factor=finish_factor,
    #                       max_steps=max_steps)
    # else:
    register_special_envs( ENV=ENV, 
                            max_steps=max_steps)
    # ----------------------------
    # 🧠 Environment factory
    # ----------------------------


    def make_env():
        env = gym.make(ENV.registered_name)
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
    algo = PPOAlgo(
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

    best_return = -np.inf # initialize best return

    while num_frames < TOTAL_FRAMES:
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        num_frames += logs["num_frames"]
        update += 1

        # Calculate the average return for the current update
        avg_return = np.mean(logs['return_per_episode'])
        
        print(
            f"Update {update} | Frames {num_frames} "
            f"| Return: {avg_return:.2f} "
            f"| Episodes: {len(logs['return_per_episode'])}"
        )

        # Check for improvement in average return
        if avg_return > best_return:
            best_return = avg_return
            no_improvement_count = 0
            # Save the model when the return improves
        else:
            no_improvement_count += 1
    
    # Early stopping condition: if average return doesn't improve for 'patience' updates
        if no_improvement_count >= patience:
            print(f"🚫 No improvement in the last {patience} updates. Stopping training...")
            break

        if update % 40 == 0:
            obs, _ = env.reset()
            preprocessed_obs = preprocess_obss([obs], device=device)
            with torch.no_grad():
                dummy_memory = torch.zeros(1, model.memory_size, device=device)
                dist, _, _ = model(preprocessed_obs, dummy_memory)
                print("🔍 Action distribution:", dist.probs.cpu().numpy())
    
    if save_indicator:
    # Create directory to save models
        save_dir = os.path.join(r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\torch_models", 
                                ENV.registered_name)
        os.makedirs(save_dir, exist_ok=True)

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_file = os.path.join(save_dir, f"model_{timestamp}.pt")

        # Save the model
        torch.save(algo.acmodel.state_dict(), model_file)
        print(f"✅ Model saved to {model_file}")

    print("✅ Training complete!")
    return algo, ENV.registered_name



# ----------------------------
# 🔒 Windows-safe entry point
# ----------------------------
if __name__ == "__main__":
    save_indicator = True 
    # env_indicator = "ord_obj"
    env_indicator = "multi_room"
    if env_indicator == "ord_obj":
        ord_obj_env = create_ord_obj_env(f_penalty=1.0)
        ENV = ord_obj_env
    elif env_indicator == "multi_room":
        multi_room_env = create_multiroom_env()
        ENV = multi_room_env
    else:
        # dfa
        raise ValueError("Current invalid environment")
    
    # ENV_NAME = "MiniGrid-TemporalTestEnv-v0"
    # ENV_NAME = "MiniGrid-UnlockPickup-v0"
    # ENV_NAME = "MiniGrid-DoorKey-8x8-v0"

    # HYPERPAREMETERS
    max_steps = None 
    NUM_ENVS = 8
    TOTAL_FRAMES = 1_500_000
    FRAMES_PER_PROC = 512
    save = True
    entropy_coef = 0.15                                                                  
    gae_lambda   = 0.90
    patience = 40
    algo, _ = PPO_train(ENV=ENV,NUM_ENVS=NUM_ENVS, 
                            TOTAL_FRAMES=TOTAL_FRAMES, 
                            FRAMES_PER_PROC=FRAMES_PER_PROC,  
                            entropy_coef=entropy_coef, 
                            gae_lambda=gae_lambda,
                            max_steps=max_steps, 
                            save_indicator = save_indicator, 
                            patience = patience)


