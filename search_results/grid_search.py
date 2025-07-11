from test_scripts.testing_train1 import PPO_train
from utils_RLHF.misc import create_ord_obj_env
import torch
import gymnasium

def save(env, logs_name, algo_name):
    # PPO hyperparameters.
    max_steps = None
    NUM_ENVS = 4
    TOTAL_FRAMES = 1_000_000
    FRAMES_PER_PROC = 512
    save = True
    entropy_coef = 0.15
    gae_lambda = 0.90
    patience = 40

    try:
        # Run PPO.
        algo, logs, _ = PPO_train(
            ENV=env,
            NUM_ENVS=NUM_ENVS,
            TOTAL_FRAMES=TOTAL_FRAMES,
            FRAMES_PER_PROC=FRAMES_PER_PROC,
            entropy_coef=entropy_coef,
            gae_lambda=gae_lambda,
            max_steps=max_steps,
            save_indicator=False,
            patience=patience
        )

        # Save results.
        torch.save(logs, f"logs/{logs_name}.pt")
        torch.save(algo.acmodel.state_dict(), f"algos/{algo_name}.pt")
    except:
        torch.save("ERROR", f"logs/{logs_name}.pt")
        torch.save("ERROR", f"algos/{algo_name}.pt")


if __name__ == "__main__":
    # Parameters to try.
    # f_reward always 2 to 5 times bigger than f_penalty.
    f_base = 1.2
    s_base = 0.8
    u_base = 0.75
    f_reward_base = 10
    f_penalty_base = 2.5
    finish_factor_base = 10

    f = [1e-3, 0.1, 1, 1.2, 5, 10]
    s = [1e-3, 0.1, 0.8, 1, 5, 10]
    u = [1e-3, 0.1, 0.75, 1, 5, 10]
    f_reward = [0.1, 1, 10, 100, 1e3]
    f_penalty = [0.25, 1, 2.5, 10, 25]
    finish_factor = [0.1, 1, 10, 100, 1e3]

    for i in range(6):
        if i == 0:
            for j in range(len(f)):
                ord_obj_env = create_ord_obj_env(
                    f=f[j],
                    s=s_base,
                    u=u_base,
                    f_reward=f_reward_base,
                    f_penalty=f_penalty_base,
                    finish_factor=finish_factor_base,
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"f{f[j]}.pt"
                algo_name = f"f{f[j]}.pt"
                save(ENV, logs_name, algo_name)
        if i == 1:
            for j in range(len(s)):
                ord_obj_env = create_ord_obj_env(
                    f=f_base,
                    s=s[j],
                    u=u_base,
                    f_reward=f_reward_base,
                    f_penalty=f_penalty_base,
                    finish_factor=finish_factor_base,
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"s{s[j]}.pt"
                algo_name = f"s{s[j]}.pt"
                save(ENV, logs_name, algo_name)
        if i == 2:
            for j in range(len(u)):
                ord_obj_env = create_ord_obj_env(
                    f=f_base,
                    s=s_base,
                    u=u[j],
                    f_reward=f_reward_base,
                    f_penalty=f_penalty_base,
                    finish_factor=finish_factor_base,
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"u{u[j]}.pt"
                algo_name = f"u{u[j]}.pt"
                save(ENV, logs_name, algo_name)
        if i == 3:
            for j in range(len(f_reward)):
                ord_obj_env = create_ord_obj_env(
                    f=f_base,
                    s=s_base,
                    u=u_base,
                    f_reward=f_reward[j],
                    f_penalty=f_penalty_base,
                    finish_factor=finish_factor_base,
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"f_reward{f_reward[j]}.pt"
                algo_name = f"f_reward{f_reward[j]}.pt"
                save(ENV, logs_name, algo_name)
        if i == 4:
            for j in range(len(f_penalty)):
                ord_obj_env = create_ord_obj_env(
                    f=f_base,
                    s=s_base,
                    u=u_base,
                    f_reward=f_reward_base,
                    f_penalty=f_penalty[j],
                    finish_factor=finish_factor_base,
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"f_penalty{f_penalty[j]}.pt"
                algo_name = f"f_penalty{f_penalty[j]}.pt"
                save(ENV, logs_name, algo_name)
        if i == 5:
            for j in range(len(finish_factor)):
                ord_obj_env = create_ord_obj_env(
                    f=f_base,
                    s=s_base,
                    u=u_base,
                    f_reward=f_reward_base,
                    f_penalty=f_penalty_base,
                    finish_factor=finish_factor[j],
                )
                ENV = ord_obj_env
                if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                    del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                logs_name = f"finish_factor{finish_factor[j]}.pt"
                algo_name = f"finish_factor{finish_factor[j]}.pt"
                save(ENV, logs_name, algo_name)
