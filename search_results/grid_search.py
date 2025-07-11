from test_scripts.testing_train1 import PPO_train
from utils_RLHF.misc import create_ord_obj_env
import torch
import gymnasium

if __name__ == "__main__":
    # PPO hyperparameters.
    max_steps = None
    NUM_ENVS = 8
    TOTAL_FRAMES = 10
    FRAMES_PER_PROC = 512
    save = True
    entropy_coef = 0.15
    gae_lambda = 0.90
    patience = 40

    # DO NOT CHANGE THE SIZE OF THESE LISTS OR IT WILL MESS UP THE FILE NAMES!!!!!!!!!!!!!!!!!
    f = [0, 1e-3, 0.1, 0.8, 1, 1.2, 1.6, 5, 10]
    s = [0, 1e-3, 0.1, 0.8, 1, 1.2, 1.6, 5, 10]
    u = [0, 1e-3, 0.1, 0.8, 1, 1.2, 1.6, 5, 10]
    f_reward = [0, 0.1, 1, 5, 10, 15, 1e2, 1e3]
    f_penalty = [0, 1e-3, 1e-2, 0.1, 0.25, 1, 10]
    finish_factor = [0, 0.1, 1, 5, 10, 15, 1e2, 1e3]

    # Iterate over every parameter in ever list.
    for i1 in range(len(f)):
        for i2 in range(len(s)):
            for i3 in range(len(u)):
                for i4 in range(len(f_reward)):
                    for i5 in range(len(f_penalty)):
                        for i6 in range(len(finish_factor)):
                            # Create environment.
                            save_indicator = False
                            ord_obj_env = create_ord_obj_env(
                                f=f[i1],
                                s=s[i2],
                                u=u[i3],
                                f_reward=f_reward[i4],
                                f_penalty=f_penalty[i5],
                                finish_factor=finish_factor[i6],
                            )
                            ENV = ord_obj_env
                            if "MiniGrid-Temporal-ord_obj-v0" in gymnasium.registry:
                                del gymnasium.registry["MiniGrid-Temporal-ord_obj-v0"]

                            # Save files as a sequence of 6 numbers denoting what value each of the hyperparameters
                            # was at for this run of the model.
                            logs_name = str(i1) + str(i2) + str(i3) + str(i4) + str(i5) + str(i6)
                            algo_name = str(i1) + str(i2) + str(i3) + str(i4) + str(i5) + str(i6)

                            # Run the model. If it fails just save a string with that info instead.
                            try:
                                algo, logs, _ = PPO_train(
                                    ENV=ENV,
                                    NUM_ENVS=NUM_ENVS,
                                    TOTAL_FRAMES=TOTAL_FRAMES,
                                    FRAMES_PER_PROC=FRAMES_PER_PROC,
                                    entropy_coef=entropy_coef,
                                    gae_lambda=gae_lambda,
                                    max_steps=max_steps,
                                    save_indicator=save_indicator,
                                    patience=patience
                                )

                                # Save results.
                                torch.save(logs, f"logs/{logs_name}.pt")
                                torch.save(algo.acmodel.state_dict(), f"algos/{algo_name}.pt")
                            except:
                                torch.save("ERROR", f"logs/{logs_name}.pt")
                                torch.save("ERROR", f"algos/{algo_name}.pt")
