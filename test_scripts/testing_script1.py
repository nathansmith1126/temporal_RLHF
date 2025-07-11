from __future__ import annotations
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Minigrid.minigrid.envs.test_envs import TestEnv, WFA_TestEnv, SPWFA_TestEnv, ordered_obj
from Minigrid.minigrid.manual_control import ManualControl
from AUTOMATA.auto_funcs import *
from utils_RLHF.misc import create_multiroom_env, create_ord_obj_env
# from utils_RLHF.misc import load_bts_est, BT_SPEC_Estimator, word2WFA_max

env_indicator = "ord_obj"
# env_indicator = "multi_room"
if env_indicator == "ord_obj":
    ord_obj_env = create_ord_obj_env(f_penalty=1.0, render_mode="human")
    ENV = ord_obj_env
elif env_indicator == "multi_room":
    multi_room_env = create_multiroom_env(f_reward = 10.0, f_penalty = 1.0, render_mode="human")
    ENV = multi_room_env
else:
    raise ValueError("Current invalid environment")

# used for debuging
sample_space = ENV.observation_space
obs, _ = ENV.reset()

# Start manual control interface
manual = ManualControl(ENV, seed=42)
manual.start()

   # auto_task = dfa_T1

    # f = 0.90
    # s = 0.02
    # # WFA_T1    = create_wfa_T1(f=f, s=s)
    # WFA = create_wfa_T1a(f,s,u=0.2)
    # auto_reward = 0.1 
    # max_steps   = 1_000

    # # extract bradley terry estimator
    # est_name = "bts_est_2025-05-22_10-30-34.pkl"
    # est_direc = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"
    # full_path = os.path.join(est_direc, est_name)
    # BTS_EST = load_bts_est(pickle_path=full_path)

    # learned_WFA = BTS_EST.learned_WFA
    # # Instantiate automata augmented env
    # # env = TestEnv( auto_task=auto_task, auto_reward=auto_reward, render_mode="human")

    # # Instantiate WFA augmented env
    # env   = SPWFA_TestEnv(WFA=learned_WFA, render_mode="human", max_steps=max_steps)