from __future__ import annotations
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Minigrid.minigrid.envs.test_envs import TestEnv, WFA_TestEnv, SPWFA_TestEnv
from Minigrid.minigrid.manual_control import ManualControl
from minigrid.core.actions import Actions
from AUTOMATA.auto_funcs import *
from utils_RLHF.misc import load_bts_est, BT_SPEC_Estimator
auto_task = dfa_T1

f = 0.90
s = 0.02
# WFA_T1    = create_wfa_T1(f=f, s=s)
WFA = create_wfa_T1a(f,s,u=0.2)
auto_reward = 0.1 
max_steps   = 1_000

# extract estimator
est_name = "bts_est_2025-05-22_10-30-34.pkl"
est_direc = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"
full_path = os.path.join(est_direc, est_name)
BTS_EST = load_bts_est(pickle_path=full_path)

learned_WFA = BTS_EST.learned_WFA
# Instantiate automata augmented env
# env = TestEnv( auto_task=auto_task, auto_reward=auto_reward, render_mode="human")

# Instantiate WFA augmented env
env   = SPWFA_TestEnv(WFA=learned_WFA, render_mode="human", max_steps=max_steps)

sample_space = env.observation_space
# for action in Actions:
#     print(f"{action.value}: {action.name}")
obs, _ = env.reset()
# Start manual control interface
manual = ManualControl(env, seed=42)
manual.start()