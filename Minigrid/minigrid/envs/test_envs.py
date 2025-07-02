from __future__ import annotations
import os 
import json 
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv # subclass of gym.Env
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid # subclass of MiniGridEnv
from minigrid.manual_control import ManualControl
from AUTOMATA.auto_funcs import dfa_T1, DFAMonitor, WFA_monitor
from automata.fa.dfa import DFA
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from typing import Optional
# import builtins

# # Disable print
# builtins.print = lambda *args, **kwargs: None


class TestEnv(RoomGrid):
    """
    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} box"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | drop an object            |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-UnlockPickup-v0`

    """

    def __init__(self, auto_task, auto_reward, max_steps: int | None = None, **kwargs):
        room_size = 6
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )
        self.path_log = []
        self.auto_task = auto_task # automaton representation of LTL task

        # initialize monitor for dfa transitions
        self.dfa_monitor = DFAMonitor( self.auto_task )

        # registration name 
        self.registered_name = "MiniGrid-TemporalTestEnv-v0"

        # set dfa to initial state
        self.dfa_monitor.reset()
        "reward gained for achieving one sub task "
        "(making forward transition in auto_task)"
        self.auto_reward = auto_reward
        self.auto_num_states = self.dfa_monitor.num_states

        if max_steps is None:
            max_steps = 8 * room_size**2
        
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )
        self.update_obs_space()
    
    def update_obs_space(self):
        """Update observation space to include automaton state"""
        # Grab the original image observation space
        # image_space = self.observation_space["image"]

        # # Define the new Dict space
        # self.observation_space = spaces.Dict({
        #     "image": image_space,
        #     "auto_state": spaces.Box(
        #         low=0.0,
        #         high=1.0,
        #         shape=(self.auto_num_states,),
        #         dtype=np.float32
        #     )
        # })
        self.observation_space["auto_state"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.auto_num_states,),
                dtype=np.float32
            )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def gen_obs(self):
        obs = super().gen_obs()
        initial_state_string = self.auto_task.initial_state
        initial_state_array  = self.dfa_monitor.state_label2array(initial_state_string)
        obs["auto_state"]    = initial_state_array
        return obs 
        
    def save_path_log(self):
        # print(f"entered save_path_log")
        save_dir = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\test_results"  # where to store
        os.makedirs(save_dir, exist_ok=True)

        filename = f"path_log_episode_{self.step_count}.json"
        full_path = os.path.join(save_dir, filename)

        with open(full_path, "w") as f:
            json.dump(self.path_log, f, indent=2)

        # print(f"✅ Path log saved to {full_path}")

    def step(self, action):
        # check if agent is carrying anything in previous time step before
        # action is propagated forward
        was_carrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        # Start with a basic log entry
        log_entry = {
            "step": int(self.step_count),
            "pos": tuple(int(x) for x in self.agent_pos),
            "dir": int(self.agent_dir),
            "action": int(action),
            "reward": float(reward),
            "event": None  # We'll populate this if an important event happens
        }
        
        # Look ahead at what the agent is facing
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # --- Detect important events ---

        # 1. Case when object is picked up
        if action == self.actions.pickup and self.carrying:
            # print("logged pickup")
            log_entry["event"] = f"pickup {self.carrying.type}"
            self.dfa_monitor.step(f"pickup {self.carrying.type}")
            "1.a case when red box is picked up. "
            "this is the final task"
            if self.carrying is self.obj:
                "add efficiency reward inheritted from minigridenv"
                reward += self._reward() 

                "termination variable updated to true"
                terminated = True

        # 2. Case when object is toggled
        elif action == self.actions.toggle and fwd_cell:
            # print("logged toggle")
            log_entry["event"] = f"toggle {fwd_cell.type}"

            # 2.a Specifically detect door open/close
            if hasattr(fwd_cell, "is_open"):
                if fwd_cell.is_open:
                    log_entry["event"] = "opened door"
                    self.dfa_monitor.step("opened door")
                else:
                    log_entry["event"] = "closed door"
                    self.dfa_monitor.step("closed door")
            else:
                # step dfa_monitor to reset progress attribute
                self.dfa_monitor.step()

        # 3. Object was dropped 
        elif action == self.actions.drop and was_carrying and self.carrying is None:
            log_entry["event"] = f"dropped {was_carrying.type}"
            self.dfa_monitor.step(f"dropped {was_carrying.type}")

        # 4. Movement (left, right, forward)
        elif action in [self.actions.forward, self.actions.left, self.actions.right]:
            self.dfa_monitor.step("movement")
            # no event saved since this will not create a transition
        else:
            self.dfa_monitor.step()

        current_auto_state_array = self.dfa_monitor.state_label2array()
        # add to observation dictionary-
        obs["auto_state"] = current_auto_state_array
        # print(f"current auto state index is {current_auto_state_array}")
        "add reward in case progress is made from the task automaton"
        if self.dfa_monitor.progress:
            reward += self.auto_reward
            # print("extra_reward")

        "store reward into path_log dictionary for specific timestep"
        log_entry["reward"] = reward

        # --- Append the log with new entry ---
        self.path_log.append(log_entry)

        "termination/truncation commands if carrying final object"
        if truncated:
            self.path_log = []
            self.dfa_monitor.reset()

        if terminated:
            # self.save_path_log()
            self.dfa_monitor.reset()
            # print(f"carrying final {self.obj}")

        return obs, reward, terminated, truncated, info

class WFA_TestEnv(RoomGrid):
    """
    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} box"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-UnlockPickup-v0`

    """

    def __init__(self, 
                 WFA_monitor, 
                 f_penalty = 0.05,
                 f_reward  = 0.75,
                 finish_factor = 3,
                 max_steps: int | None = None, 
                 **kwargs):
        room_size = 6
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )
        self.path_log = []

        # initialize monitor for wfa transitions
        self.WFA_monitor = WFA_monitor

        # set dfa to initial state
        # self.dfa_monitor.reset()
        "reward gained for achieving one sub task "
        "(making forward transition in auto_task)"

        self.registered_name = "MiniGrid-TemporalWFATestEnv-v0"

        self.WFA_num_states = self.WFA_monitor.num_states
        self.f_penalty = f_penalty
        self.f_reward  = f_reward
        self.finish_factor = finish_factor
        self.goal_ind      = False
        if max_steps is None:
            max_steps = 8 * room_size**2
            print(f"{max_steps}")
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )
        self.update_obs_space()
    
    def update_obs_space(self):
        """Update observation space to include automaton state"""
        # Grab the original image observation space
        # image_space = self.observation_space["image"]

        # # Define the new Dict space
        # self.observation_space = spaces.Dict({
        #     "image": image_space,
        #     "auto_state": spaces.Box(
        #         low=0.0,
        #         high=1.0,
        #         shape=(self.auto_num_states,),
        #         dtype=np.float32
        #     )
        # })
        self.observation_space["auto_state"] = spaces.Box(
                low=-1.0,
                high=2.0,
                shape=(self.WFA_num_states, ),
                dtype=np.float32
            )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def gen_obs(self):
        obs = super().gen_obs()
        initial_state_array  = self.WFA_monitor.initial.reshape(self.WFA_num_states, )
        obs["auto_state"]    = initial_state_array
        return obs 
        
    def save_path_log(self):
        # print(f"entered save_path_log")
        save_dir = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\test_results"  # where to store
        os.makedirs(save_dir, exist_ok=True)

        filename = f"WFA_TestEnv_log_episode_{self.step_count}.json"
        full_path = os.path.join(save_dir, filename)

        with open(full_path, "w") as f:
            json.dump(self.path_log, f, indent=2)

        # print(f"✅ Path log saved to {full_path}")

    def step(self, action):
        # progress indicator and completion ind is set to false prior to each step
        self.progress = False
        self.goal_ind = False 

        # check if agent is carrying anything in previous time step before
        # action is propagated forward
        was_carrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        # Start with a basic log entry
        log_entry = {
            "step": int(self.step_count),
            "pos": tuple(int(x) for x in self.agent_pos),
            "dir": int(self.agent_dir),
            "action": int(action),
            "reward": float(reward),
            "event": None  # We'll populate this if an important event happens
        }
        
        # Look ahead at what the agent is facing
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # --- Detect important events ---

        # 1. Case when action is pickup
        if action == self.actions.pickup:
            """
            1.a Case when agent actually picked something up
            ie "carrying and pickup"
            """
            if self.carrying:
                # print("logged pickup")
                # carrying key or box
                log_entry["event"] = f"pickup {self.carrying.type}"
                self.WFA_monitor.step(f"pickup {self.carrying.type}")
                self.progress = True
            else:
                # carrying nothing
                self.WFA_monitor.step("useless")
        
        # 2. Case when object is toggled
        elif action == self.actions.toggle:
            # check if we toggled a door
            if hasattr(fwd_cell, "is_open"):
                # good now determine if we closed it or not
                # 2.a Specifically detect door open/close
                if fwd_cell.is_open:
                    # we opened door
                    log_entry["event"] = "opened door"
                    self.WFA_monitor.step("opened door")
                    self.progress = True
                else:
                    # we closed door
                    log_entry["event"] = "closed door"
                    self.WFA_monitor.step("closed door")
                    self.progress = True
            else:
                """
                we did not toggle a door 
                therefore the action was useless 
                and pass "useless symbol into WFA"
                """
                self.WFA_monitor.step("useless")

        # 3. action is drop 
        elif action == self.actions.drop:
            # log progress because the agent ACTUALLY dropped an item
            if was_carrying and self.carrying is None:
                # used to carry and no longer carry
                log_entry["event"] = f"dropped {was_carrying.type}"
                self.WFA_monitor.step(f"dropped {was_carrying.type}")
                self.progress = True
                "3.a case when red box is picked up. "
                "this is the final task"
                # if was_carrying is self.obj:
                    # dropped box
                if self.WFA_monitor.complete_ind == True:
                    "termination variable updated to true"
                    terminated = True

                    "completed goal indicator"
                    self.goal_ind = True
            else:
                # dropping without carrying is a useless action
                self.WFA_monitor.step("useless")
        # 4. Movement (left, right, forward)
        # elif action in [self.actions.forward, self.actions.left, self.actions.right]:
        #     self.WFA_monitor.step("movement")
        #     # no event saved since this will not create a transition
        else: # movement occurs or irrelevant action
            self.WFA_monitor.step()
            # print("no_event")
        current_auto_state_array = self.WFA_monitor.current_state
        # add to observation dictionary-
        obs["auto_state"] = current_auto_state_array.reshape(self.WFA_num_states, )
        # print(f"current auto state index is {current_auto_state_array}")
        
        """
        REWARD PROGRESS AND HIGH LIKELIHOOD
        PUNISH PROGRESS AND LOW LIKELIHOOD
        TERMINATE IF LIKELIHOOD IS TOO LOW
        """   
        # consolidate rewards and penalties  
        if self.WFA_monitor.f_decrease:
            # penalize if WFA func decreases
            reward -= self.f_penalty 
            # print("bad decision")
            if self.WFA_monitor.unstable_ind:
                # penalize if WFA func is too close to zero
                terminated = True
                print("UNSTABLE PATH")
        else:
            # good decision
            if self.WFA_monitor.f_increase:
                # progress was made if score increased
                reward += self.f_reward
                if self.WFA_monitor.complete_ind:
                    "add efficiency reward inheritted from minigridenv"
                    reward += self.finish_factor*self._reward() 
                # print("good decision")

        "store reward into path_log dictionary for specific timestep"
        log_entry["reward"] = reward

        # --- Append the log with new entry ---
        self.path_log.append(log_entry)

        "termination/truncation commands if carrying final object"
        if truncated:
            self.path_log = []
            self.WFA_monitor.reset()
            print("TRUNCATED")
        if terminated:
            self.WFA_monitor.reset()
            if self.goal_ind:
                # self.save_path_log()
                pass 
            else:
                self.path_log=[]

        return obs, reward, terminated, truncated, info

class SPWFA_TestEnv(RoomGrid):
    """
    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} box"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-UnlockPickup-v0`

    """

    def __init__(self, 
                 WFA, 
                 f_penalty = 0.01,
                 f_reward  = 0.75,
                 finish_factor = 2,
                 auto_reward = None, 
                 max_steps: int | None = None, 
                 **kwargs):
        room_size = 6
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )
        self.path_log = []
        self.WFA = WFA # WFA representation of LTL task

        # initialize monitor for wfa transitions
        self.WFA_monitor = WFA_monitor(WFA=WFA)

        # registered name with gym
        self.registered_name = "MiniGrid-TemporalSPWFATestEnv-v0"

        # set dfa to initial state
        # self.dfa_monitor.reset()
        "reward gained for achieving one sub task "
        "(making forward transition in auto_task)"
        self.auto_reward = auto_reward
        self.WFA_num_states = self.WFA.n
        self.f_penalty = f_penalty
        self.f_reward  = f_reward
        self.finish_factor = finish_factor
        self.goal_ind      = False
        if max_steps is None:
            max_steps = 8 * room_size**2
            print(f"{max_steps}")
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )
        self.update_obs_space()
    
    def update_obs_space(self):
        """Update observation space to include automaton state"""
        # Grab the original image observation space
        # image_space = self.observation_space["image"]

        # # Define the new Dict space
        # self.observation_space = spaces.Dict({
        #     "image": image_space,
        #     "auto_state": spaces.Box(
        #         low=0.0,
        #         high=1.0,
        #         shape=(self.auto_num_states,),
        #         dtype=np.float32
        #     )
        # })
        self.observation_space["auto_state"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.WFA_num_states, ),
                dtype=np.float64
            )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def gen_obs(self):
        obs = super().gen_obs()
        initial_state_array  = self.WFA_monitor.initial.reshape(self.WFA_num_states, )
        obs["auto_state"]    = initial_state_array
        return obs 
        
    def save_path_log(self):
        # print(f"entered save_path_log")
        save_dir = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\test_results"  # where to store
        os.makedirs(save_dir, exist_ok=True)

        filename = f"WFA_TestEnv_log_episode_{self.step_count}.json"
        full_path = os.path.join(save_dir, filename)

        with open(full_path, "w") as f:
            json.dump(self.path_log, f, indent=2)

        # print(f"✅ Path log saved to {full_path}")

    def step(self, action):
        # progress indicator and completion ind is set to false prior to each step
        self.progress = False
        self.goal_ind = False 

        # check if agent is carrying anything in previous time step before
        # action is propagated forward
        was_carrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        # Start with a basic log entry
        log_entry = {
            "step": int(self.step_count),
            "pos": tuple(int(x) for x in self.agent_pos),
            "dir": int(self.agent_dir),
            "action": int(action),
            "reward": float(reward),
            "event": None  # We'll populate this if an important event happens
        }
        
        # Look ahead at what the agent is facing
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # --- Detect important events ---

        # 1. Case when action is pickup
        if action == self.actions.pickup:
            """
            1.a Case when agent actually picked something up
            ie "carrying and pickup"
            """
            if self.carrying:
                # print("logged pickup")
                # carrying key or box
                log_entry["event"] = f"pickup {self.carrying.type}"
                self.WFA_monitor.step(f"pickup {self.carrying.type}")
                self.progress = True
                "1.a case when red box is picked up. "
                "this is the final task"
                if self.carrying is self.obj:
                    # carrying bod
                    "add efficiency reward inheritted from minigridenv"
                    reward += self.finish_factor*self._reward() 

                    "termination variable updated to true"
                    terminated = True

                    "completed goal indicator"
                    self.goal_ind   = True
            else:
                # carrying nothing
                self.WFA_monitor.step("useless")
                # self.WFA_monitor.step()
        # 2. Case when object is toggled
        elif action == self.actions.toggle:
            # check if we toggled a door
            if hasattr(fwd_cell, "is_open"):
                # good now determine if we closed it or not
                # 2.a Specifically detect door open/close
                if fwd_cell.is_open:
                    # we opened door
                    log_entry["event"] = "opened door"
                    self.WFA_monitor.step("opened door")
                    self.progress = True
                else:
                    # we closed door
                    log_entry["event"] = "closed door"
                    self.WFA_monitor.step("closed door")
                    self.progress = True
            else:
                """
                we did not toggle a door 
                therefore the action was useless 
                and pass "useless symbol into WFA"
                """
                self.WFA_monitor.step("useless")
                # self.WFA_monitor.step()

        # 3. action is drop 
        elif action == self.actions.drop:
            # log progress because the agent ACTUALLY dropped an item
            if was_carrying and self.carrying is None:
                # used to carry and no longer carry
                log_entry["event"] = f"dropped {was_carrying.type}"
                self.WFA_monitor.step(f"dropped {was_carrying.type}")
                self.progress = True
            else:
                # dropping without carrying is a useless action
                self.WFA_monitor.step("useless")
                # self.WFA_monitor.step()
        # 4. Movement (left, right, forward)
        # elif action in [self.actions.forward, self.actions.left, self.actions.right]:
        #     self.WFA_monitor.step("movement")
        #     # no event saved since this will not create a transition
        else: # movement occurs or irrelevant action
            self.WFA_monitor.step()
            # print("no_event")
        current_auto_state_array = self.WFA_monitor.current_state
        # add to observation dictionary-
        obs["auto_state"] = current_auto_state_array.reshape(self.WFA_num_states, )
        # print(f"current auto state index is {current_auto_state_array}")
        
        """
        Must have mechanism to 
        1. reward forward progress through automaton
        2. penalize useless actions
        3. penalize unnecessary movement eg circle
        4. Allow for movement through automaton after imperfect decisions are made

        REWARD PROGRESS AND HIGH LIKELIHOOD
        PUNISH PROGRESS AND LOW LIKELIHOOD
        TERMINATE IF LIKELIHOOD IS TOO LOW
        """
        if self.WFA_monitor.recent_event == "useless":
            reward -= self.f_penalty
        
        elif self.WFA_monitor.prob_increase:
            # good decision
            if self.progress:
                # progress was made and it was a good decision
                # if probability increases significantly 
                # then it was a good decision
                reward += self.f_reward
                print("good decision")

        # if self.WFA_monitor.unstable_ind:
        #         # probability is too close to zero
        #         terminated = True
        #         print(f"UNSTABLE PATH step count: {self.step_count}")
        # elif self.WFA_monitor.prob_increase:
        #     # good decision
        #     if self.progress:
        #         # progress was made and it was a good decision
        #         # if probability increases significantly 
        #         # then it was a good decision
        #         reward += self.f_reward
        #         # print("good decision")
        # else:
        #     pass 

        "store reward into path_log dictionary for specific timestep"
        log_entry["reward"] = reward

        # --- Append the log with new entry ---
        self.path_log.append(log_entry)

        "termination/truncation commands if carrying final object"
        if truncated:
            self.path_log = []
            self.WFA_monitor.reset()
            print("TRUNCATED")
        if terminated:
            self.WFA_monitor.reset()
            if self.goal_ind:
                # self.save_path_log()
                pass 
            else:
                self.path_log=[]

        return obs, reward, terminated, truncated, info

class ordered_obj(MiniGridEnv):
    """
    ## Description

    The agent must pick up and drop boxes in a specific order. 

    ## Mission Space

    "pick up object 1 - drop object 1 - pick up object 2 - drop object 2"

    {color} is the color of the object. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".
    {obj_type} is the type of the object. Can be "key", "ball", "box".

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | pickup object        |
    | 4   | drop         | drop object          |
    | 5   | toggle       | toggle project       |
    | 6   | done         | Done completing task |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent interacts with objects in proper order and hits done
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-GoToObject-6x6-N2-v0`
    - `MiniGrid-GoToObject-8x8-N2-v0`

    """

    def __init__(self, wfa_monitor: WFA_monitor, 
                 actions_list: list[str],
                 objects_list: list[str],  
                 size=6,
                 f_reward: Optional[float] = 10.0, 
                 f_penalty: Optional[float] = 0.25,
                 finish_factor: Optional[float] = 10.0,
                 max_steps: int | None = None, **kwargs):
        # number of objects in environment
        self.numObjs = len(objects_list)

        # string name of environment used for gym registration
        self.registered_name = "MiniGrid-Temporal-ord_obj-v0"

        # width/length of environment
        self.size = size

        # reward parameters
        self.f_reward = f_reward
        self.f_penalty = f_penalty 
        self.finish_factor = finish_factor

        # WFA monitor of preferred path
        self.wfa_monitor = wfa_monitor 
        self.WFA_num_states = wfa_monitor.num_states

        # list of actions and objects in order
        self.actions_list = actions_list
        self.objects_list = objects_list

        # Types of objects to be generated
        self.obj_types = ["key", "ball", "box"]

        self.path_log = []
        # check if word in WFA_monitor matches word fom actions and objects list
        # self.check_word()

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )
        

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.update_obs_space()
    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"go to the {color} {obj_type}"

    def update_obs_space(self):
        """Update observation space to include automaton state"""
        self.observation_space["auto_state"] = spaces.Box(
                low=-1.0,
                high=2.0,
                shape=(self.WFA_num_states, ),
                dtype=np.float32
            )
    def gen_obs(self):
        obs = super().gen_obs()
        initial_state_array  = self.wfa_monitor.initial.reshape(self.WFA_num_states, )
        obs["auto_state"]    = initial_state_array
        return obs 
    
    # def check_word(self):
    #     for index, action in enumerate( self.actions_list ):
    #         event_from_monitor = self.wfa_monitor.word[index]
    #         event_from_inputs  = action + " " + self.objects_list[index]

    #         if event_from_inputs != event_from_monitor:
    #             raise ValueError(f"Preferred words do not match: {event_from_monitor} neq {event_from_inputs} at index {index}")

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ["key", "ball", "box"]
        
        objs = []
        objPos = []

        for index, objType in enumerate(self.objects_list):

            objColor = self._rand_elem(COLOR_NAMES)
            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj)
            objColor = self._rand_elem(COLOR_NAMES)
            objs.append((objType, objColor))
            objPos.append(pos)
        self.final_obj = obj

        # Randomize the agent start position and orientation
        self.place_agent()

        # # Until we have generated all the objects
        # while len(objs) < self.numObjs:
        #     objType = self._rand_elem(types)
        #     objColor = self._rand_elem(COLOR_NAMES)

        #     # If this object already exists, try again
        #     if (objType, objColor) in objs:
        #         continue

        #     if objType == "key":
        #         obj = Key(objColor)
        #     elif objType == "ball":
        #         obj = Ball(objColor)
        #     elif objType == "box":
        #         obj = Box(objColor)
        #     else:
        #         raise ValueError(
        #             "{} object type given. Object type can only be of values key, ball and box.".format(
        #                 objType
        #             )
        #         )

        #     pos = self.place_obj(obj)
        #     objs.append((objType, objColor))
        #     objPos.append(pos)

        # # Randomize the agent start position and orientation
        # self.place_agent()

        # Choose a random object to be picked up
        # objIdx = self._rand_int(0, len(objs))
        # self.targetType, self.target_color = objs[objIdx]
        # self.target_pos = objPos[objIdx]

        # descStr = f"{self.target_color} {self.targetType}"
        # self.mission = "go to the %s" % descStr
        # print(self.mission)

    def save_path_log(self):
        # print(f"entered save_path_log")
        save_dir = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\test_results"  # where to store
        os.makedirs(save_dir, exist_ok=True)

        filename = f"WFA_TestEnv_log_episode_{self.step_count}.json"
        full_path = os.path.join(save_dir, filename)

        with open(full_path, "w") as f:
            json.dump(self.path_log, f, indent=2)

        # print(f"✅ Path log saved to {full_path}")

    def step(self, action):

        #completion ind is set to false prior to each step
        self.goal_ind = False 

        # check if agent is carrying anything in previous time step before
        # action is propagated forward
        was_carrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        # Start with a basic log entry
        log_entry = {
            "step": int(self.step_count),
            "pos": tuple(int(x) for x in self.agent_pos),
            "dir": int(self.agent_dir),
            "action": int(action),
            "reward": float(reward),
            "event": None  # We'll populate this if an important event happens
        }
        
        # Look ahead at what the agent is facing
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # --- Detect important events ---

        # 1. Case when action is pickup
        if action == self.actions.pickup:
            """
            1.a Case when agent actually picked something up
            ie "carrying and pickup"
            """
            if self.carrying:
                # carrying key, ball or box
                log_entry["event"] = f"pickup {self.carrying.type}"
                self.wfa_monitor.step(f"pickup {self.carrying.type}")
            else:
                # carrying nothing
                self.wfa_monitor.step("useless")
        # 2. action is drop 
        elif action == self.actions.drop:
            # log progress because the agent ACTUALLY dropped an item
            if was_carrying and self.carrying is None:
                # used to carry and no longer carry
                log_entry["event"] = f"dropped {was_carrying.type}"
                self.wfa_monitor.step(f"dropped {was_carrying.type}")
                "2.a case when final object is dropped. "
                if was_carrying is self.final_obj:
                    # dropped final object
                    
                    if self.wfa_monitor.complete_ind == True:
                        "termination variable updated to true"
                        terminated = True

                        "completed goal indicator"
                        self.goal_ind   = True
            else:
                # dropping without carrying is a useless action
                self.wfa_monitor.step("useless")
        elif action == self.actions.done or action == self.actions.toggle:
            # toggle and done are unused
            self.wfa_monitor.step("useless")
        else: 
            # case 3 movement occurs or irrelevant action
            self.wfa_monitor.step()
            # print("no_event")
        current_auto_state_array = self.wfa_monitor.current_state
        # add to observation dictionary-
        obs["auto_state"] = current_auto_state_array.reshape(self.WFA_num_states, )
        # print(f"current auto state index is {current_auto_state_array}")
        
        # consolidate rewards and penalties  
        if self.wfa_monitor.f_decrease:
            # penalize if WFA func decreases
            reward -= self.f_penalty 
            # print("bad decision")
            if self.wfa_monitor.unstable_ind:
                # penalize if WFA func is too close to zero
                terminated = True
                print("UNSTABLE PATH")
        else:
            # good decision
            if self.wfa_monitor.f_increase:
                # progress was made if score increased
                reward += self.f_reward
                if self.goal_ind:
                    "add efficiency reward inheritted from minigridenv"
                    reward += self.finish_factor*self._reward() 
                # print("good decision")

        "store reward into path_log dictionary for specific timestep"
        log_entry["reward"] = reward

        # --- Append the log with new entry ---
        self.path_log.append(log_entry)

        "termination/truncation commands if dropped final object"
        if truncated:
            self.path_log = []
            self.wfa_monitor.reset()
            print("TRUNCATED")
        if terminated:
            self.wfa_monitor.reset()
            if self.goal_ind:
                # self.save_path_log()
                print("reached goal!")
                # pass 
            else:
                self.path_log=[]


        return obs, reward, terminated, truncated, info

# # Instantiate your custom DoorKeyEnv
# env = TestEnv( render_mode="human")

# # Start manual control interface
# manual = ManualControl(env, seed=42)
# manual.start()