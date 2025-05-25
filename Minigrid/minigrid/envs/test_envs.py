from __future__ import annotations
import os 
import json 
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.manual_control import ManualControl
from AUTOMATA.auto_funcs import dfa_T1, DFAMonitor, WFA_monitor
from automata.fa.dfa import DFA
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
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
                 WFA, 
                 prob_penalty = 0.05,
                 prob_reward  = 0.75,
                 finish_factor = 3,
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

        # set dfa to initial state
        # self.dfa_monitor.reset()
        "reward gained for achieving one sub task "
        "(making forward transition in auto_task)"
        self.auto_reward = auto_reward
        self.WFA_num_states = self.WFA.n
        self.prob_penalty = prob_penalty
        self.prob_reward  = prob_reward
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
        if self.WFA_monitor.prob_decrease:
            reward -= self.prob_penalty 
            # print("bad decision")
            if self.WFA_monitor.unstable_ind:
                # probability is too close to zero
                terminated = True
                # print("UNSTABLE PATH")
        else:
            # good decision
            if self.progress:
                # progress was made and it was a good decision
                # if probability does not decrease significantly 
                # then it was a good decision
                reward += self.prob_reward
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
                 prob_penalty = 0.01,
                 prob_reward  = 0.75,
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

        # set dfa to initial state
        # self.dfa_monitor.reset()
        "reward gained for achieving one sub task "
        "(making forward transition in auto_task)"
        self.auto_reward = auto_reward
        self.WFA_num_states = self.WFA.n
        self.prob_penalty = prob_penalty
        self.prob_reward  = prob_reward
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
            reward -= self.prob_penalty
        
        elif self.WFA_monitor.prob_increase:
            # good decision
            if self.progress:
                # progress was made and it was a good decision
                # if probability increases significantly 
                # then it was a good decision
                reward += self.prob_reward
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
        #         reward += self.prob_reward
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

# # Instantiate your custom DoorKeyEnv
# env = TestEnv( render_mode="human")

# # Start manual control interface
# manual = ManualControl(env, seed=42)
# manual.start()