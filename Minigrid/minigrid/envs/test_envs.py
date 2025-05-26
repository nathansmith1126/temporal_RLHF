from __future__ import annotations
import os 
import json 
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.manual_control import ManualControl
from AUTOMATA.auto_funcs import dfa_T1, DFAMonitor
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
        self.trace = []

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

        self.trace.append(action)

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

# # Instantiate your custom DoorKeyEnv
# env = TestEnv( render_mode="human")

# # Start manual control interface
# manual = ManualControl(env, seed=42)
# manual.start()
