from abc import ABC, abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

# old
class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

# new
# class ACModel(nn.Module, ABC):
#     recurrent = False

#     def __init__(self):
#         super().__init__()

#     @abstractmethod
#     def forward(self, obs):
#         pass

# old 
class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass

# new
# class RecurrentACModel(ACModel):
#     recurrent = True

#     @abstractmethod
#     def forward(self, obs, memory):
#         pass

#     @property
#     @abstractmethod
#     def memory_size(self):
#         pass
