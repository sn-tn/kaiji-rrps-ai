from abc import ABC, abstractmethod,  classmethod
from gym_core.info import Info
import gymnasium as gym


class RRPSEnvCore(gym.Env, ABC):

    @abstractmethod
    def _get_obs(self): ...

    """return the observation used in the hash function/state key"""

    @abstractmethod
    def _get_info(self) -> Info: ...

    """return the info object used for reviewing/inspecting/degugging the enviroment"""

    @abstractmethod
    def step(self, action): ...

    @abstractmethod
    def reset(self, *, seed=None, options=None): ...

    