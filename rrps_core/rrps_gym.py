from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from rrps_core.info import Info
import gymnasium as gym

ObsType = TypeVar("ObsType")


class RRPSEnvCore(gym.Env, ABC, Generic[ObsType]):

    @abstractmethod
    def _get_obs(self):
        """return the observation used in the hash function/state key"""
        ...

    @abstractmethod
    def _get_info(self) -> Info:
        """return the info object used for reviewing/inspecting/degugging the enviroment"""
        ...

    @abstractmethod
    def step(self, action) -> tuple[ObsType, float, bool, bool, Info]: ...

    @abstractmethod
    def reset(self, *, seed=None, options=None): ...
