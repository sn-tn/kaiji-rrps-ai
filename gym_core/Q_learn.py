from typing import Dict, Any
from abc import ABC, abstractmethod
from gym_core.info import Info
import gymnasium as gym
from gym_core.rrps_gym import RRPSEnvCore
from typing import TypeVar, Generic, Generator
import numpy as np
from tqdm import tqdm
import pickle

ObsType = TypeVar("ObsType")


class RRPSQLearnCore(Generic[ObsType]):
    def __init__(self, env: RRPSEnvCore, agent_name: str) -> None:
        if not isinstance(env, RRPSEnvCore):
            raise TypeError(
                "env must be an instance of RRPSEnvCore, got"
                f" {type(env).__name__}"
            )
        self.env = env
        self.agent_name = agent_name
        self.Q_table: Dict[Any, Any] | None = None

    @abstractmethod
    def hash(self, obs) -> np.ndarray:
        """function used to convert the observation to a"""
        ...

    @staticmethod
    def softmax(x, temp=1.0):
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / e_x.sum(axis=0)

    def agent_move(self, obs):
        if self.Q_table is None:
            raise "No agent Loaded"
        state = self.hash(obs)
        Q_val = self.Q_table[state]
        return self.softmax(Q_val)

    def tabular_train(
        self,
        train_episodes: int,
        gamma: float,
        decay_rate: float,
        epsilon: float = 1,
        gui: bool = False,
    ):
        Q_table = {}
        Q_update_counts = {}

        for _ in tqdm(range(train_episodes)):
            start_obs, _ = self.env.reset()
            prev_state_key = self.hash(start_obs)
            if prev_state_key not in Q_update_counts:
                Q_update_counts[prev_state_key] = np.zeros(
                    self.env.action_space.n
                )
            if prev_state_key not in Q_table:
                Q_table[prev_state_key] = np.zeros(self.env.action_space.n)
            while True:
                # initialize prev state keys if not already
                action = None

                # want random action w/ prob P = epsilon
                if np.random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(Q_table[prev_state_key])

                # transition to state s'
                new_obs, reward, terminated, truncated, info = self.env.step(
                    action
                )
                new_state_key = self.hash(new_obs)
                ## initalize state action if not already
                if new_state_key not in Q_update_counts:
                    Q_update_counts[new_state_key] = np.zeros(
                        self.env.action_space.n
                    )
                if new_state_key not in Q_table:
                    Q_table[new_state_key] = np.zeros(self.env.action_space.n)

                # calculate Q vals
                Q_old_update_counts = Q_update_counts[prev_state_key][action]
                Q_old = Q_table[prev_state_key][action]
                V_opt_old = np.max(Q_table[new_state_key])
                eta = 1 / (1 + Q_old_update_counts)
                Q_new = (1 - eta) * Q_old + eta * (reward + gamma * V_opt_old)

                # update table
                Q_table[prev_state_key][action] = Q_new
                Q_update_counts[prev_state_key][action] += 1

                if gui:
                    self.render_gui(terminated, truncated, info)
                # update epsilon and end or continue w/ new step as prev

                if terminated:
                    epsilon *= decay_rate
                    break
                else:
                    prev_state_key = new_state_key
        with open(
            self.agent_name
            + str(train_episodes)
            + "_"
            + str(decay_rate)
            + ".pickle",
            "wb",
        ) as handle:
            pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    @classmethod
    def render_gui(cls, terminated, truncated, info: dict[str, Any]):
        """optional, render visualisation of training"""
        ...

    def play_agent(self, num_episodes: int, gui: bool = False) -> Generator[tuple[ObsType, float, bool, bool, dict], None, None]:
        """generator that plays through agent for a number of episodes, yields the gym step return values"""
        obs, info = self.env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            try:
                action = np.random.choice(
                    self.env.action_space.n, p=self.agent_move(obs)
                )  # Select action using softmax over Q-values
            except KeyError:
                action = (
                    self.env.action_space.sample()
                )  # Fallback to random action if state not in Q-table

            obs, reward, terminated, truncated, info = self.env.step(action)

            yield obs, reward, terminated, truncated, info

            if gui:
                self.render_gui(terminated, truncated, info)
            total_reward += reward
        return self
