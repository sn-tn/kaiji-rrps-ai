import random
from dataclasses import dataclass
from typing import TypedDict
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

import numpy as np

from gym_core.player import PlayerDict
from gym_core.matchup_table import MatchupDict
from gym_core.challenge_table import ChallengeTable
from gym_core.observation import Observation
from gym_core.cards import Card

from gym_core.player import Budget


@dataclass
class RewardConfig:
    win_matchup: float = 100
    lose_matchup: float = -100
    tie_matchup: float = 10
    eliminated: float = -300
    victory: float = 500
    invalid_move: float = -1
    within_challenge_range: float = 1
    approach_opponent: float = 0.5
    has_cards_at_end: float = -500
    has_sub_3_stars_at_end: float = -500


# ── helpers ────────────────────────────────────────────────────────────────────────────


def resolve(m1: Card, m2: Card) -> int:
    """Returns 1 if m1 wins, -1 if m2 wins, 0 on tie."""

    wins_against = {
        Card.rock: Card.scissors,
        Card.paper: Card.rock,
        Card.scissors: Card.paper,
    }
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


# ── environment ───────────────────────────────────────────────────────────────────


class RestrictedRPSEnv(gym.Env):
    def __init__(
        self,
        n_opponents: int = 6,
        stars: int = 3,
        agent_budget: Budget = {
            "paper_total": 3,
            "rock_total": 3,
            "scissors_total": 3,
        },
        player_budget: Budget = {
            "paper_total": 1,
            "rock_total": 1,
            "scissors_total": 1,
        },
        grid_size: int = 20,
        challenge_radius: int = 1,
        max_turns: int = 2000,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.n_players = n_opponents + 1
        self.initial_stars = stars
        self.initial_agent_budget = agent_budget
        self.initial_player_budget = player_budget
        self.grid_size = grid_size
        self.challenge_radius = challenge_radius
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        # observation space
        self.observation_space = spaces.Dict(
            {i: spaces.Discrete(self.num_players) for i in range(n_opponents)}
        )

        # action space opponents * rps
        self.action_space = gym.spaces.Discrete((self.n_players - 1) * 3)

        self._action_to_challenge = [
            {0: Card.PAPER, 1: Card.ROCK, 2: Card.SCISSORS}
            for _ in range(self.n_players - 1)
        ]
        #
        # self._action_to_direction = {
        #     0: np.array([0, 1]),  # Move right (column + 1)
        #     1: np.array([-1, 0]),  # Move up (row - 1)
        #     2: np.array([0, -1]),  # Move left (column - 1)
        #     3: np.array([1, 0]),  # Move down (row + 1)
        # }
        self.player_dict: PlayerDict = {}
        self.challenge_table: ChallengeTable = pd.DataFrame(
            {
                "player_id": pd.Series(dtype=int),
                "card": pd.Series(dtype=str),
                "target_id": pd.Series(dtype=int),
            }
        )
        self.matchup_dict: MatchupDict = {}

    def reset(self):
        # agent always zero initilize w/ params
        self.player_dict[0] = {
            **self.initial_agent_budget,
            "stars_total": self.initial_stars,
        }

        # rest of player initilized, start at 1
        player_init = {
            **self.initial_player_budget,
            "stars_total": self.initial_stars,
        }
        self.player_dict = {i: player_init for i in range(1, self.n_players)}

    def _get_obs(self) -> Observation:

        obs: Observation = {
            "player_dict": self.player_dict,
            "turn": self.turn,
        }
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # agent always zero initilize w/ params
        self.player_dict[0] = {
            **self.initial_agent_budget,
            "stars_total": self.initial_stars,
        }
        # rest of player initilized, start at 1
        player_init = {
            **self.initial_player_budget,
            "stars_total": self.initial_stars,
        }
        self.player_dict = {i: player_init for i in range(1, self.n_players)}
        self.turn = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


    def close(self):
        pass
