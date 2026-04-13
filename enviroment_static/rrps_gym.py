from __future__ import annotations
import random
from dataclasses import dataclass
from typing import TypedDict
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pandera.pandas as pa
import numpy as np

from gym_core.matchup_dict import MatchupDict
from gym_core.challenge_table import ChallengeTable, ChallengeSchema
from gym_core.observation import Observation
from gym_core.cards import Card
from gym_core.info import Info, GameStatus
from gym_core.player import PlayerDict, PlayerID, Budget, Player
from gym_core.matchup_dict import MatchupPair, MatchupDict


@dataclass
class RewardConfig:
    win_matchup: float = 100
    lose_matchup: float = -100
    tie_matchup: float = 10
    eliminated: float = -500
    victory: float = 2000
    invalid_move: float = -10
    within_challenge_range: float = 1
    approach_opponent: float = 0.5
    has_cards_at_end: float = -200
    has_sub_3_stars_at_end: float = -300


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
            "paper_total": 9,
            "rock_total": 0,
            "scissors_total": 0,
        },
        max_turns: int = 500,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.n_players = n_opponents + 1
        self.initial_stars = stars
        self.initial_agent_budget = agent_budget
        self.initial_player_budget = player_budget
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()

        # observation space
        self.observation_space = spaces.Dict(
            {i: spaces.Discrete(self.n_players) for i in range(n_opponents)}
        )

        # action space opponents * rps
        self.action_space = gym.spaces.Discrete((self.n_players - 1) * 3)

        self._action_to_card = {0: Card.rock, 1: Card.paper, 2: Card.scissors}

        #
        # self._action_to_direction = {
        #     0: np.array([0, 1]),  # Move right (column + 1)
        #     1: np.array([-1, 0]),  # Move up (row - 1)
        #     2: np.array([0, -1]),  # Move left (column - 1)
        #     3: np.array([1, 0]),  # Move down (row + 1)
        # }
        self.player_dict: PlayerDict = {}

    def _initialize_players(self):
        self.player_dict = {
            0: {**self.initial_agent_budget, "stars_total": self.initial_stars}
        }
        self.player_dict.update(
            {
                i: {
                    **self.initial_player_budget,
                    "stars_total": self.initial_stars,
                }
                for i in range(1, self.n_players)
            }
        )
        self.still_playing_dict = self.player_dict

    def _get_obs(self) -> Observation:

        obs: Observation = {
            "player_dict": self.player_dict,
            "turn": self.turn,
        }
        return obs

    def _get_info(
        self,
        initial_alive_player_dict: PlayerDict,
        game_status: GameStatus,
        challenge_table: ChallengeTable | None = None,
        matchup_dict: MatchupDict | None = None,
    ) -> Info:
        return {
            "challenge_table": challenge_table,
            "matchup_dict": matchup_dict,
            "alive_player_dict": self.still_playing_dict,
            "round_number": self.turn,
            "initial_alive_player_dict": initial_alive_player_dict,
            "game_status": game_status,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._initialize_players()
        self.turn = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def action_resolve(self, action: int) -> tuple[PlayerID, Card]:
        """converts action space
        0..3 will be one player so action//3 = playerId
        0..3 will be rock paper scissors so action%3 = card"""
        target_pid = list(self.player_dict.keys())[action // 3 + 1]
        card = self._action_to_card[action % 3]
        return target_pid, card

    def _total_cards(self, player: Player):
        return (
            player["rock_total"]
            + player["scissors_total"]
            + player["paper_total"]
        )

    def _update_playing(self):
        new_playing_dict = {}
        for pid in self.still_playing_dict.keys():
            player = self.player_dict[pid]
            is_playing = (
                self._total_cards(player) > 0 and player["stars_total"] > 0
            )

            if is_playing:
                new_playing_dict[pid] = player

        new_playing_dict[0] = self.player_dict[
            0
        ]  # agent death handled at end of step function
        self.still_playing_dict = new_playing_dict

    # all players passed in through PlayerDict MUST be eligible to play (cards remaining)

    def _rank_opponents(
        self, pid: PlayerID, table: PlayerDict
    ) -> list[PlayerID]:
        candidates = [oid for oid in table if oid != pid]
        return random.sample(candidates, min(3, len(candidates)))

    # TODO
    def _agent_rank_opponents(
        self, pid: PlayerID, table: PlayerDict
    ) -> list[PlayerID]:
        candidates = [oid for oid in table if oid != pid]
        return random.sample(candidates, min(3, len(candidates)))

    def _select_move(self, pid: PlayerID, table: PlayerDict) -> Card:
        available = [card for card in Card if table[pid][card.value] > 0]
        return random.choice(available)

    def _get_card(
        self, pid: PlayerID, challenge_table: ChallengeTable
    ) -> Card:
        row = challenge_table.loc[challenge_table["player_id"] == pid].iloc[0]
        return Card(row["card"])

    def build_challenge_table(
        self,
        table: PlayerDict,
        agent_card: Card | None = None,
        agent_target: PlayerID | None = None,
    ) -> ChallengeTable:
        rows: list[dict] = []

        for pid in table:
            if pid == 0:
                card = agent_card
                targets = [agent_target] + self._agent_rank_opponents(
                    pid, table
                )
                targets = targets[:3]  # keep max 3
            else:
                card = self._select_move(pid, table)
                targets = self._rank_opponents(pid, table)

            for rank, target_id in enumerate(targets):
                rows.append(
                    {
                        "player_id": pid,
                        "card": str(card),
                        "priority": rank,
                        "target_id": target_id,
                    }
                )

        df = pd.DataFrame(
            rows, columns=["player_id", "card", "priority", "target_id"]
        )
        return ChallengeSchema.validate(df)

    def resolve_challenges(
        self, table: PlayerDict, challenge_table: ChallengeTable
    ) -> MatchupDict:
        prefs: dict[PlayerID, list[PlayerID]] = (
            challenge_table.sort_values(
                ["player_id", "priority"]
            )  # priority col makes order guaranteed
            .groupby("player_id", sort=False)["target_id"]
            .apply(list)
            .to_dict()
        )

        all_ids = list(table.keys())
        unmatched: set[PlayerID] = set(all_ids)
        matchups: MatchupDict = {}

        # first pass - match mutual top picks
        # shuffled to avoid priority
        shuffled = all_ids.copy()
        random.shuffle(shuffled)

        for pid in shuffled:
            if pid not in unmatched:
                continue
            for candidate in prefs.get(pid, []):
                if candidate not in unmatched:
                    continue
                if pid in prefs.get(candidate, []):
                    matchups[(pid, candidate)] = (
                        self._get_card(pid, challenge_table),
                        self._get_card(candidate, challenge_table),
                    )
                    unmatched -= {pid, candidate}
                    break

        # second pass - one-sided interest
        for pid in shuffled:
            if pid not in unmatched:
                continue
            for candidate in prefs.get(pid, []):
                if candidate in unmatched:
                    matchups[(pid, candidate)] = (
                        self._get_card(pid, challenge_table),
                        self._get_card(candidate, challenge_table),
                    )
                    unmatched -= {pid, candidate}
                    break

        # third pass - make sure the agents paired
        if 0 in unmatched:
            opponents = [pid for pid in unmatched if pid != 0]
            if opponents:
                opponent = random.choice(opponents)
                matchups[(0, opponent)] = (
                    self._get_card(0, challenge_table),
                    self._get_card(opponent, challenge_table),
                )
                unmatched -= {0, opponent}

        # fourth pass - random fallback
        leftover = list(unmatched)
        random.shuffle(leftover)
        while len(leftover) >= 2:
            a = leftover.pop()
            b = leftover.pop()
            matchups[(a, b)] = (
                self._get_card(a, challenge_table),
                self._get_card(b, challenge_table),
            )

        return matchups

    def resolve_matchups(
        self, matchup_dict: MatchupDict, player_dict: PlayerDict
    ):
        for (pid1, pid2), (card1, card2) in matchup_dict.items():
            result = resolve(card1, card2)
            if result == 1:  # pid1 wins
                player_dict[pid1]["stars_total"] += 1
                player_dict[pid2]["stars_total"] -= 1
            elif result == -1:  # pid2 wins
                player_dict[pid2]["stars_total"] += 1
                player_dict[pid1]["stars_total"] -= 1
            # deduct cards regardless of outcome
            player_dict[pid1][card1.value] -= 1
            player_dict[pid2][card2.value] -= 1

    def step(self, action: int):
        # assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        game_status: GameStatus = "playing"
        info = {}

        # decode action
        target_pid, card = self.action_resolve(action)

        # validate
        target = self.player_dict[target_pid]
        agent = self.player_dict[0]

        challenge_table: ChallengeTable | None = None
        matchup_dict: MatchupDict | None = None
        initial_alive_player_dict: PlayerDict = self.still_playing_dict
        if target_pid not in self.still_playing_dict:
            reward += self.reward_config.invalid_move
        elif agent[card.value] == 0:
            reward += self.reward_config.invalid_move
        else:
            # build challenge table
            challenge_table = self.build_challenge_table(
                self.still_playing_dict,
                agent_card=card,
                agent_target=target_pid,
            )

            matchup_dict = self.resolve_challenges(
                self.still_playing_dict, challenge_table
            )
            # resolve matchups
            agent_stars_before = self.player_dict[0]["stars_total"]
            self.resolve_matchups(matchup_dict, self.player_dict)
            agent_stars_after = self.player_dict[0]["stars_total"]

            # reward based on matchup result
            if agent_stars_after > agent_stars_before:
                reward += self.reward_config.win_matchup
            elif agent_stars_after < agent_stars_before:
                reward += self.reward_config.lose_matchup
            else:
                reward += self.reward_config.tie_matchup

        # update alive_dict
        self._update_playing()

        # check termination
        agent_no_cards_left = self._total_cards(self.player_dict[0]) == 0
        agent_gte_three_stars = self.player_dict[0]["stars_total"] >= 3
        if agent_no_cards_left:
            terminated = True
            # no cards 3 or more stars
            if agent_gte_three_stars:
                game_status = "victory"
                reward += self.reward_config.victory
            # no cards less than 3 stars
            else:
                game_status = "lost"
                reward += self.reward_config.eliminated
        # max turns or last player alive
        elif self.turn > self.max_turns or len(self.still_playing_dict) == 1:
            reward += self.reward_config.eliminated
            truncated = True
            game_status = "lost"

        obs = self._get_obs()
        info = self._get_info(
            initial_alive_player_dict,
            game_status,
            challenge_table,
            matchup_dict,
        )
        self.turn += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
