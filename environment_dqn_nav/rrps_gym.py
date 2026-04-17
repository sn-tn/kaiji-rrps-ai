from __future__ import annotations
import random
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rrps_core.types.matchup_dict import MatchupDict
from rrps_core.types.cards import Card
from rrps_core.types.info import Info, GameStatus
from rrps_core.types.player import PlayerDict, PlayerID, Budget, Player
from rrps_core.reward_config import RewardConfig as _BaseRewardConfig
from rrps_core.rrps_gym import RRPSEnvCore



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


class RestrictedRPSEnv(RRPSEnvCore):
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
            "paper_total": 3,
            "rock_total": 3,
            "scissors_total": 3,
        },
        grid_size: int = 6,
        max_turns: int = 500,
        reward_config: RewardConfig | None = None,
        n_obs_opponents: int | None = 4,
        view_radius: int = 2,
    ):
        super().__init__()
        self.n_players = n_opponents + 1
        self.n_opponents = n_opponents
        # How many nearest opponents to include in the observation.
        # Defaults to all opponents; set lower to cap obs size when scaling.
        self.n_obs_opponents = n_obs_opponents
        self.view_radius = view_radius
        self.initial_stars = stars
        self.initial_agent_budget = agent_budget
        self.initial_player_budget = player_budget
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.reward_config = reward_config or RewardConfig()

        # Flat observation vector:
        #   agent:    row, col, rock, paper, scissors, stars, turn_progress  (7)
        #   per nearest opp: is_alive, row, col, has_rock, has_paper, has_scissors  (6 each)
        #   padded with zeros when fewer than n_obs_opponents are alive
        max_cards = max(
            max(agent_budget.values()), max(player_budget.values())
        )
        n_features = 7 + 6 * self.n_obs_opponents

        high = np.array(
            [
                grid_size - 1,
                grid_size - 1,
                max_cards,
                max_cards,
                max_cards,
                20.0,
                1.0,
            ]
            + [1.0, grid_size - 1, grid_size - 1, 1.0, 1.0, 1.0]
            * self.n_obs_opponents,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.zeros(n_features, dtype=np.float32),
            high=high,
            dtype=np.float32,
        )

        # action = direction * 3 + card  (4 dirs × 3 cards = 12)
        self.action_space = gym.spaces.Discrete(12)

        self._action_to_card = {0: Card.rock, 1: Card.paper, 2: Card.scissors}

        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),  # Move down (row + 1)
        }
        self.player_dict: PlayerDict = {}

    def _random_position(self) -> np.ndarray:
        return self.np_random.integers(0, self.grid_size, size=2).astype(
            np.float32
        )

    def _initialize_players(self):
        self.player_dict = {
            i: {
                **(
                    self.initial_agent_budget
                    if i == 0
                    else self.initial_player_budget
                ),
                "stars_total": self.initial_stars,
                "position": self._random_position(),
            }
            for i in range(self.n_players)
        }
        self.still_playing_dict = self.player_dict.copy()

    def _get_obs(self) -> np.ndarray:
        """"""
        agent = self.player_dict[0]
        agent_pos = agent["position"]
        obs = [
            float(agent_pos[0]),
            float(agent_pos[1]),
            float(agent["rock_total"]),
            float(agent["paper_total"]),
            float(agent["scissors_total"]),
            float(agent["stars_total"]),
            float(self.turn) / self.max_turns,
        ]

        alive_opps = sorted(
            (
                (pid, self.player_dict[pid])
                for pid in self.still_playing_dict
                if pid != 0
                and int(
                    np.sum(
                        np.abs(self.player_dict[pid]["position"] - agent_pos)
                    )
                )
                <= self.view_radius
            ),
            key=lambda x: int(np.sum(np.abs(x[1]["position"] - agent_pos))),
        )
        for i in range(self.n_obs_opponents):
            if i < len(alive_opps):
                _, p = alive_opps[i]
                obs += [
                    1.0,
                    float(p["position"][0]),
                    float(p["position"][1]),
                    float(p["rock_total"]),
                    float(p["paper_total"]),
                    float(p["scissors_total"]),
                ]
            else:
                obs += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return np.array(obs, dtype=np.float32)

    def _get_info(
        self,
        initial_alive_player_dict: PlayerDict,
        game_status: GameStatus,
        challenge_table: None = None,
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

    def _total_cards(self, player: Player) -> int:
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
        ]  # agent death handled at step end
        self.still_playing_dict = new_playing_dict

    def _select_move(self, pid: PlayerID, table: PlayerDict) -> Card:
        available = [card for card in Card if table[pid][card.value] > 0]
        return random.choice(available)

    def _resolve_matchups(self, matchup_dict: MatchupDict):
        for (pid1, pid2), (card1, card2) in matchup_dict.items():
            result = resolve(card1, card2)
            if result == 1:
                self.player_dict[pid1]["stars_total"] += 1
                self.player_dict[pid2]["stars_total"] -= 1
            elif result == -1:
                self.player_dict[pid2]["stars_total"] += 1
                self.player_dict[pid1]["stars_total"] -= 1
            self.player_dict[pid1][card1.value] -= 1
            self.player_dict[pid2][card2.value] -= 1

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False
        game_status: GameStatus = "playing"
        matchup_dict: MatchupDict = {}

        initial_alive_player_dict: PlayerDict = self.still_playing_dict.copy()
        agent = self.player_dict[0]

        # ── Decode action: direction * 3 + card ──────────────────────────
        direction = self._action_to_direction[action // 3]
        card = self._action_to_card[action % 3]

        if agent[card.value] == 0:
            reward += self.reward_config.invalid_move
            available = [c for c in Card if agent[c.value] > 0]
            card = random.choice(available) if available else card

        # ── Move agent, reward for closing distance ───────────────────────
        # Snapshot opponent positions before anyone moves
        old_pos = agent["position"].copy()
        new_pos = np.clip(agent["position"] + direction, 0, self.grid_size - 1)

        opp_positions = [
            p["position"].copy()
            for pid, p in self.still_playing_dict.items()
            if pid != 0
        ]
        if opp_positions:
            old_dist = min(np.sum(np.abs(old_pos - q)) for q in opp_positions)
            new_dist = min(np.sum(np.abs(new_pos - q)) for q in opp_positions)
            if new_dist < old_dist:
                reward += self.reward_config.approach_opponent

        agent["position"] = new_pos

        # ── Move opponents (random walk) ──────────────────────────────────
        for pid in list(self.still_playing_dict.keys()):
            if pid == 0:
                continue
            opp_dir = self._action_to_direction[random.randint(0, 3)]
            p = self.player_dict[pid]
            p["position"] = np.clip(
                p["position"] + opp_dir, 0, self.grid_size - 1
            )

        # ── Build matchups from co-located pairs ──────────────────────────
        # Each player fights at most once per step to prevent card counts going negative.
        alive_pids = list(self.still_playing_dict.keys())
        matched: set[PlayerID] = set()

        for i, pid1 in enumerate(alive_pids):
            if pid1 in matched:
                continue
            for pid2 in alive_pids[i + 1 :]:
                if pid2 in matched:
                    continue
                p1 = self.player_dict[pid1]
                p2 = self.player_dict[pid2]
                if not np.array_equal(p1["position"], p2["position"]):
                    continue
                if self._total_cards(p1) == 0 or self._total_cards(p2) == 0:
                    continue

                if pid1 == 0:
                    c1 = card
                    c2 = self._select_move(pid2, self.still_playing_dict)
                elif pid2 == 0:
                    c1 = self._select_move(pid1, self.still_playing_dict)
                    c2 = card
                else:
                    c1 = self._select_move(pid1, self.still_playing_dict)
                    c2 = self._select_move(pid2, self.still_playing_dict)

                matchup_dict[(pid1, pid2)] = (c1, c2)
                matched.add(pid1)
                matched.add(pid2)
                break  # pid1 is now matched; move to next pid1

        # ── Resolve matchups ──────────────────────────────────────────────
        agent_stars_before = agent["stars_total"]
        self._resolve_matchups(matchup_dict)
        agent_stars_after = agent["stars_total"]

        agent_matched = any(0 in pair for pair in matchup_dict)
        if agent_matched:
            delta = agent_stars_after - agent_stars_before
            if delta > 0:
                reward += self.reward_config.win_matchup
            elif delta < 0:
                reward += self.reward_config.lose_matchup
            else:
                reward += self.reward_config.tie_matchup

        # Proximity reward: agent shares a cell with a live opponent
        if any(
            np.array_equal(
                agent["position"], self.player_dict[pid]["position"]
            )
            for pid in alive_pids
            if pid != 0
        ):
            reward += self.reward_config.within_challenge_range

        # ── Update alive dict ─────────────────────────────────────────────
        self._update_playing()

        # ── Termination ───────────────────────────────────────────────────
        agent_no_cards = self._total_cards(agent) == 0
        agent_no_stars = agent["stars_total"] <= 0

        if agent_no_cards or agent_no_stars:
            terminated = True
            if agent["stars_total"] >= self.initial_stars:
                game_status = "victory"
                reward += self.reward_config.victory
            else:
                game_status = "eliminated"
                reward += self.reward_config.eliminated
        elif self.turn >= self.max_turns:
            terminated = True
            game_status = "eliminated"
            reward += self.reward_config.eliminated
        elif len(self.still_playing_dict) == 1:
            # agent is the last player standing
            terminated = True
            game_status = "victory"
            reward += self.reward_config.victory

        obs = self._get_obs()
        info = self._get_info(
            initial_alive_player_dict, game_status, None, matchup_dict
        )
        self.turn += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
