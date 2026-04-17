from __future__ import annotations

import random
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rrps_core.cards import Card
from rrps_core.info import Info, GameStatus
from rrps_core.matchup_dict import MatchupDict
from rrps_core.observation import Observation
from rrps_core.player import PlayerDict, PlayerID, Player, Budget
from rrps_core.reward_config import RewardConfig as _BaseRewardConfig
from rrps_core.rrps_gym import RRPSEnvCore





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


class RestrictedRPSEnv(RRPSEnvCore):
    """
    A single-agent Gymnasium environment for Restricted Rock Paper Scissors.

    The *agent* controls player 0; all other players move toward the nearest
    opponent each step. Matchups occur when two players share the same cell.

    Action (Discrete 12):  direction * 3 + card
        Directions: 0=right, 1=up, 2=left, 3=down
        Cards:      0=rock,  1=paper, 2=scissors
    """

    metadata = {"render_modes": ["human"]}

    _action_to_direction = {
        0: np.array([0, 1]),
        1: np.array([-1, 0]),
        2: np.array([0, -1]),
        3: np.array([1, 0]),
    }
    _action_to_card = {0: Card.rock, 1: Card.paper, 2: Card.scissors}

    def __init__(
        self,
        n_opponents: int = 3,
        stars: int = 3,
        agent_budget: Budget = {
            "rock_total": 4,
            "paper_total": 4,
            "scissors_total": 4,
        },
        player_budget: Budget = {
            "rock_total": 4,
            "paper_total": 4,
            "scissors_total": 4,
        },
        grid_size: int = 20,
        max_turns: int = 2000,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.n_opponents = n_opponents
        self.initial_stars = stars
        self.initial_agent_budget = agent_budget
        self.initial_player_budget = player_budget
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        max_stars = stars + n_opponents * stars
        g = grid_size - 1
        player_space = spaces.Dict(
            {
                "player_id": spaces.Discrete(n_opponents + 1),
                "stars_total": spaces.Discrete(max_stars + 1),
                "budget": spaces.Dict(
                    {
                        "rock_total": spaces.Discrete(2),
                        "paper_total": spaces.Discrete(2),
                        "scissors_total": spaces.Discrete(2),
                    }
                ),
                "position": spaces.Tuple(
                    (spaces.Discrete(g + 1), spaces.Discrete(g + 1))
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "agent": player_space,
                "opponent": player_space,
                "opponents_alive": spaces.Discrete(n_opponents + 1),
            }
        )
        self.action_space = spaces.Discrete(12)

        self.player_dict: PlayerDict = {}
        self.still_playing_dict: PlayerDict = {}

    # ── private helpers ───────────────────────────────────────────────────────

    def _random_position(self) -> np.ndarray:
        return np.array(
            [
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            ]
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
            for i in range(self.n_opponents + 1)
        }
        self.still_playing_dict = self.player_dict.copy()

    def _total_cards(self, player: Player) -> int:
        return (
            player["rock_total"]
            + player["paper_total"]
            + player["scissors_total"]
        )

    def _has_cards(self, pid: PlayerID) -> bool:
        return self._total_cards(self.player_dict[pid]) > 0

    def _is_alive(self, pid: PlayerID) -> bool:
        return self.player_dict[pid]["stars_total"] > 0

    def _update_playing(self):
        new_playing = {}
        for pid, player in self.still_playing_dict.items():
            if self._total_cards(player) > 0 and player["stars_total"] > 0:
                new_playing[pid] = player
        new_playing[0] = self.player_dict[0]  # agent death handled at step end
        self.still_playing_dict = new_playing

    def _alive_opponents(self) -> list[PlayerID]:
        return [pid for pid in self.still_playing_dict if pid != 0]

    def _nearest(
        self, pid: PlayerID, candidates: list[PlayerID]
    ) -> PlayerID | None:
        if not candidates:
            return None
        pos = self.player_dict[pid]["position"]
        return min(
            candidates,
            key=lambda p: int(
                np.sum(np.abs(self.player_dict[p]["position"] - pos))
            ),
        )

    def _select_move(self, pid: PlayerID) -> Card:
        available = [
            c for c in Card if self.still_playing_dict[pid][c.value] > 0
        ]
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

    def _get_obs(self) -> Observation:

        obs: Observation = {
            "player_dict": self.player_dict,
            "turn": self._turn,
        }
        return obs

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
            "round_number": self._turn,
            "initial_alive_player_dict": initial_alive_player_dict,
            "game_status": game_status,
        }

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._initialize_players()
        self._turn = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        game_status: GameStatus = "playing"
        matchup_dict: MatchupDict = {}

        initial_alive_player_dict: PlayerDict = self.still_playing_dict.copy()
        agent = self.player_dict[0]

        # ── Decode action: direction * 3 + card ──────────────────────────────
        direction = self._action_to_direction[action // 3]
        card = self._action_to_card[action % 3]

        if agent[card.value] == 0:
            reward += self.reward_config.invalid_move
            available = [c for c in Card if agent[c.value] > 0]
            card = random.choice(available) if available else card

        # ── Move agent ────────────────────────────────────────────────────────
        agent["position"] = np.clip(
            agent["position"] + direction, 0, self.grid_size - 1
        )

        # ── Move opponents (random walk) ──────────────────────────────────
        for pid in list(self.still_playing_dict.keys()):
            if pid == 0:
                continue
            opp_dir = self._action_to_direction[random.randint(0, 3)]
            p = self.player_dict[pid]
            p["position"] = np.clip(
                p["position"] + opp_dir, 0, self.grid_size - 1
            )

        # ── Build matchups from co-located pairs ──────────────────────────────
        """
        If two players are in the same cell auto match 
        """
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
                c1 = card if pid1 == 0 else self._select_move(pid1)
                c2 = card if pid2 == 0 else self._select_move(pid2)
                matchup_dict[(pid1, pid2)] = (c1, c2)
                matched.add(pid1)
                matched.add(pid2)
                break

        # ── Resolve matchups ──────────────────────────────────────────────────
        agent_stars_before = agent["stars_total"]
        self._resolve_matchups(matchup_dict)
        agent_stars_after = agent["stars_total"]

        if any(0 in pair for pair in matchup_dict):
            delta = agent_stars_after - agent_stars_before
            if delta > 0:
                reward += self.reward_config.win_matchup
            elif delta < 0:
                reward += self.reward_config.lose_matchup
            else:
                reward += self.reward_config.tie_matchup

        # ── Update alive dict ─────────────────────────────────────────────────
        self._update_playing()

        # ── Termination ───────────────────────────────────────────────────────
        agent_no_cards = self._total_cards(agent) == 0
        agent_no_stars = agent["stars_total"] <= 0

        if agent_no_cards or agent_no_stars:
            terminated = True
            game_status = (
                "victory"
                if agent["stars_total"] >= self.initial_stars
                else "eliminated"
            )
            reward += (
                self.reward_config.victory
                if game_status == "victory"
                else self.reward_config.eliminated
            )
        elif len(self.still_playing_dict) == 1:
            terminated = True
            game_status = "victory"
            reward += self.reward_config.victory
        elif self._turn >= self.max_turns:
            terminated = True
            game_status = "eliminated"
            reward += self.reward_config.eliminated

        obs = self._get_obs()
        info = self._get_info(
            initial_alive_player_dict, game_status, None, matchup_dict
        )
        self._turn += 1
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        p = self.player_dict[0]
        print(
            f"[Agent] pos={p['position']} stars={p['stars_total']}"
            f" budget=R{p['rock_total']}/P{p['paper_total']}/S{p['scissors_total']}"
            f" | Alive opponents: {len(self._alive_opponents())}"
        )

    def close(self):
        pass
