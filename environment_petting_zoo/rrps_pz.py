from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from gym_core.cards import Card
from gym_core.player import Budget, Player, PlayerDict


CARD_ORDER: tuple[Card, Card, Card] = (Card.rock, Card.paper, Card.scissors)


@dataclass
class RewardConfig:
    win_matchup: float = 5.0
    lose_matchup: float = -5.0
    tie_matchup: float = 0.5
    invalid_action: float = -2.0
    cleared_game: float = 20.0
    failed_clear: float = -20.0
    eliminated: float = -25.0
    last_agent_standing: float = 30.0


def resolve(m1: Card, m2: Card) -> int:
    """Return 1 if m1 wins, -1 if m2 wins, 0 on tie."""
    wins_against = {
        Card.rock: Card.scissors,
        Card.paper: Card.rock,
        Card.scissors: Card.paper,
    }
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


def env(**kwargs):
    """AEC-style wrapped environment, matching the PettingZoo docs pattern."""
    render_mode = kwargs.get("render_mode")
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    wrapped = raw_env(**{**kwargs, "render_mode": internal_render_mode})
    if render_mode == "ansi":
        wrapped = wrappers.CaptureStdoutWrapper(wrapped)
    wrapped = wrappers.AssertOutOfBoundsWrapper(wrapped)
    wrapped = wrappers.OrderEnforcingWrapper(wrapped)
    return wrapped


def raw_env(**kwargs):
    """AEC view generated from the core parallel env."""
    return parallel_to_aec(parallel_env(**kwargs))


class parallel_env(ParallelEnv):
    """
    Minimal simultaneous-move PettingZoo environment for the static RRPS setting.

    Design goals for this first pass:
    - lives entirely inside environment_petting_zoo
    - reuses gym_core's Card and Player/Budget types
    - no movement mechanic
    - fixed-size action space with action masking
    - per-episode history that can be dumped to JSON for later analysis
    """

    metadata = {
        "render_modes": ["human"],
        "name": "kaiji_rrps_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        n_players: int = 4,
        stars: int = 3,
        initial_budget: Budget | None = None,
        max_rounds: int = 100,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ):
        if n_players < 2:
            raise ValueError("n_players must be at least 2")

        self.n_players = n_players
        self.initial_stars = stars
        self.initial_budget: Budget = initial_budget or {
            "rock_total": 4,
            "paper_total": 4,
            "scissors_total": 4,
        }
        self.max_rounds = max_rounds
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(self.n_players)]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.id_to_agent = {
            idx: agent for agent, idx in self.agent_name_mapping.items()
        }

        self._action_size = self.n_players * len(CARD_ORDER)
        self._obs_size = 1 + 5 * self.n_players

        self.player_dict: PlayerDict = {}
        self.agents: list[str] = []
        self.round_count = 0
        self.history: list[dict[str, Any]] = []
        self.last_round_summary: list[str] = []
        self.np_random = None
        self.np_random_seed = None

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Discrete(self._action_size, seed=self.np_random_seed)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        obs_high = np.full(self._obs_size, 1000, dtype=np.int32)
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=obs_high,
                    shape=(self._obs_size,),
                    dtype=np.int32,
                ),
                "action_mask": spaces.MultiBinary(self._action_size),
            }
        )

    def _new_player(self) -> Player:
        return {
            "rock_total": self.initial_budget["rock_total"],
            "paper_total": self.initial_budget["paper_total"],
            "scissors_total": self.initial_budget["scissors_total"],
            "stars_total": self.initial_stars,
            "position": None,
        }

    def _cards_remaining(self, pid: int) -> int:
        player = self.player_dict[pid]
        return (
            player["rock_total"]
            + player["paper_total"]
            + player["scissors_total"]
        )

    def _alive(self, pid: int) -> bool:
        player = self.player_dict[pid]
        return player["stars_total"] > 0 and self._cards_remaining(pid) > 0

    def _can_clear(self, pid: int) -> bool:
        player = self.player_dict[pid]
        return self._cards_remaining(pid) == 0 and player["stars_total"] >= 3

    def _failed_clear(self, pid: int) -> bool:
        player = self.player_dict[pid]
        return self._cards_remaining(pid) == 0 and player["stars_total"] < 3

    def _active_ids(self) -> list[int]:
        return [
            self.agent_name_mapping[agent]
            for agent in self.agents
            if self._alive(self.agent_name_mapping[agent])
        ]

    def _action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(self._action_size, dtype=np.int8)
        pid = self.agent_name_mapping[agent]
        if not self._alive(pid):
            return mask

        player = self.player_dict[pid]
        active_targets = [oid for oid in self._active_ids() if oid != pid]
        if not active_targets:
            return mask

        for target_id in active_targets:
            for card_idx, card in enumerate(CARD_ORDER):
                if player[card.value] > 0:
                    mask[target_id * len(CARD_ORDER) + card_idx] = 1
        return mask

    def _decode_action(self, action: int) -> tuple[int, Card]:
        target_id = action // len(CARD_ORDER)
        card = CARD_ORDER[action % len(CARD_ORDER)]
        return target_id, card

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        pid = self.agent_name_mapping[agent]
        ordered_ids = [pid] + [i for i in range(self.n_players) if i != pid]
        values: list[int] = [self.round_count]

        for oid in ordered_ids:
            player = self.player_dict[oid]
            values.extend(
                [
                    int(self._alive(oid)),
                    player["rock_total"],
                    player["paper_total"],
                    player["scissors_total"],
                    player["stars_total"],
                ]
            )

        return {
            "observation": np.asarray(values, dtype=np.int32),
            "action_mask": self._action_mask(agent),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        elif self.np_random is None:
            self.np_random, self.np_random_seed = seeding.np_random(None)

        self.agents = self.possible_agents[:]
        self.round_count = 0
        self.history = []
        self.last_round_summary = []
        self.player_dict = {
            idx: self._new_player() for idx in range(self.n_players)
        }

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {
            agent: {
                "action_mask": observations[agent]["action_mask"],
                "player_id": self.agent_name_mapping[agent],
            }
            for agent in self.agents
        }
        return observations, infos

    def _build_pairings(
        self, submitted: dict[str, tuple[str, Card]]
    ) -> list[tuple[str, str]]:
        unmatched = list(submitted.keys())
        pairs: list[tuple[str, str]] = []
        used: set[str] = set()

        # 1) mutual selections first
        for agent in unmatched:
            if agent in used:
                continue
            target, _ = submitted[agent]
            if target in used or target not in submitted:
                continue
            target_target, _ = submitted[target]
            if target_target == agent:
                pairs.append((agent, target))
                used.add(agent)
                used.add(target)

        # 2) then one-sided pairings if both are still unmatched
        for agent in unmatched:
            if agent in used:
                continue
            target, _ = submitted[agent]
            if target in used or target not in submitted:
                continue
            pairs.append((agent, target))
            used.add(agent)
            used.add(target)

        return pairs

    def _spend_card(self, pid: int, card: Card) -> None:
        self.player_dict[pid][card.value] -= 1

    def _transfer_star(self, winner: int, loser: int) -> None:
        if self.player_dict[loser]["stars_total"] <= 0:
            return
        self.player_dict[winner]["stars_total"] += 1
        self.player_dict[loser]["stars_total"] -= 1

    def _episode_stats(self) -> dict[str, Any]:
        return {
            "rounds_played": self.round_count,
            "players": {
                agent: {
                    "player_id": self.agent_name_mapping[agent],
                    **self.player_dict[self.agent_name_mapping[agent]],
                    "cards_remaining": self._cards_remaining(
                        self.agent_name_mapping[agent]
                    ),
                }
                for agent in self.possible_agents
            },
        }

    def get_episode_log(self) -> dict[str, Any]:
        return {
            "env_name": self.metadata["name"],
            "n_players": self.n_players,
            "initial_stars": self.initial_stars,
            "initial_budget": dict(self.initial_budget),
            "max_rounds": self.max_rounds,
            "reward_config": asdict(self.reward_config),
            "rounds": self.history,
            "final_state": self._episode_stats(),
        }

    def save_episode(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.get_episode_log(), f, indent=2)
        return path

    def step(self, actions: dict[str, int]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        self.round_count += 1

        rewards = {agent: 0.0 for agent in current_agents}
        terminations = {agent: False for agent in current_agents}
        truncations = {agent: False for agent in current_agents}
        infos = {agent: {} for agent in current_agents}

        submitted: dict[str, tuple[str, Card]] = {}
        round_log: dict[str, Any] = {
            "round": self.round_count,
            "submitted_actions": {},
            "matches": [],
            "invalid_agents": [],
        }
        self.last_round_summary = []

        for agent in current_agents:
            mask = self._action_mask(agent)
            infos[agent]["action_mask"] = mask

            action = actions.get(agent)
            if action is None:
                rewards[agent] += self.reward_config.invalid_action
                infos[agent]["result"] = "missing_action"
                round_log["invalid_agents"].append(
                    {"agent": agent, "reason": "missing_action"}
                )
                self.last_round_summary.append(f"{agent}: missing action")
                continue

            if not self.action_space(agent).contains(action):
                rewards[agent] += self.reward_config.invalid_action
                infos[agent]["result"] = "out_of_bounds_action"
                round_log["invalid_agents"].append(
                    {
                        "agent": agent,
                        "reason": "out_of_bounds_action",
                        "action": int(action),
                    }
                )
                self.last_round_summary.append(
                    f"{agent}: out-of-bounds action {action}"
                )
                continue

            if mask[action] == 0:
                rewards[agent] += self.reward_config.invalid_action
                infos[agent]["result"] = "masked_illegal_action"
                round_log["invalid_agents"].append(
                    {
                        "agent": agent,
                        "reason": "masked_illegal_action",
                        "action": int(action),
                    }
                )
                self.last_round_summary.append(
                    f"{agent}: illegal masked action {action}"
                )
                continue

            target_id, card = self._decode_action(action)
            target_agent = self.id_to_agent[target_id]
            submitted[agent] = (target_agent, card)
            round_log["submitted_actions"][agent] = {
                "action": int(action),
                "target": target_agent,
                "card": str(card),
            }

        pairings = self._build_pairings(submitted)
        paired_agents = {agent for pair in pairings for agent in pair}

        for agent_a, agent_b in pairings:
            pid_a = self.agent_name_mapping[agent_a]
            pid_b = self.agent_name_mapping[agent_b]
            target_a, card_a = submitted[agent_a]
            target_b, card_b = submitted[agent_b]

            self._spend_card(pid_a, card_a)
            self._spend_card(pid_b, card_b)

            outcome = resolve(card_a, card_b)
            if outcome == 1:
                self._transfer_star(pid_a, pid_b)
                rewards[agent_a] += self.reward_config.win_matchup
                rewards[agent_b] += self.reward_config.lose_matchup
                result = f"{agent_a} beat {agent_b}"
            elif outcome == -1:
                self._transfer_star(pid_b, pid_a)
                rewards[agent_a] += self.reward_config.lose_matchup
                rewards[agent_b] += self.reward_config.win_matchup
                result = f"{agent_b} beat {agent_a}"
            else:
                rewards[agent_a] += self.reward_config.tie_matchup
                rewards[agent_b] += self.reward_config.tie_matchup
                result = f"{agent_a} tied {agent_b}"

            match_record = {
                "agents": [agent_a, agent_b],
                "targets": {agent_a: target_a, agent_b: target_b},
                "cards": {agent_a: str(card_a), agent_b: str(card_b)},
                "result": result,
            }
            round_log["matches"].append(match_record)
            self.last_round_summary.append(result)
            infos[agent_a]["result"] = result
            infos[agent_b]["result"] = result

        for agent in current_agents:
            if agent not in submitted:
                continue
            if agent not in paired_agents:
                infos[agent].setdefault("result", "unmatched")
                self.last_round_summary.append(f"{agent}: unmatched")

        for agent in current_agents:
            pid = self.agent_name_mapping[agent]
            if self.player_dict[pid]["stars_total"] <= 0:
                terminations[agent] = True
                rewards[agent] += self.reward_config.eliminated
                infos[agent]["result"] = "eliminated"
            elif self._failed_clear(pid):
                terminations[agent] = True
                rewards[agent] += self.reward_config.failed_clear
                infos[agent]["result"] = "failed_clear"
            elif self._can_clear(pid):
                terminations[agent] = True
                rewards[agent] += self.reward_config.cleared_game
                infos[agent]["result"] = "cleared_game"

        survivors = [
            agent
            for agent in current_agents
            if not terminations[agent]
            and self.player_dict[self.agent_name_mapping[agent]]["stars_total"] > 0
        ]

        if len(survivors) == 1:
            winner = survivors[0]
            rewards[winner] += self.reward_config.last_agent_standing
            infos[winner]["result"] = "last_agent_standing"
            for agent in survivors:
                terminations[agent] = True
            survivors = []

        if self.round_count >= self.max_rounds:
            for agent in current_agents:
                if not terminations[agent]:
                    truncations[agent] = True
                    infos[agent].setdefault("result", "max_rounds")
            survivors = []

        round_log["end_of_round_state"] = {
            agent: {
                "player_id": self.agent_name_mapping[agent],
                **self.player_dict[self.agent_name_mapping[agent]],
                "cards_remaining": self._cards_remaining(
                    self.agent_name_mapping[agent]
                ),
            }
            for agent in self.possible_agents
        }
        self.history.append(round_log)

        self.agents = [
            agent
            for agent in current_agents
            if not terminations[agent] and not truncations[agent]
        ]

        observations = {agent: self.observe(agent) for agent in self.agents}

        episode_stats = self._episode_stats()
        for agent in current_agents:
            infos[agent]["episode_stats"] = episode_stats

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render() without specifying render_mode."
            )
            return

        print(f"\n--- RRPS round {self.round_count} ---")
        for agent in self.possible_agents:
            pid = self.agent_name_mapping[agent]
            player = self.player_dict[pid]
            print(
                f"{agent}: stars={player['stars_total']} "
                f"R={player['rock_total']} P={player['paper_total']} "
                f"S={player['scissors_total']}"
            )
        if self.last_round_summary:
            print("Results:")
            for line in self.last_round_summary:
                print(f"  - {line}")

    def close(self):
        pass
