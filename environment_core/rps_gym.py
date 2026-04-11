from dataclasses import dataclass
from typing import TypedDict
import gymnasium as gym
from gymnasium import spaces
from environment_core.player import Player, BasicPlayer, AgentPlayer
from environment_core.move import Card, chebyshev
from environment_core.matchup_table import MatchupTable


class BudgetObs(TypedDict):
    rock: int
    paper: int
    scissors: int


class PlayerObs(TypedDict):
    player_id: int
    stars: int
    budget: BudgetObs
    position: tuple[int, int]


class Observation(TypedDict):
    agent: PlayerObs
    opponent: PlayerObs
    opponents_alive: int


@dataclass
class RewardConfig:
    victory: float = 500
    loss: float = -500
    invalid_move: float = -10


# ── helpers ────────────────────────────────────────────────────────────────────────────


def resolve(m1: Card, m2: Card) -> int:
    """Returns 1 if m1 wins, -1 if m2 wins, 0 on tie."""
    if not (isinstance(m1, Card) and isinstance(m2, Card)):
        raise TypeError("Arguments must be valid Move values")

    wins_against = {
        Card.ROCK: Card.SCISSORS,
        Card.PAPER: Card.ROCK,
        Card.SCISSORS: Card.PAPER,
    }
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


# ── environment ───────────────────────────────────────────────────────────────────


class RestrictedRPSEnv(gym.Env):
    """
    A single-agent Gymnasium environment for Restricted Rock Paper Scissors.

    The *agent* controls player 0; all other players act randomly on a 2D grid.
    Players can only challenge opponents within challenge_radius squares (Chebyshev
    distance). Opponents will move randomly when no target is in range.

    Observation (Dict):
        agent: Dict
            stars:    int   -- current star count
            budget:   Dict  -- {rock, paper, scissors} remaining counts
            position: Tuple -- (x, y) grid position
        opponent: Dict  (nearest in-range opponent; zeros if none)
            stars:    int
            budget:   Dict  -- {rock, paper, scissors}
            position: Tuple -- (x, y)
        opponents_alive: int -- total surviving opponents

    Action (Discrete 7):
        0 = Move Up    (y - 1)
        1 = Move Down  (y + 1)
        2 = Move Left  (x - 1)
        3 = Move Right (x + 1)
        4 = ROCK
        5 = PAPER
        6 = SCISSORS

    Reward (defaults, override via RewardConfig):
        +1   win a matchup (steal opponent's star)
        -1   lose a matchup
         0   tie, move action, or no opponent in range
        +5   win the tournament
        -3   eliminated

    Args:
        n_opponents:      number of opponents in the tournament (default 3)
        stars:            starting stars per player (default 3)
        budget:           starting count of each move per player (default 4)
        grid_size:        side length of the square grid (default 20)
        challenge_radius: Chebyshev distance within which players can fight (default 1)
        render_mode:      "human" to print state each step, or None
        reward_config:    RewardConfig instance to customise reward values

    Episode ends when the tournament is over (one player left standing).
    """

    metadata = {"render_modes": ["human"]}

    # action constants
    _MOVE_ACTIONS = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }
    _RPS_ACTIONS = {4: Card.ROCK, 5: Card.PAPER, 6: Card.SCISSORS}

    def __init__(
        self,
        opponents: list[Player],
        stars: int = 3,
        budget: int = 4,
        grid_size: int = 20,
        challenge_radius: int = 1,
        max_turns: int = 2000,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self._opponents = opponents
        self.n_opponents = len(opponents)
        self.initial_stars = stars
        self.initial_budget = budget
        self.grid_size = grid_size
        self.challenge_radius = challenge_radius
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        # observation bounds derived from game parameters
        # max stars: agent wins all opponents' stars
        max_stars = stars + self.n_opponents * stars
        # budget never increases, so initial value is the max
        max_budget = budget

        # action space: 4 move directions + 3 RPS moves
        self.action_space = spaces.Discrete(7)
        # observation space: nested dict
        g = grid_size - 1
        player_space = spaces.Dict(
            {
                "player_id": spaces.Discrete(self.n_opponents + 1),
                "stars": spaces.Discrete(max_stars + 1),
                "budget": spaces.Dict(
                    {
                        "rock": spaces.Discrete(max_budget + 1),
                        "paper": spaces.Discrete(max_budget + 1),
                        "scissors": spaces.Discrete(max_budget + 1),
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
                "opponents_alive": spaces.Discrete(self.n_opponents + 1),
            }
        )

        self._agent: Player | None = None

    # ── private helpers ───────────────────────────────────────────────────────────────────

    def _random_position(self) -> tuple[int, int]:
        return (
            int(self.np_random.integers(0, self.grid_size)),
            int(self.np_random.integers(0, self.grid_size)),
        )

    def _intialize_players(self):
        self._agent = AgentPlayer(
            player_id=0,
            stars=self.initial_stars,
            budget=self.initial_budget,
            position=self._random_position(),
        )
        for op in self._opponents:
            op.position = self._random_position()
            op.stars = self.initial_stars
            op.budget = {Card.ROCK: self.initial_budget, Card.PAPER: self.initial_budget, Card.SCISSORS: self.initial_budget}
        self.matchup_table = MatchupTable([*self._opponents, self._agent])

    def _alive_opponents(self) -> list[Player]:
        return [p for p in self._opponents if p.is_alive()]

    def _in_range(self, a: Player, b: Player) -> bool:
        return chebyshev(a.position, b.position) <= self.challenge_radius

    def _can_move(self, pos: tuple[int, int], delta: tuple[int, int]) -> bool:
        """Returns True if applying delta keeps the position within the grid."""
        x = pos[0] + delta[0]
        y = pos[1] + delta[1]
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _player_obs(self, p: Player) -> PlayerObs:
        """Build the observation dict for a single player."""
        return {
            "player_id": p.id,
            "stars": p.stars,
            "budget": {
                "rock": p.budget[Card.ROCK],
                "paper": p.budget[Card.PAPER],
                "scissors": p.budget[Card.SCISSORS],
            },
            "position": p.position,
        }

    def _null_opponent_obs(self) -> PlayerObs:
        """Zero-filled opponent obs returned when no opponents remain."""
        return {
            "player_id": 0,
            "stars": 0,
            "budget": {"rock": 0, "paper": 0, "scissors": 0},
            "position": (0, 0),
        }

    def _get_obs(self) -> Observation:
        ag = self._agent
        alive = self._alive_opponents()
        if alive:
            in_range = [op for op in alive if self._in_range(ag, op)]
            op = min(
                in_range or alive,
                key=lambda p: chebyshev(ag.position, p.position),
            )
            opponent_obs = self._player_obs(op)
        else:
            opponent_obs = self._null_opponent_obs()

        return {
            "agent": self._player_obs(ag),
            "opponent": opponent_obs,
            "opponents_alive": len(alive),
        }

    # ── gym API ───────────────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._intialize_players()
        self._turn = 0
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        info = {}
        self._turn += 1
        paired: set[int] = set()

        all_alive = [
            p for p in [self._agent] + self._opponents if p.is_alive()
        ]
        needs_move: list[Player] = []
        # ── Phase 1: opponents declare challenges (can target agent) ──────────────

        # reset matchup table
        self.matchup_table.reset()
        # get all alive opponents
        alive = self._alive_opponents()
        indices = list(range(len(alive)))
        self.np_random.shuffle(indices)

        for i in indices:
            p = alive[i]
            if not p.is_alive() or not p.has_cards():
                continue
            in_range = [
                q for q in all_alive if q is not p and self._in_range(p, q)
            ]
            if not in_range:
                needs_move.append(p)
                continue
            target = p.challenge_opponent(in_range)
            self.matchup_table.challenge(p, p.select_card(target), target)

        # ── Phase 2: agent decides ────────────────────────────────────────────────
        if self._agent.is_alive():
            if action in self._MOVE_ACTIONS:
                dx, dy = self._MOVE_ACTIONS[action]
                if self._can_move(self._agent.position, (dx, dy)):
                    self._agent.move(
                        self._agent.position[0] + dx,
                        self._agent.position[1] + dy,
                    )
                else:
                    reward += self.reward_config.invalid_move
            else:
                agent_card = self._RPS_ACTIONS[action]
                in_range = [
                    op
                    for op in self._alive_opponents()
                    if self._in_range(self._agent, op)
                ]
                if (
                    not in_range
                    or agent_card not in self._agent.available_cards()
                ):
                    reward += self.reward_config.invalid_move
                else:
                    target = min(
                        in_range,
                        key=lambda p: chebyshev(
                            self._agent.position, p.position
                        ),
                    )
                    # Accept if target already challenged the agent; otherwise initiate
                    incoming = self.matchup_table.get_incoming(self._agent)
                    challenged_by = next(
                        ((c, card) for c, card in incoming if c is target),
                        None,
                    )
                    if challenged_by:
                        challenger, challenger_card = challenged_by
                        self._agent.use_card(agent_card)
                        challenger.use_card(challenger_card)
                        result = resolve(challenger_card, agent_card)
                        if result == 1:
                            challenger.steal_star(self._agent)
                        elif result == -1:
                            self._agent.steal_star(challenger)
                        paired.update({self._agent.id, challenger.id})
                        info["player_card"] = agent_card
                        info["opponent_card"] = challenger_card
                    else:
                        self.matchup_table.challenge(
                            self._agent, agent_card, target
                        )

        # ── Phase 3: opponents accept or reject their incoming challenges ─────────

        for p in alive:
            if p.id in paired or not p.is_alive():
                continue
            incoming = self.matchup_table.get_incoming(p)
            if not incoming:
                needs_move.append(p)
                continue
            accepted = False
            for challenger, challenger_card in incoming:
                if (
                    not challenger.is_alive()
                    or challenger.id in paired
                    or p.id in paired
                ):
                    continue
                if p.accept_challenge(challenger):
                    p_card = p.select_card(challenger)
                    p.use_card(p_card)
                    challenger.use_card(challenger_card)
                    result = resolve(challenger_card, p_card)
                    if result == 1:
                        challenger.steal_star(p)
                    elif result == -1:
                        p.steal_star(challenger)
                    paired.update({p.id, challenger.id})
                    accepted = True
                    break
            if not accepted:
                needs_move.append(p)

        # ─── Movement for opponents who didn't fight ──────────────────────────────
        all_alive = [
            p for p in [self._agent] + self._opponents if p.is_alive()
        ]
        for p in needs_move:
            if not p.is_alive() or p.id in paired:
                continue
            delta = p.select_direction([q for q in all_alive if q is not p])
            if self._can_move(p.position, delta.value):
                p.position = (
                    p.position[0] + delta.value[0],
                    p.position[1] + delta.value[1],
                )

        # ── termination check ──────────────────────────────────────────────────────────────────
        total_alive_players = [
            p for p in [self._agent] + self._opponents if p.is_alive()
        ]

        # truncated is when the match is over based on defined max turns
        truncated = (
            self.max_turns is not None
            and self._turn >= self.max_turns
            and not terminated
        )

        ## round ends before truncation awards
        # REWARD: agent has run out of stars or moves -- eliminated from tournament

        if not self._agent.is_alive():
            reward += self.reward_config.loss
            terminated = True
            info["result"] = "eliminated"
        # REWARD: agent is the last player standing -- tournament won
        elif len(total_alive_players) == 1:
            reward += self.reward_config.victory
            terminated = True
            info["result"] = "victory"
        elif not self._agent.has_cards() and self._agent.stars > 3:
            reward += self.reward_config.victory
            terminated = True
            info["result"] = "victory"
        ## round ends due to truncations (max turns hit)
        if truncated:
            if self._agent.has_cards() or self._agent.stars < 3:
                reward += self.reward_config.loss
                info["result"] = "eliminated"
            else:
                reward += self.reward_config.victory
                info["result"] = "victory"

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        ag = self._agent
        print(
            f"[Agent] pos={ag.position} stars={ag.stars}"
            f" budget=R{ag.budget[Card.ROCK]}/P{ag.budget[Card.PAPER]}/S{ag.budget[Card.SCISSORS]}"
            f" | Alive opponents: {len(self._alive_opponents())}"
        )

    def close(self):
        pass
