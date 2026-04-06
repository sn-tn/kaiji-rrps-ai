import random
from dataclasses import dataclass
from typing import TypedDict
import gymnasium as gym
from gymnasium import spaces
from environment.player import Player
from environment.move import Move


class BudgetObs(TypedDict):
    rock:     int
    paper:    int
    scissors: int


class PlayerObs(TypedDict):
    stars:    int
    budget:   BudgetObs
    position: tuple[int, int]


class Observation(TypedDict):
    agent:           PlayerObs
    opponent:        PlayerObs
    opponents_alive: int


@dataclass
class RewardConfig:
    win_matchup: float = 1.0
    lose_matchup: float = -1.0
    tie_matchup: float = 0.0
    eliminated: float = -3.0
    victory: float = 5.0


# ── helpers ────────────────────────────────────────────────────────────────────────────


def resolve(m1: Move, m2: Move) -> int:
    """Returns 1 if m1 wins, -1 if m2 wins, 0 on tie."""
    if not (isinstance(m1, Move) and isinstance(m2, Move)):
        raise TypeError("Arguments must be valid Move values")

    wins_against = {
        Move.ROCK: Move.SCISSORS,
        Move.PAPER: Move.ROCK,
        Move.SCISSORS: Move.PAPER,
    }
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev (chessboard) distance between two grid positions."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


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

    # observation bounds
    _MAX_STARS = 20
    _MAX_BUDGET = 10
    _MAX_OPS = 20

    # action constants
    _MOVE_ACTIONS = {
        0: (0, -1),  # up
        1: (0,  1),  # down
        2: (-1, 0),  # left
        3: (1,  0),  # right
    }
    _RPS_ACTIONS = {4: Move.ROCK, 5: Move.PAPER, 6: Move.SCISSORS}

    def __init__(
        self,
        n_opponents: int = 3,
        stars: int = 3,
        budget: int = 4,
        grid_size: int = 20,
        challenge_radius: int = 1,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.n_opponents = n_opponents
        self.initial_stars = stars
        self.initial_budget = budget
        self.grid_size = grid_size
        self.challenge_radius = challenge_radius
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        # action space: 4 move directions + 3 RPS moves
        self.action_space = spaces.Discrete(7)

        # observation space: nested dict
        g = grid_size - 1
        player_space = spaces.Dict({
            "stars":    spaces.Discrete(self._MAX_STARS + 1),
            "budget":   spaces.Dict({
                "rock":     spaces.Discrete(self._MAX_BUDGET + 1),
                "paper":    spaces.Discrete(self._MAX_BUDGET + 1),
                "scissors": spaces.Discrete(self._MAX_BUDGET + 1),
            }),
            "position": spaces.Tuple((spaces.Discrete(g + 1), spaces.Discrete(g + 1))),
        })
        self.observation_space = spaces.Dict({
            "agent":           player_space,
            "opponent":        player_space,
            "opponents_alive": spaces.Discrete(self._MAX_OPS + 1),
        })

        self._agent: Player | None = None
        self._opponents: list[Player] = []

    # ── private helpers ───────────────────────────────────────────────────────────────────

    def _random_position(self) -> tuple[int, int]:
        return (
            int(self.np_random.integers(0, self.grid_size)),
            int(self.np_random.integers(0, self.grid_size)),
        )

    def _make_players(self):
        self._agent = Player(
            player_id=0,
            stars=self.initial_stars,
            budget=self.initial_budget,
            position=self._random_position(),
        )
        self._opponents = [
            Player(
                player_id=i + 1,
                stars=self.initial_stars,
                budget=self.initial_budget,
                position=self._random_position(),
            )
            for i in range(self.n_opponents)
        ]

    def _alive_opponents(self) -> list[Player]:
        return [p for p in self._opponents if p.is_alive()]

    def _in_range(self, a: Player, b: Player) -> bool:
        return chebyshev(a.position, b.position) <= self.challenge_radius

    def _clamp_move(self, pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        x = max(0, min(self.grid_size - 1, pos[0] + delta[0]))
        y = max(0, min(self.grid_size - 1, pos[1] + delta[1]))
        return (x, y)

    def _player_obs(self, p: Player) -> PlayerObs:
        """Build the observation dict for a single player."""
        return {
            "stars":    p.stars,
            "budget":   {
                "rock":     p.budget[Move.ROCK],
                "paper":    p.budget[Move.PAPER],
                "scissors": p.budget[Move.SCISSORS],
            },
            "position": p.position,
        }

    def _null_opponent_obs(self) -> PlayerObs:
        """Zero-filled opponent obs returned when no opponents remain."""
        return {
            "stars":    0,
            "budget":   {"rock": 0, "paper": 0, "scissors": 0},
            "position": (0, 0),
        }

    def _get_obs(self) -> Observation:
        ag = self._agent
        alive = self._alive_opponents()
        if alive:
            in_range = [op for op in alive if self._in_range(ag, op)]
            op = random.choice(in_range) if in_range else random.choice(alive)
            opponent_obs = self._player_obs(op)
        else:
            opponent_obs = self._null_opponent_obs()

        return {
            "agent":           self._player_obs(ag),
            "opponent":        opponent_obs,
            "opponents_alive": len(alive),
        }

    def _run_opponent_matchups(self):
        """Pair up alive opponents and resolve fights; unpaired opponents move randomly."""
        alive = self._alive_opponents()
        random.shuffle(alive)
        paired: set[int] = set()

        for p in alive:
            if p.id in paired or not p.is_alive():
                continue

            in_range = [q for q in alive if q is not p and self._in_range(p, q)]
            if not in_range:
                # no one nearby -- move randomly
                delta = random.choice(list(self._MOVE_ACTIONS.values()))
                p.position = self._clamp_move(p.position, delta)
                continue

            accepted, _, op = p.select_opponent(in_range)
            if accepted and op is not None and op.id not in paired:
                m1 = random.choice(p.available_moves())
                m2 = random.choice(op.available_moves())
                p.use_move(m1)
                op.use_move(m2)
                result = resolve(m1, m2)
                if result == 1:
                    p.steal_life(op)
                elif result == -1:
                    op.steal_life(p)
                paired.update({p.id, op.id})
            else:
                # challenge refused -- move randomly
                delta = random.choice(list(self._MOVE_ACTIONS.values()))
                p.position = self._clamp_move(p.position, delta)

    # ── gym API ───────────────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._make_players()
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        info = {}

        if self._agent.is_alive():
            if action in self._MOVE_ACTIONS:
                # ── movement ────────────────────────────────────────────────────────────
                self._agent.position = self._clamp_move(
                    self._agent.position, self._MOVE_ACTIONS[action]
                )
            else:
                # ── RPS matchup ──────────────────────────────────────────────────────────
                move = self._RPS_ACTIONS[action]
                in_range = [
                    op for op in self._alive_opponents()
                    if self._in_range(self._agent, op)
                ]

                if in_range and move in self._agent.available_moves():
                    op = random.choice(in_range)
                    if op.accept_opponent(self._agent):
                        op_move = random.choice(op.available_moves())
                        self._agent.use_move(move)
                        op.use_move(op_move)

                        result = resolve(move, op_move)
                        # REWARD: agent wins the matchup -- steals a star from opponent
                        if result == 1:
                            self._agent.steal_life(op)
                            reward = self.reward_config.win_matchup
                        # REWARD: agent loses the matchup -- opponent steals a star
                        elif result == -1:
                            op.steal_life(self._agent)
                            reward = self.reward_config.lose_matchup
                        # REWARD: tie -- no stars change hands
                        else:
                            reward = self.reward_config.tie_matchup

        # ── opponent matchups ──────────────────────────────────────────────────────────────────
        self._run_opponent_matchups()

        # ── termination check ──────────────────────────────────────────────────────────────────
        alive = [p for p in [self._agent] + self._opponents if p.is_alive()]

        # REWARD: agent has run out of stars or moves -- eliminated from tournament
        if not self._agent.is_alive():
            reward += self.reward_config.eliminated
            terminated = True
            info["result"] = "eliminated"
        # REWARD: agent is the last player standing -- tournament won
        elif len(alive) == 1:
            reward += self.reward_config.victory
            terminated = True
            info["result"] = "victory"

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        ag = self._agent
        print(
            f"[Agent] pos={ag.position} stars={ag.stars}"
            f" budget=R{ag.budget[Move.ROCK]}/P{ag.budget[Move.PAPER]}/S{ag.budget[Move.SCISSORS]}"
            f" | Alive opponents: {len(self._alive_opponents())}"
        )

    def close(self):
        pass
