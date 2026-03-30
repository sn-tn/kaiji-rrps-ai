import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from player import Player
from move import Move


# ── helpers ──────────────────────────────────────────────────────────────────

def resolve(m1: Move, m2: Move) -> int:
    """Returns 1 if m1 wins, -1 if m2 wins, 0 on tie."""
    if not (isinstance(m1, Move) and isinstance(m2, Move)):
        raise TypeError("Arguments must be valid Move values")

    wins_against = {Move.ROCK: Move.SCISSORS, Move.PAPER: Move.ROCK, Move.SCISSORS: Move.PAPER}
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


# ── environment ───────────────────────────────────────────────────────────────

class RestrictedRPSEnv(gym.Env):
    """
    A single-agent Gymnasium environment for Restricted Rock Paper Scissors.

    The *agent* controls player 0; all other players act randomly.

    Observation (Box, float32):
        [agent_lives,
         agent_rock_budget, agent_paper_budget, agent_scissors_budget,
         opponent_lives,
         opponent_rock_budget, opponent_paper_budget, opponent_scissors_budget,
         num_alive_opponents]          ← 9 values total

    Action (Discrete 3):
        0 = ROCK, 1 = PAPER, 2 = SCISSORS

    Reward:
        +1   win a matchup (steal opponent's life)
        -1   lose a matchup
         0   tie or no matchup this step (agent had no opponent)
        +5   win the tournament
        -3   eliminated

    Episode ends when the tournament is over (one player left standing).
    """

    metadata = {"render_modes": ["human"]}

    # observation bounds
    _MAX_LIVES  = 20
    _MAX_BUDGET = 10
    _MAX_OPS    = 20

    def __init__(
        self,
        n_opponents: int = 3,
        lives: int = 3,
        budget: int = 4,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.n_opponents = n_opponents
        self.initial_lives = lives
        self.initial_budget = budget
        self.render_mode = render_mode

        # action space: choose ROCK / PAPER / SCISSORS
        self.action_space = spaces.Discrete(3)

        # observation space
        high = np.array(
            [self._MAX_LIVES] +
            [self._MAX_BUDGET] * 3 +
            [self._MAX_LIVES] +
            [self._MAX_BUDGET] * 3 +
            [self._MAX_OPS],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)

        self._agent: Player | None = None
        self._opponents: list[Player] = []

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_players(self):
        self._agent = Player(player_id=0, lives=self.initial_lives, budget=self.initial_budget)
        self._opponents = [
            Player(player_id=i + 1, lives=self.initial_lives, budget=self.initial_budget)
            for i in range(self.n_opponents)
        ]

    def _alive_opponents(self) -> list[Player]:
        return [p for p in self._opponents if p.is_alive()]

    def _opponent_obs(self) -> np.ndarray:
        """
        Returns the observation slice for one opponent.
        If no opponents remain, returns zeros.
        """
        alive = self._alive_opponents()
        if not alive:
            return np.zeros(4, dtype=np.float32)
        op = random.choice(alive)   # focus on a random live opponent
        return np.array(
            [op.lives,
             op.budget[Move.ROCK],
             op.budget[Move.PAPER],
             op.budget[Move.SCISSORS]],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        ag = self._agent
        return np.array(
            [ag.lives,
             ag.budget[Move.ROCK],
             ag.budget[Move.PAPER],
             ag.budget[Move.SCISSORS],
             *self._opponent_obs(),
             len(self._alive_opponents())],
            dtype=np.float32,
        ).clip(self.observation_space.low, self.observation_space.high)

    def _run_opponent_matchups(self):
        """Pair up alive opponents (excluding agent) and resolve their fights."""
        alive = self._alive_opponents()
        random.shuffle(alive)
        paired: set[int] = set()

        for p in alive:
            if p.id in paired or not p.is_alive():
                continue
            accepted, _, op = p.select_opponent(alive)
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

    # ── gym API ───────────────────────────────────────────────────────────────

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

        move = list(Move)[action]
        reward = 0.0
        terminated = False
        info = {}

        # ── agent's matchup ──────────────────────────────────────────────────
        if self._agent.is_alive():
            alive_ops = self._alive_opponents()

            if alive_ops and move in self._agent.available_moves():
                op = random.choice(alive_ops)
                if op.accept_opponent(self._agent):
                    agent_move = move
                    op_move = random.choice(op.available_moves())

                    self._agent.use_move(agent_move)
                    op.use_move(op_move)

                    result = resolve(agent_move, op_move)
                    if result == 1:
                        self._agent.steal_life(op)
                        reward = 1.0
                    elif result == -1:
                        op.steal_life(self._agent)
                        reward = -1.0
                    # tie → reward stays 0
            # else: action unavailable or no opponents → no matchup, reward 0

        # ── opponent matchups ────────────────────────────────────────────────
        self._run_opponent_matchups()

        # ── termination check ────────────────────────────────────────────────
        alive = [p for p in [self._agent] + self._opponents if p.is_alive()]

        if not self._agent.is_alive():
            reward += -3.0
            terminated = True
            info["result"] = "eliminated"
        elif len(alive) == 1:
            reward += 5.0
            terminated = True
            info["result"] = "victory"

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        ag = self._agent
        print(
            f"[Agent] lives={ag.lives} "
            f"budget=R{ag.budget[Move.ROCK]}/P{ag.budget[Move.PAPER]}/S{ag.budget[Move.SCISSORS]} | "
            f"Alive opponents: {len(self._alive_opponents())}"
        )

    def close(self):
        pass