from dataclasses import dataclass


@dataclass
class RewardConfig:
    win_matchup: float = 100
    lose_matchup: float = -100
    tie_matchup: float = 0
    eliminated: float = -2000
    victory: float = 2000
    invalid_move: float = -10
    within_challenge_range: float = 0
    approach_opponent: float = 0
