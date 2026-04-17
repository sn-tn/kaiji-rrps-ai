from typing import TypedDict
from rrps_core.player import PlayerID, PlayerDict


class Observation(TypedDict):
    """player 0 is always the agent"""

    player_dict: PlayerDict
    turn: int


class BudgetObs(TypedDict):
    rock_total: int
    paper_total: int
    scissors_total: int


class PlayerObs(TypedDict):
    player_id: PlayerID
    stars_total: int
    budget: BudgetObs
    position: tuple[int, int]
