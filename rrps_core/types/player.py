from __future__ import annotations
from typing import TypedDict
from typing import TypeAlias
import numpy as np

class Budget(TypedDict):
    rock_total: int
    paper_total: int
    scissors_total: int


class Player(Budget):
    stars_total: int
    position: np.ndarray


PlayerID: TypeAlias = int

PlayerDict: TypeAlias = dict[PlayerID, Player]
