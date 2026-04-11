from __future__ import annotations
from typing import TypedDict
from typing import TypeAlias

class Player(TypedDict):
    rock_total: int
    paper_total: int
    scissors_total: int
    stars_total: int
    position: tuple | None

PlayerID: TypeAlias = int

PlayerDict: TypeAlias = dict[PlayerID, Player]
