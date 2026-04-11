from typing import TypedDict
from gym_core.player import PlayerDict

class Observation(TypedDict):
    """player 0 is always the agent"""
    player_dict: PlayerDict
    turn: int
