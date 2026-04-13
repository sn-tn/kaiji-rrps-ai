from typing import TypedDict
from gym_core.matchup_dict import MatchupDict
from gym_core.challenge_table import ChallengeTable
from gym_core.player import PlayerDict


class Info(TypedDict):
    initial_alive_player_dict: PlayerDict
    matchup_dict: MatchupDict
    challenge_table: ChallengeTable
    alive_player_dict: PlayerDict
    round_number: int
