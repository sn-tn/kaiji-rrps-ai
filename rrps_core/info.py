from typing import TypedDict
from rrps_core.matchup_dict import MatchupDict
from rrps_core.challenge_table import ChallengeTable
from rrps_core.player import PlayerDict
from typing import Literal
from typing import TypeAlias

GameStatus: TypeAlias = Literal["victory", "eliminated", "playing"]

class Info(TypedDict):
    initial_alive_player_dict: PlayerDict
    matchup_dict: MatchupDict
    challenge_table: ChallengeTable
    alive_player_dict: PlayerDict
    round_number: int
    game_status: GameStatus
