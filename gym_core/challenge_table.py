from __future__ import annotations
from typing import TypeAlias
from gym_core.player import PlayerID
from gym_core.cards import Card
import pandera as pa
from pandera.typing import DataFrame, Series

ChallengerId: TypeAlias = PlayerID
TargetPlayerId: TypeAlias = PlayerID
ChallengeCard: TypeAlias = Card


ChallengeTable: TypeAlias = dict[
    ChallengerId, tuple[ChallengeCard, list[TargetPlayerId]]
]


class ChallengeSchema(pa.DataFrameModel):
    player_id: Series[int] # its a series because its rows not a list
    card: Series[str]
    target_id: Series[int]


ChallengeTable: TypeAlias = DataFrame[ChallengeSchema]
