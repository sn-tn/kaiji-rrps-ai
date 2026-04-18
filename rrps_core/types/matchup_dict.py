from __future__ import annotations
from typing import TypeAlias
from rrps_core.types.player import PlayerID
from rrps_core.types.cards import Card

MatchupPair: TypeAlias = tuple[PlayerID, PlayerID]
CardPair: TypeAlias = tuple[Card, Card]

MatchupDict: TypeAlias = dict[MatchupPair, CardPair]
