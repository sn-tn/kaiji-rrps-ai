from __future__ import annotations
from typing import TypeAlias
from rrps_core.player import PlayerID
from rrps_core.cards import Card

MatchupPair: TypeAlias = tuple[PlayerID, PlayerID]
CardPair: TypeAlias = tuple[Card, Card]

MatchupDict: TypeAlias = dict[MatchupPair, CardPair]
