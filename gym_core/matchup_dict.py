from __future__ import annotations
from typing import TypeAlias
from gym_core.player import PlayerID
from gym_core.cards import Card

MatchupPair: TypeAlias = tuple[PlayerID, PlayerID]
CardPair: TypeAlias = tuple[Card, Card]

MatchupDict: TypeAlias = dict[MatchupPair, CardPair]
