from environment_core.player import Player
from environment_core.move import Card


class MatchupTable:
    """Tracks all challenges declared during the challenge phase.

    challenges: dict mapping challenger → (card, target)
    Cleared each challenge phase via reset().
    """

    def __init__(self):
        self.challenges: dict[Player, tuple[Card, Player]] = {}

    def challenge(
        self, challenger: Player, card: Card, challenged: Player
    ) -> None:
        """Record that challenger declared a challenge to challenged with card."""
        self.challenges[challenger] = (card, challenged)

    def get_incoming(self, player: Player) -> list[tuple[Player, Card]]:
        """Return (challenger, card) for all incoming challenges to this player."""
        return [
            (c, card)
            for c, (card, t) in self.challenges.items()
            if t is player
        ]

    def get_outgoing(self, player: Player) -> tuple[Card, Player] | None:
        """Return (card, target) for this player's outgoing challenge, or None."""
        return self.challenges.get(player)

    def is_challenger(self, player: Player) -> bool:
        """Return True if this player declared a challenge this phase."""
        return player in self.challenges

    def has_incoming(self, player: Player) -> bool:
        """Return True if this player has at least one incoming challenge."""
        return any(t is player for _, t in self.challenges.values())

    def reset(self) -> None:
        """Clear all challenges for the next challenge phase."""
        self.challenges = {}
