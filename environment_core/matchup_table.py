from environment_core.player import Player
from environment_core.move import Move


class MatchupTable:
    """Tracks all challenges declared during the challenge phase.

    challenges: flat list of (challenger, card, challenged)
    Cleared each challenge phase via reset().
    """

    def __init__(self, players: list[Player]):
        self.challenges: list[tuple[Player, Move, Player]] = []

    def challenge(self, challenger: Player, card: Move, challenged: Player) -> None:
        """Record that challenger declared a challenge to challenged with card."""
        self.challenges.append((challenger, card, challenged))

    def get_incoming(self, player: Player) -> list[tuple[Player, Move]]:
        """Return (challenger, card) for all incoming challenges to this player."""
        return [(c, card) for c, card, t in self.challenges if t is player]

    def get_outgoing(self, player: Player) -> tuple[Player, Move] | None:
        """Return (challenged, card) for this player's outgoing challenge, or None."""
        for c, card, t in self.challenges:
            if c is player:
                return t, card
        return None

    def is_challenger(self, player: Player) -> bool:
        """Return True if this player declared a challenge this phase."""
        return any(c is player for c, _, _ in self.challenges)

    def has_incoming(self, player: Player) -> bool:
        """Return True if this player has at least one incoming challenge."""
        return any(t is player for _, _, t in self.challenges)

    def reset(self) -> None:
        """Clear all challenges for the next challenge phase."""
        self.challenges = []
