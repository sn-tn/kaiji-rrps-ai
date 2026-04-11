from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum
import numpy

from environment_core.move import Move, Direction, chebyshev


class PlayerType(Enum):
    """Enumeration of supported player behavior types."""

    AGENT = "Agent"
    RANDOM = "Random"
    AGGRESSIVE = "Aggressive"
    CONSERVATIVE = "Conservative"


class Player(ABC):
    """Abstract base class representing a player in the tournament.

    Each player has:
        - a unique player ID
        - a star count
        - a move budget for rock, paper, and scissors
        - a board position
    """

    rng = numpy.random.default_rng()

    def __init__(
        self,
        player_id: int,
        stars: int = 3,
        budget: int = 4,
        position: tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize a player.

        Args:
            player_id: Unique identifier for the player.
            stars: Starting number of stars.
            budget: Starting count for each move type.
            position: Starting board position as (x, y).
        """
        self.id = player_id
        self.stars = stars
        self.budget = {
            Move.ROCK: budget,
            Move.PAPER: budget,
            Move.SCISSORS: budget,
        }
        self.position = position

    def has_cards(self):
        return len(self.available_cards()) > 0

    def move(self, x: int, y: int) -> tuple[int, int]:
        self.position = (x, y)
        return (x, y)

    def available_cards(self) -> list[Move]:
        """Return the list of moves the player can still use.

        Returns:
            A list of moves with remaining budget greater than zero.
        """
        return [move for move, count in self.budget.items() if count > 0]

    def is_alive(self) -> bool:
        """Return whether the player is still active.

        A player is considered alive if they have at least one star

        Returns:
            True if the player is alive, otherwise False.
        """
        return self.stars > 0

    def use_card(self, move: Move) -> None:
        """Consume one unit of budget for the given move.

        Args:
            move: The move being used.
        """
        self.budget[move] -= 1

    def steal_star(self, other: "Player") -> None:
        """Take one star from another player.

        Args:
            other: The player losing a star.
        """
        self.stars += 1
        other.lose_life()

    def lose_life(self) -> None:
        """Reduce this player's star count by one."""
        self.stars -= 1

    def _toward(self, target: "Player") -> Direction:
        """Return the direction that moves this player toward a target.

        Args:
            target: The player to move toward.

        Returns:
            The direction that best reduces the distance to the target.
        """
        dx = numpy.sign(target.position[0] - self.position[0])
        dy = numpy.sign(target.position[1] - self.position[1])

        best_score = min(
            abs(d.value[0] - dx) + abs(d.value[1] - dy) for d in Direction
        )
        best = [
            d for d in Direction
            if abs(d.value[0] - dx) + abs(d.value[1] - dy) == best_score
        ]
        return random.choice(best)

    def has_cards(self):
        return len(self.available_cards()) > 0

    @abstractmethod
    def select_card(self, opponent: "Player" | None = None) -> Move:
        """Choose a move to play.

        Args:
            opponent: The opposing player, if applicable.

        Returns:
            The selected move.
        """
        ...

    @abstractmethod
    def challenge_opponent(
        self,
        available_opponents: list["Player"],
        alive_players: list["Player"],
    ) -> tuple[bool, list["Player"], "Player" | None]:
        """Choose an opponent to challenge.

        Args:
            available_opponents: Players currently in range to challenge.
            alive_players: All players still active in the round.

        Returns:
            A tuple containing:
                - whether the challenge was accepted
                - the updated list of alive players excluding matched players
                - the selected opponent, or None if no match occurred
        """
        ...

    @abstractmethod
    def accept_challenge(self, opponent: "Player") -> bool:
        """Decide whether to accept a challenge.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            True if the challenge is accepted, otherwise False.
        """
        ...

    @abstractmethod
    def get_playertype(self) -> PlayerType:
        """Return this player's behavior type.

        Returns:
            The corresponding PlayerType enum value.
        """
        ...

    @abstractmethod
    def select_direction(self, alive_players: list["Player"]) -> Direction:
        """Choose a movement direction.

        Args:
            alive_players: All currently active players.

        Returns:
            The selected movement direction.
        """
        ...


class AgentPlayer(Player):
    """Player controlled externally by the environment.

    Opponent selection and movement are not handled here.
    The agent always accepts challenges.
    """

    def select_card(self, opponent: Player | None = None) -> Move:
        """Raise an error because move selection is environment-controlled.

        Args:
            opponent: The opposing player, if applicable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "AgentPlayer moves are controlled by the environment"
        )

    def challenge_opponent(
        self,
        available_opponents: list[Player],
        alive_players: list[Player],
    ) -> tuple[bool, list[Player], Player | None]:
        """Raise an error because opponent selection is environment-controlled.

        Args:
            available_opponents: Players currently in challenge range.
            alive_players: All players still active in the round.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "AgentPlayer challenges are controlled by the environment"
        )

    def accept_challenge(self, opponent: Player) -> bool:
        """Always accept a challenge.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            Always True.
        """
        if not self.has_cards():
            return False
        return True

    def get_playertype(self) -> PlayerType:
        """Return the player type.

        Returns:
            PlayerType.AGENT.
        """
        return PlayerType.AGENT

    def select_direction(self, alive_players: list[Player]) -> Direction:
        """Raise an error because movement is environment-controlled.

        Args:
            alive_players: All currently active players.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "AgentPlayer moves are controlled by the environment"
        )


class BasicPlayer(Player):
    """Player moves towards nearest player and challenges them.

    Behavior:
        - Opponent selection: Select nearest player
        - Move selection: Select nearest player
        - Challenge acceptance: Always accepts
    """

    def select_card(self, opponent: Player | None = None) -> Move:
        """Randomly select one available move.

        Args:
            opponent: The opposing player, unused.

        Returns:
            A randomly selected available move.
        """

        return random.choice(self.available_cards())

    def challenge_opponent(
        self,
        available_opponents: list[Player],
    ) -> tuple[bool, list[Player], Player | None]:
        """Challenge the nearest opponent in range.

        Args:
            available_opponents: Players currently in challenge range.
            alive_players: All players still active in the round.

        Returns:
            A tuple containing:
                - whether the challenge was accepted
                - the updated alive-player list
                - the selected opponent, or None if no challenge occurred
        """
        if not available_opponents or not self.has_cards():
            return None

        opponent = min(
            available_opponents,
            key=lambda p: chebyshev(self.position, p.position),
        )

        return opponent

    def accept_challenge(self, opponent: Player) -> bool:
        """Always accept a challenge.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            True if this player has cards, otherwise False.
        """
        return self.has_cards()

    def get_playertype(self) -> PlayerType:
        """Return the player type.

        Returns:
            PlayerType.RANDOM.
        """
        return PlayerType.RANDOM

    def select_direction(self, alive_players: list[Player]) -> Direction:
        """Move toward the nearest player.

        Args:
            alive_players: All currently active players.

        Returns:
            The direction that best closes distance to the nearest player.
        """
        others = [p for p in alive_players if p is not self]
        if not others:
            return random.choice(list(Direction))
        nearest = min(
            others, key=lambda p: chebyshev(self.position, p.position)
        )

        return self._toward(nearest)
