from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum

import numpy

from environment.move import Move, Direction, chebyshev


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

    def available_moves(self) -> list[Move]:
        """Return the list of moves the player can still use.

        Returns:
            A list of moves with remaining budget greater than zero.
        """
        return [move for move, count in self.budget.items() if count > 0]

    def is_alive(self) -> bool:
        """Return whether the player is still active.

        A player is considered alive if they have at least one star and at
        least one available move remaining.

        Returns:
            True if the player is alive, otherwise False.
        """
        return self.stars > 0 and len(self.available_moves()) > 0

    def use_move(self, move: Move) -> None:
        """Consume one unit of budget for the given move.

        Args:
            move: The move being used.
        """
        self.budget[move] -= 1

    def steal_life(self, other: "Player") -> None:
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
        return min(
            Direction,
            key=lambda direction: abs(direction.value[0] - dx)
            + abs(direction.value[1] - dy),
        )

    def _away(self, target: "Player") -> Direction:
        """Return the direction that moves this player away from a target.

        Args:
            target: The player to move away from.

        Returns:
            The direction that best increases separation from the target.
        """
        dx = numpy.sign(self.position[0] - target.position[0])
        dy = numpy.sign(self.position[1] - target.position[1])
        return min(
            Direction,
            key=lambda direction: abs(direction.value[0] - dx)
            + abs(direction.value[1] - dy),
        )
    
    def has_cards(self):
        return len(self.available_moves()) > 0

    @abstractmethod
    def select_move(self, opponent: "Player" | None = None) -> Move:
        """Choose a move to play.

        Args:
            opponent: The opposing player, if applicable.

        Returns:
            The selected move.
        """
        ...

    @abstractmethod
    def select_opponent(
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
    def accept_opponent(self, opponent: "Player") -> bool:
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


class RandomPlayer(Player):
    """Player with randomized behavior.

    Behavior:
        - Opponent selection: random available opponent
        - Move selection: random available move
        - Challenge acceptance: 80% chance to accept
    """

    def select_move(self, opponent: Player | None = None) -> Move:
        """Randomly select one available move.

        Args:
            opponent: The opposing player, unused.

        Returns:
            A randomly selected available move.
        """
        return random.choice(self.available_moves())

    def select_opponent(
        self,
        available_opponents: list[Player],
        alive_players: list[Player],
    ) -> tuple[bool, list[Player], Player | None]:
        """Randomly choose an opponent from those in range.

        Args:
            available_opponents: Players currently in challenge range.
            alive_players: All players still active in the round.

        Returns:
            A tuple containing:
                - whether the challenge was accepted
                - the updated alive-player list
                - the selected opponent, or None if no challenge occurred
        """
        if not available_opponents:
            return False, alive_players, None

        opponent = random.choice(available_opponents)
        if opponent.accept_opponent(self):
            remaining_players = [
                player
                for player in alive_players
                if player is not opponent and player is not self
            ]
            return True, remaining_players, opponent

        return False, alive_players, None

    def accept_opponent(self, opponent: Player) -> bool:
        """Accept a challenge with 80% probability.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            True with probability 0.8, otherwise False.
        """
        return bool(self.rng.choice([True, False], p=[0.8, 0.2]))

    def get_playertype(self) -> PlayerType:
        """Return the player type.

        Returns:
            PlayerType.RANDOM.
        """
        return PlayerType.RANDOM

    def select_direction(self, alive_players: list[Player]) -> Direction:
        """Choose a random movement direction.

        Args:
            alive_players: All currently active players, unused.

        Returns:
            A randomly selected direction.
        """
        return random.choice(list(Direction))


class AgentPlayer(Player):
    """Player controlled externally by the environment.

    Opponent selection and movement are not handled here.
    The agent always accepts challenges.
    """

    def select_move(self, opponent: Player | None = None) -> Move:
        """Raise an error because move selection is environment-controlled.

        Args:
            opponent: The opposing player, if applicable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "AgentPlayer moves are controlled by the environment"
        )

    def select_opponent(
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

    def accept_opponent(self, opponent: Player) -> bool:
        """Always accept a challenge.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            Always True.
        """
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


class AggressivePlayer(Player):
    """Player that prefers stronger engagement.

    Behavior:
        - Move selection: move with the highest remaining budget
        - Opponent selection: opponent in range with the most stars
        - Challenge acceptance: always accept
        - Movement: move toward the nearest player
    """

    def select_move(self, opponent: Player | None = None) -> Move:
        """Select the move with the highest remaining budget.

        Args:
            opponent: The opposing player, unused.

        Returns:
            The move with the highest remaining count.
        """
        return max(self.budget, key=lambda move: self.budget[move])

    def select_opponent(
        self,
        available_opponents: list[Player],
        alive_players: list[Player],
    ) -> tuple[bool, list[Player], Player | None]:
        """Choose the in-range opponent with the most stars.

        Args:
            available_opponents: Players currently in challenge range.
            alive_players: All players still active in the round.

        Returns:
            A tuple containing:
                - whether the challenge was accepted
                - the updated alive-player list
                - the selected opponent, or None if no challenge occurred
        """
        if not available_opponents:
            return False, alive_players, None

        opponent = max(available_opponents, key=lambda player: player.stars)

        if opponent.accept_opponent(self):
            remaining_players = [
                player
                for player in alive_players
                if player is not opponent and player is not self
            ]
            return True, remaining_players, opponent

        return False, alive_players, None

    def accept_opponent(self, opponent: Player) -> bool:
        """Always accept a challenge.

        Args:
            opponent: The player issuing the challenge.

        Returns:
            Always True.
        """
        return True

    def get_playertype(self) -> PlayerType:
        """Return the player type.

        Returns:
            PlayerType.AGGRESSIVE.
        """
        return PlayerType.AGGRESSIVE

    def select_direction(self, alive_players: list[Player]) -> Direction:
        """Move toward the nearest player.

        Args:
            alive_players: All currently active players.

        Returns:
            The direction toward the nearest player.
        """
        target = min(
            alive_players,
            key=lambda player: chebyshev(self.position, player.position),
        )
        return self._toward(target)


class ConservativePlayer(Player):
    """Player that avoids unnecessary risk.

    Behavior:
        - Move selection: move with the highest remaining budget
        - Opponent selection: opponent in range with the fewest stars
        - Challenge acceptance:
            * 30% random chance to accept
            * otherwise accept only if this player has at least as many stars
        - Movement:
            * move toward weaker or equal opponents
            * move away from stronger opponents
    """

    def select_move(self, opponent: Player | None = None) -> Move:
        """Select the move with the highest remaining budget.

        Args:
            opponent: The opposing player, unused.

        Returns:
            The move with the highest remaining count.
        """
        return max(self.budget, key=lambda move: self.budget[move])

    def select_opponent(
        self,
        available_opponents: list[Player],
        alive_players: list[Player],
    ) -> tuple[bool, list[Player], Player | None]:
        """Choose the in-range opponent with the fewest stars.

        Args:
            available_opponents: Players currently in challenge range.
            alive_players: All players still active in the round.

        Returns:
            A tuple containing:
                - whether the challenge was accepted
                - the updated alive-player list
                - the selected opponent, or None if no challenge occurred
        """
        if not available_opponents:
            return False, alive_players, None

        opponent = min(available_opponents, key=lambda player: player.stars)

        if opponent.accept_opponent(self):
            remaining_players = [
                player
                for player in alive_players
                if player is not opponent and player is not self
            ]
            return True, remaining_players, opponent

        return False, alive_players, None

    def accept_opponent(self, opponent: Player) -> bool:
        """Decide whether to accept a challenge.

        Acceptance rule:
            - 30% chance to accept regardless
            - otherwise accept only if this player has at least as many stars
              as the challenger

        Args:
            opponent: The player issuing the challenge.

        Returns:
            True if the challenge is accepted, otherwise False.
        """
        if numpy.random.choice([True, False], p=[0.3, 0.7]):
            return True

        return self.stars >= opponent.stars

    def get_playertype(self) -> PlayerType:
        """Return the player type.

        Returns:
            PlayerType.CONSERVATIVE.
        """
        return PlayerType.CONSERVATIVE

    

    def select_direction(self, alive_players: list[Player]) -> Direction:
        """Move relative to the nearest player based on star advantage.

        Args:
            alive_players: All currently active players.

        Returns:
            A direction toward the nearest player if this player is at least as
            strong, otherwise a direction away from that player.
        """
        target = min(
            alive_players,
            key=lambda player: chebyshev(self.position, player.position),
        )
        if self.stars >= target.stars:
            return self._toward(target)
        return self._away(target)
