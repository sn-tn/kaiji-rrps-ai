from __future__ import annotations
import random
import numpy
from environment.move import Move, Direction, chebyshev
from abc import ABC, abstractmethod
from enum import Enum

class PlayerType(Enum):
    AGENT = "Agent"
    RANDOM = "Random"
    AGGRESSIVE = "Aggressive"
    CONSERVATIVE = "Conservative"


class Player(ABC):
    '''
    Abstract representation of a player in a tournament
    
    Each player has a fixed move budget and number of stars
    '''

    rng = numpy.random.default_rng()
    #===================== Universal =====================
    '''
    Instantiation
    Args:
        player_id   (int) the id of each player
        stars       (int) the number of stars the player starts with
        budget      (int) the number of each move the players start with
    '''
    def __init__(self, player_id: int, stars: int = 3, budget: int = 4, position: tuple[int, int] = (0, 0)):
        self.id       = player_id
        self.stars    = stars
        self.budget   = {Move.ROCK: budget, Move.PAPER: budget, Move.SCISSORS: budget}
        self.position = position

    def available_moves(self) -> list[Move]:
        '''
        A list containing the moves available

        Return:
            list[Move]
        '''
        return [m for m, n in self.budget.items() if n > 0]
    
    def is_alive(self) -> bool:
        '''
        Determines if the player is still alive

        Return:
            bool
        '''
        return self.stars > 0 and len(self.available_moves()) > 0
    
    def use_move(self, move: Move):
        '''
        Manages the move budget after an action
        Args:
            move (Move) the action taken by player
        '''
        self.budget[move] -= 1

    def steal_life(self, other: "Player"):
       '''
        Steals the other player's star
        Args:
            other   (Player) opponent star is being stolen from
        '''

       self.stars += 1
       other.lose_life()

    def lose_life(self):
        '''
        Deducts a star
        '''
        self.stars -= 1

    def _toward(self, target: Player) -> Direction:
        '''
        Finds the direction to get to a specified player
        Args:
            taget   (Player) the destination
        Return:
            Direction
        '''
        dx = numpy.sign(target.position[0] - self.position[0])
        dy = numpy.sign(target.position[1] - self.position[1])
        # find the Direction whose value matches the delta
        return min(Direction, key=lambda d: abs(d.value[0] - dx) + abs(d.value[1] - dy))

    def _away(self, target: Player) -> Direction:
        '''
        Finds the direction to evade a player
        Args:
            target  (Player) the avoided player
        Return:
            Direction
        '''
        dx = numpy.sign(self.position[0] - target.position[0])
        dy = numpy.sign(self.position[1] - target.position[1])
        return min(Direction, key=lambda d: abs(d.value[0] - dx) + abs(d.value[1] - dy))

    #===================== Abstract Methods =====================

    
    @abstractmethod
    def select_move(self, op: "Player" | None = None) -> Move: 
        '''
        Selects particular move from available moves based on specified player behavior
        Args:
            op          (Player) the opponent

        Return:
            Move
        '''
        ...

    
    @abstractmethod
    def select_opponent(self, in_range: list["Player"], all_alive: list ["Player"]) -> tuple[bool, list["Player"], "Player" | None]:
        '''
        Selects an opponent to challenge from avilable opponents in range based on player behavior
        Args:
            in_range    (list[Player]) list of players within challenge range
            all_alive   (list[Player]) list of all players still involved
        Return:
            tuple[bool, list[Player, Player or None]

        Return statement includes, in order; Whether challenge was accepted, 
                                                new list of players without opponents, 
                                                opponent or None if rejected
        '''
        ...
    
    @abstractmethod
    def accept_opponent(self, opponent: "Player") -> bool: 
        '''
        Determines wether or not to accept a challenge based on player behavior
        Args:
            oopponent   (Player) the player sending the challenge
        Return:
            bool
        '''
        ...

    
    @abstractmethod
    def get_playertype(self) -> PlayerType:
        '''
        Returns the player type from the enum
        
        Return:
            PlayerType
        '''
        ...

    
    @abstractmethod
    def select_direction(self, all_alive: list["Player"]) -> tuple[int, int]:
        '''
        Selects the direction to move based on PlayerType
        Args:
            alive       (list[Player]) the list of alive opponents
        Return
            tuple[int, int]
        '''
        ...


class RandomPlayer(Player):
    '''
    Player with completely randomized behavior

    * Opponent selection:   selects a random opponent from list of viable opponents

    * Move selection:       Selects a random move from remaining moveset

    * Accept opponent:      80% chance to accept a given opponent
    '''
    def select_move(self, op: Player) -> Move:
        '''
        Randomly selects a move from the set of available moves

        Return
            Move
        '''
        return random.choice(self.available_moves())

    def select_opponent(self, in_range: list[Player], all_alive: list [Player]) -> tuple[bool, list[Player], Player | None]:
        '''
        Selects a random opponent from the provided list to 'battle'
        If accepted, returns if accepted and list without them, and opponent selected
        Cannot select self as opponent
        Args:
            all_alive (list[Players]) list of opponents

        Return
            tuple[bool, list[Player], Player]
        '''
        # if no one avaiable abort
        if not in_range:
            return False, all_alive, None
        
        # challenge random opponent
        op = random.choice(in_range)
        if op.accept_opponent(self):
            return True, [p for p in all_alive if p is not op and p is not self], op
        
        # if rejected abort
        return False, all_alive, None

    def accept_opponent(self, challenger: Player) -> bool:
        '''
        Decides wether to accept a challenge with probability 0.8 yes

        Return
            bool
        '''
        return bool(self.rng.choice([True, False], p=[0.8, 0.2]))
    
    def get_playertype(self) -> PlayerType:
        '''
        Returns Enum of player type

        Return
            PlayerType
        '''
        return PlayerType.RANDOM
    
    def select_direction(self, all_alive: list[Player]) -> tuple[int, int]:
        '''
        Decides a random direction for the player to move
        '''
        return random.choice(list(Direction))
    
'''
The agent being trained

* Opponent and movement selection are handled externally and should not be called

* Agent should always accept opponent
'''
class AgentPlayer(Player):
    def select_move(self, op: Player | None) -> Move:
        # Agent moves are driven externally via env.step(action)
        # This fallback should never be called during normal training
        raise NotImplementedError("AgentPlayer moves are controlled by the environment")
    
    def select_opponent(self, in_range: list[Player], all_alive: list [Player]) -> tuple[bool, list[Player], Player | None]:
        # Agent challenges are also handled externally in step()
        raise NotImplementedError("AgentPlayer challenges are controlled by the environment")

    def accept_opponent(self, challenger: Player) -> bool:
        # Agent always accepts — the environment handles this interaction
        return True
    
    def get_playertype(self) -> PlayerType:
        return PlayerType.AGENT
    
    def select_direction(self, all_alive: list[Player]):
        # Agent moves are driven externally via env.step(action)
        # This fallback should never be called during normal training
        raise NotImplementedError("AgentPlayer moves are controlled by the environment")

'''
Aggressive player

* select_opponent(): Finds the player with the most stars
'''
class AggressivePlayer(Player):
    
    def select_move(self, op: "Player" | None = None) -> Move: 
        '''
        Selects the move it has the most left of
        Args:
            op          (Player) the opponent

        Return:
            Move
        '''
        return max(self.budget, key=lambda m: self.budget[m])

    def select_opponent(self, in_range: list[Player], all_alive: list [Player]) -> tuple[bool, list[Player], Player | None]:
        '''
        Targets opponent with highest starcount in range
        Args:
            in_range    (list[Player]) list of players within challenge range
            all_alive   (list[Player]) list of all players still involved
        Return:
            tuple[bool, list[Player, Player or None]

        Return statement includes, in order; Whether challenge was accepted, 
                                                new list of players without opponents, 
                                                opponent or None if rejected
        '''
        if not in_range:
            return False, all_alive, None
        
        op = max(in_range, key=lambda x: x.stars)

        if op.accept_opponent(self):
            return True, [p for p in all_alive if p is not op and p is not self], op
        
        return False, all_alive, None

    def accept_opponent(self, opponent: Player) -> bool: 
        '''
        Always accepts a challenger
        Args:
            oopponent   (Player) the player sending the challenge
        Return:
            bool
        '''
        return True
 
    def get_playertype(self) -> PlayerType:
        '''
        Returns the player type from the enum
        
        Return:
            PlayerType
        '''
        return PlayerType.AGGRESSIVE

    def select_direction(self, all_alive: list[Player]) -> Direction:
        '''
        Moves towards closest player
        Args:
            alive       (list[Player]) the list of alive opponents
        Return
            tuple[int, int]
        '''
        target = min(all_alive, key=lambda p: chebyshev(self.position, p.position))
        return self._toward(target)
    

class ConservativePlayer(Player):
    def select_move(self, op: "Player" | None = None) -> Move: 
        '''
        Selects the move it has the most left of
        Args:
            op          (Player) the opponent

        Return:
            Move
        '''
        return max(self.budget, key=lambda m: self.budget[m])

    def select_opponent(self, in_range: list[Player], all_alive: list [Player]) -> tuple[bool, list[Player], Player | None]:
        '''
        Targets opponent with lowest starcount in range
        Args:
            in_range    (list[Player]) list of players within challenge range
            all_alive   (list[Player]) list of all players still involved
        Return:
            tuple[bool, list[Player, Player or None]

        Return statement includes, in order; Whether challenge was accepted, 
                                                new list of players without opponents, 
                                                opponent or None if rejected
        '''
        if not in_range:
            return False, all_alive, None
        
        op = min(in_range, key=lambda x: x.stars)

        if op.accept_opponent(self):
            return True, [p for p in all_alive if p is not op and p is not self], op
        
        return False, all_alive, None

    def accept_opponent(self, opponent: Player) -> bool: 
        '''
        Only accepts if has more stars (0.3 chance accepting regardless)
        Args:
            oopponent   (Player) the player sending the challenge
        Return:
            bool
        '''
        if numpy.random.choice([True, False], p=[0.3, 0.7]):
            return True
        
        return self.stars >= opponent.stars
 
    def get_playertype(self) -> PlayerType:
        '''
        Returns the player type from the enum
        
        Return:
            PlayerType
        '''
        return PlayerType.CONSERVATIVE

    def select_direction(self, all_alive: list[Player]) -> Direction:
        '''
        Moves away from nearest player unless they have fewer or equal stars
        '''
        target = min(all_alive, key=lambda p: chebyshev(self.position, p.position))
        if self.stars >= target.stars:
            return self._toward(target)
        return self._away(target)