from __future__ import annotations
import random
import numpy
from environment.move import Move

'''
    Representation of a player in a tournament
    
    Each player has a fixed move budget and number of lives
'''
class Player:
    rng = numpy.random.default_rng()

    '''
    Instantiation
    Args:
        player_id   (int) the id of each player
        lives       (int) the number of lives the player starts with
        budget      (int) the number of each move the players start with
    '''
    def __init__(self, player_id: int, lives: int = 3, budget: int = 4, position: tuple[int, int] = (0, 0)):
        self.id       = player_id
        self.lives    = lives
        self.budget   = {Move.ROCK: budget, Move.PAPER: budget, Move.SCISSORS: budget}
        self.position = position

    '''
    A list containing the moves available

    Return:
        list[Move]
    '''
    def available_moves(self) -> list[Move]:
        return [m for m, n in self.budget.items() if n > 0]

    '''
    Determines if the player is still alive

    Return:
        bool
    '''
    def is_alive(self) -> bool:
        return self.lives > 0 and len(self.available_moves()) > 0

    '''
    Manages the move budget after an action
    Args:
        move (Move) the action taken by player
    '''
    def use_move(self, move: Move):
        self.budget[move] -= 1

    '''
    Steals the other player's life
    Args:
        other   (Player) opponent life is being stolen from
    '''
    def steal_life(self, other: Player):
       self.lives += 1
       other.lose_life()

    '''
    Deducts a life
    '''
    def lose_life(self):
        self.lives -= 1

    '''
    Selects a random opponent from the provided list to 'battle'
    If accepted, returns if accepted and list without them, and opponent selected
    Cannot select self as opponent
    Args:
        ops (list[Players]) list of opponents

    Return
        tuple[bool, list[Player], Player]
    '''
    def select_opponent(self, ops: list[Player]) -> tuple[bool, list[Player], Player | None]:
        candidates = [p for p in ops if p is not self]
        if not candidates:
            return False, ops, None

        op = random.choice(candidates)
        if op.accept_opponent(self):
            remaining = [p for p in ops if p is not op and p is not self]
            return True, remaining, op
        return False, ops, None

        
    '''
    Decides wether to accept a challenge with probability 0.8 yes

    Return
        bool
    '''
    def accept_opponent(self, opponent: Player) -> bool:
        return bool(self.rng.choice([True, False], p=[0.8, 0.2], size=1))
