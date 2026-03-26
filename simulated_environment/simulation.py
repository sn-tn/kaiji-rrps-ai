from enum import Enum
import random
from player import Player

#representation of moveset
class Move(Enum):
  ROCK=1
  PAPER=2
  SCISSORS=3

'''
    Method to determine winner of encounter
    Args:
        m1  (Move) first player's move
        m2  (Move) second player's move
    Returns 0 if tie, 1 if m1 wins, -1 m2 wins
'''
def resolve(m1: Move, m2: Move) -> int:
    if not (m1 in Move and m1 in Move):
      raise TypeError('Arguments must be valid moves')

    match m1:
      case Move.ROCK:
        if m2 == Move.ROCK:
          return 0
        elif m2 == Move.PAPER:
          return -1
        else:
          return 1
      case Move.PAPER:
        if m2 == Move.ROCK:
          return 1
        elif m2 == Move.PAPER:
          return 0
        else:
          return -1
      case Move.SCISSORS:
        if m2 == Move.ROCK:
          return -1
        elif m2 == Move.PAPER:
          return 1
        else:
          return 0

"""
    Manages a restricted Rock Paper Scissors tournament between multiple players.

    Each round, alive players are randomly paired and each pair plays one move
    against each other. The loser of each matchup loses a life. A player is
    eliminated when they run out of lives or exhaust their move budget. The
    tournament runs until one player remains.
"""
class Tournament:

    '''
    Args:
        players (list[player]) the players involved in the tournament
    '''
    def __init__(self, players: list[Player]):
       self.players = players
       
    '''
    The players remaining in the tournament

    Return:
        list[Player]
    '''
    def _alive_players(self) -> list[Player]:
        return [p for p in self.players if p.is_alive()]
       
    '''
    Randomly pairs up two opponents from the list of alive players

    Return:
        list[tuple[Player, Player]]
    '''
    def _pair_players(self) -> list[tuple[Player, Player]]:
        unpaired = self._alive_players()
        pairs = []
        for p in unpaired:
           attempt = p.select_opponent
           if attempt[1]:
              pairs.append[p, attempt[3]]
              unpaired = attempt[2]
        return pairs

    '''
    Simulates a single round of RPS
    '''
    def _run_round(self):
        for p1 in self._pair_players():
            m1 = random.choice(p1.available_moves())
            m2 = random.choice(p2.available_moves())
            outcome = resolve(m1, m2)
            p1.use_move(m1)
            p2.use_move(m2)
            if outcome == 1:
                p1.steal_life(p2)
            elif outcome == -1:
                p1.lose_life()
    
    '''
    Simulates a tournament until one player remains. Returns the winner

    Return:
        Player
    '''
    def run(self) -> Player:
       
        while(len(self._alive_players) > 1):
            self._pair_players()
            self._run_round()
        
        return self._alive_players()[0]