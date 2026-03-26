from enum import Enum
import random

#representation of moveset
class Move(Enum):
  ROCK=1
  PAPER=2
  SCISSORS=3

'''
    Representation of a player in a tournament
    
    Each player has a fixed move budget and number of lives
'''
class Player:
    '''
    Instantiation
    Args:
        player_id   (int) the id of each player
        lives       (int) the number of lives the player starts with
        budget      (int) the number of each move the players start with
    '''
    def __init__(self, player_id: int, lives: int = 3, budget: int = 4):
        self.id     = player_id
        self.lives  = lives
        self.budget = {Move.ROCK: budget, Move.PAPER: budget, Move.SCISSORS: budget}

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
    Deducts a life
    '''
    def lose_life(self):
        self.lives -= 1

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
        alive = self._alive_players()
        random.shuffle(alive)
        return [(alive[i], alive[i+1]) for i in range(0, len(alive) - 1, 2)]

    '''
    Simulates a single round of RPS
    '''
    def _run_round(self):
        for p1, p2 in self._pair_players():
            m1 = random.choice(p1.available_moves())
            m2 = random.choice(p2.available_moves())
            outcome = resolve(m1, m2)
            p1.use_move(m1)
            p2.use_move(m2)
            if outcome == 1:
                p2.lose_life()
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