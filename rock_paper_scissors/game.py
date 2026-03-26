from enum import Enum

class RPSMove(Enum):
  '''Represenst all possible moves in Rock Paper Scissors'''
  ROCK=1
  PAPER=2
  SCISSORS=3

  def compare(rps_move, other):
    '''
    Compare two RPSMoves to see if the first move beats the other. If
    one or more arguments are not RPSMoves, TypeError exception is thrown.
    
    Args:
      rps_move(RPSMove): first move to compare
      other(RPSMove): second move to compare
    
    Return:
      int from -1 to 1. -1 represents a loss, 0 represents a tie, 1 represents a win
    '''
    if not (rps_move in RPSMove and other in RPSMove):
      raise TypeError('One or both objects need to be RPSMoves.')

    match rps_move:
      case RPSMove.ROCK:
        if other == RPSMove.ROCK:
          return 0
        elif other == RPSMove.PAPER:
          return -1
        else:
          return 1
      case RPSMove.PAPER:
        if other == RPSMove.ROCK:
          return 1
        elif other == RPSMove.PAPER:
          return 0
        else:
          return -1
      case RPSMove.SCISSORS:
        if other == RPSMove.ROCK:
          return -1
        elif other == RPSMove.PAPER:
          return 1
        else:
          return 0
    
    

class GAME_RESULT(Enum):
  '''Represents a possible set of outcomes for Rock Paper Scissors'''
  TIE=0
  P1_WIN=1
  P2_WIN=2
  

class RockPaperScissors:
  def __init__(self):
    self.__p1_move = None
    self.__p2_move = None
    print('Rock Paper Scissors Game Started')

  def set_move(self, player_num, move):
    if not move in RPSMove:
      raise TypeError("That's not a proper move.") 
    if player_num == 1:
      self.__p1_move = move
    elif player_num == 2:
      self.__p2_move = move
    else:
      raise ValueError("That's not a proper player.")

  def get_winner(self):
    result = RPSMove.compare(self.__p1_move, self.__p2_move)
    if result == 1:
      return GAME_RESULT.P1_WIN
    elif result == -1:
      return GAME_RESULT.P2_WIN
    else:
      return GAME_RESULT.TIE