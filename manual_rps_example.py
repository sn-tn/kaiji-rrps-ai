from rock_paper_scissors.game import RockPaperScissors, RPSMove, GAME_RESULT

MOVE_PROMPT = 'Enter a move (\'r\', \'p\', \'s\'):'

def prompt_move(player_num):
  move = ''
  while not (move == 'r' or move == 'p' or move =='s'):
    print(f'Player {player_num}: {MOVE_PROMPT}')
    move = input()
  if move == 'r':
    return RPSMove.ROCK
  elif move == 'p':
    return RPSMove.PAPER
  else:
    return RPSMove.SCISSORS

test_game = RockPaperScissors()

p1_move = prompt_move(1)
test_game.set_move(1, p1_move)

p2_move = prompt_move(2)
test_game.set_move(2, p2_move)

print(test_game.get_winner())