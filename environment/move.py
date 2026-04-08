from enum import Enum

class Move(Enum):
    ROCK=0
    PAPER=1
    SCISSORS=2

class Direction(Enum):
    UP   = (0, -1)  # up
    DOWN = (0,  1)  # down
    LEFT = (-1, 0)  # left
    RIGHT= (1,  0)  # right

def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev (chessboard) distance between two grid positions."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))