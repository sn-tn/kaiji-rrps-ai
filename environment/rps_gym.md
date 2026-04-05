# RestrictedRPSEnv — Gymnasium Environment

A single-agent [Gymnasium](https://gymnasium.farama.org/) environment for **Restricted Rock Paper Scissors** (RRPS), inspired by the anime _Kaiji_. The agent competes in a tournament against randomly-acting opponents. Each player has a limited budget of moves and a finite number of lives — when either runs out, the player is eliminated.

---

## Quick Start

```python
import gymnasium as gym
from rps_gym import RestrictedRPSEnv

env = RestrictedRPSEnv(n_opponents=3, lives=3, budget=4, render_mode="human")
obs, info = env.reset(seed=0)

terminated = False
total_reward = 0.0

while not terminated:
    action = env.action_space.sample()          # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode over. Result: {info.get('result')} | Total reward: {total_reward}")
env.close()
```

---

## Helper Function: `resolve`

```python
from rps_gym import resolve
from move import Move

result = resolve(Move.ROCK, Move.SCISSORS)  # → 1  (Rock wins)
result = resolve(Move.PAPER, Move.ROCK)     # → 1  (Paper wins)
result = resolve(Move.ROCK, Move.ROCK)      # → 0  (Tie)
```

Returns `1` if the first move wins, `-1` if the second wins, `0` on a tie. Raises `TypeError` if either argument is not a `Move`.

## Dependencies

| Package     | Purpose                                   |
| ----------- | ----------------------------------------- |
| `gymnasium` | RL environment interface                  |
| `numpy`     | Observation arrays                        |
| `player.py` | `Player` class                            |
| `move.py`   | `Move` enum (`ROCK`, `PAPER`, `SCISSORS`) |

---

## Game Rules

- Every player starts with `lives` lives and `budget` uses of each move (Rock, Paper, Scissors).
- Each step, the agent picks a move. A random live opponent is selected; if the opponent accepts the challenge (80% probability), the matchup resolves.
- **Win** a matchup → steal one life from the opponent (`reward +1`).
- **Lose** a matchup → opponent steals one life from you (`reward -1`).
- **Tie** → no life changes (`reward 0`).
- After the agent's matchup, alive opponents are paired randomly and fight each other.
- A player is **eliminated** when they have 0 lives **or** no moves remaining.
- The episode ends when only one player is left standing.

---

## Class: `RestrictedRPSEnv`

```python
from rps_gym import RestrictedRPSEnv

env = RestrictedRPSEnv(
    n_opponents=3,   # number of opponents (default: 3)
    lives=3,         # starting lives for every player (default: 3)
    budget=4,        # starting uses of each move per player (default: 4)
    render_mode=None # "human" to print state each step, None for silent
)
```

### Constructor Parameters

| Parameter     | Type          | Default | Description                                |
| ------------- | ------------- | ------- | ------------------------------------------ |
| `n_opponents` | `int`         | `3`     | Number of randomly-acting opponents        |
| `lives`       | `int`         | `3`     | Starting lives for all players             |
| `budget`      | `int`         | `4`     | Starting uses of each move per player      |
| `render_mode` | `str \| None` | `None`  | `"human"` prints state to stdout each step |

---

## Spaces

### Action Space — `Discrete(3)`

| Integer | Move     |
| ------- | -------- |
| `0`     | Rock     |
| `1`     | Paper    |
| `2`     | Scissors |

### Observation Space — `Box(9,)` (`float32`)

| Index | Value                           | Range     |
| ----- | ------------------------------- | --------- |
| 0     | Agent lives                     | `[0, 20]` |
| 1     | Agent Rock budget               | `[0, 10]` |
| 2     | Agent Paper budget              | `[0, 10]` |
| 3     | Agent Scissors budget           | `[0, 10]` |
| 4     | A random opponent's lives       | `[0, 20]` |
| 5     | That opponent's Rock budget     | `[0, 10]` |
| 6     | That opponent's Paper budget    | `[0, 10]` |
| 7     | That opponent's Scissors budget | `[0, 10]` |
| 8     | Number of alive opponents       | `[0, 20]` |

> Indices 4–7 show a randomly sampled live opponent each step. If no opponents remain, these are all `0`.

---

## Reward Structure

| Event              | Reward                             |
| ------------------ | ---------------------------------- |
| Win a matchup      | `+1`                               |
| Lose a matchup     | `-1`                               |
| Tie / no matchup   | `0`                                |
| Win the tournament | `+5` (added on top of step reward) |
| Eliminated         | `-3` (added on top of step reward) |

---

## API Methods

### `reset(seed=None, options=None) → (obs, info)`

Starts a new episode. Creates all players with the configured lives and budgets.

```python
obs, info = env.reset(seed=42)
```

### `step(action: int) → (obs, reward, terminated, truncated, info)`

Advances the environment by one step.

- `terminated` is `True` when the tournament ends (agent wins or is eliminated).
- `truncated` is always `False` (no step limit).
- `info["result"]` is `"victory"` or `"eliminated"` when the episode ends.

```python
obs, reward, terminated, truncated, info = env.step(0)  # play Rock
```

### `render()`

Prints the agent's current state to stdout. Called automatically each step when `render_mode="human"`.

```
[Agent] lives=3 budget=R4/P3/S4 | Alive opponents: 2
```

### `close()`

No-op. Included for Gymnasium API compliance.

---
