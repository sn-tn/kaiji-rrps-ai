# kaiji-rrps-ai

# Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

To install packages run in root:

```bash
uv sync
```

> Make sure `.pickle` and `.zip` model files are git ignored — they can be large enough to break version control.

## Overview

Three approaches to training an agent to play Restricted Rock Paper Scissors (RRPS), ranging from a static tabular environment to a navigable grid with Deep Q-Learning.

---

## Project Structure

### `rrps_core/`

Shared template classes, logic, and objects used across all environments. Contains the base Q-learning class, reward config, and the core gym class.

### `rrps_core/types`

Shared types used across all environments (player, card, matchup dict, observation, etc).

### Environment Folders

Each environment folder contains:

- `rrps_gym.py` — environment construction
- `Q_learn.py` — agent logic specific to that environment

### `environment_static/`

A static (non-grid) environment where the agent challenges opponents directly by selecting a target and card each round.

### `environment_tabular_nav/`

A navigable grid environment with a tabular Q-learning agent. The agent moves on a grid and matchups occur when two players share the same cell.

### `environment_dqn_nav/`

A navigable grid environment with a Deep Q-learning agent via Stable-Baselines3. Same matchup logic as tabular nav but with a DQN.

### `analysis/`

Scripts that train and evaluate each approach and generate the results referenced in the executive summary.

---

## Running Analysis Scripts

All scripts are run from the project root. Each supports `--train`, `--file`, `--load`, and `--gui` flags.

### Static Q-Learning

```bash
uv run -m analysis.static
uv run -m analysis.static --train 20000
uv run -m analysis.static --load analysis/static_100000_0.999.pickle --gui
```

### Tabular Nav Q-Learning

[tabular_nav_20000_0.999.pickle download](https://drive.google.com/drive/folders/1UtotVdp7LyLn43jnPGk_qa_U5ed6dY9j)

```bash
uv run -m analysis.tabular_nav
uv run -m analysis.tabular_nav --train 20000
uv run -m analysis.tabular_nav --load tabular_nav_20000_0.999.pickle --gui
```

### DQN Nav

```bash
uv run -m analysis.dqn
uv run -m analysis.dqn --train 1000000
uv run -m analysis.dqn --load analysis/dqn_nav --gui
```

### Comparing All Three (`compare_base/`)

Trains all three agents under identical environment settings and produces bar charts comparing training time, average reward, and win rate. Split into a train and eval script so results can be re-plotted without retraining. This is where the graphic in the executive summary comes from.

```bash
# Train all three agents (saves models + train times to analysis/compare_base/)
uv run -m analysis.compare_base.compare_base_train

# Evaluate and plot using saved models
uv run -m analysis.compare_base.compare_base_eval
```

---
