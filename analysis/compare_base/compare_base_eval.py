import sys, os, json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from environment_static.Q_learn import QLearnStatic
from environment_static.rrps_gym import StaticRRPSEnv
from environment_tabular_nav.Q_learn import QLearnTabularNav
from environment_tabular_nav.rps_gym import RestrictedRPSEnv as TabularEnv
from environment_dqn_nav.Q_learn import QLearnDQNNav
from environment_dqn_nav.rrps_gym import RestrictedRPSEnv as DQNEnv
from rrps_core.reward_config import RewardConfig

EVAL_EPISODES = 100_000
N_OPPONENTS = 10
STARS = 3
GRID_SIZE = 12
BUDGET = {"rock_total": 3, "paper_total": 3, "scissors_total": 3}
REWARD_CONFIG = RewardConfig(
    win_matchup=100,
    lose_matchup=-100,
    tie_matchup=0,
    eliminated=-2000,
    victory=2000,
    invalid_move=-10,
    within_challenge_range=1,
    approach_opponent=0.5,
)

# ── Load agents ───────────────────────────────────────────────────────────────

static_env = StaticRRPSEnv(
    n_opponents=N_OPPONENTS,
    agent_budget=BUDGET,
    player_budget=BUDGET,
    reward_config=REWARD_CONFIG,
)
static_agent = QLearnStatic(agent_name="compare_static", env=static_env)
static_agent.load_from_path(os.path.join(HERE, "compare_static_20000_0.999.pickle"))

tabular_env = TabularEnv(
    n_opponents=N_OPPONENTS,
    stars=STARS,
    grid_size=GRID_SIZE,
    agent_budget=BUDGET,
    player_budget=BUDGET,
    reward_config=REWARD_CONFIG,
)
tabular_agent = QLearnTabularNav(agent_name="compare_tabular", env=tabular_env)
tabular_agent.load_from_path(os.path.join(HERE, "compare_tabular_20000_0.999.pickle"))

dqn_env = DQNEnv(
    n_opponents=N_OPPONENTS,
    stars=STARS,
    grid_size=GRID_SIZE,
    n_obs_opponents=4,
    agent_budget=BUDGET,
    player_budget=BUDGET,
    reward_config=REWARD_CONFIG,
)
dqn_agent = QLearnDQNNav(agent_name="compare_dqn", env=dqn_env)
dqn_agent.load(os.path.join(HERE, "compare_dqn"))

with open(os.path.join(HERE, "train_times.json")) as f:
    train_times = json.load(f)

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(agent):
    rewards, wins = [], 0
    for _ in tqdm(range(EVAL_EPISODES)):
        total = 0.0
        for _, reward, _, _, info in agent.play_agent():
            total += reward
        rewards.append(total)
        if info["game_status"] == "victory":
            wins += 1
    return np.mean(rewards), wins / EVAL_EPISODES * 100

static_avg, static_win = evaluate(static_agent)
tabular_avg, tabular_win = evaluate(tabular_agent)
dqn_avg, dqn_win = evaluate(dqn_agent)

# ── Plot ──────────────────────────────────────────────────────────────────────

labels = ["Static\nQ-Learn", "Tabular\nNav", "DQN\nNav"]
colors = ["steelblue", "mediumseagreen", "darkorange"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"Algorithm Comparison  ({EVAL_EPISODES:,} eval episodes each)", fontsize=13)

ax = axes[0]
bars = ax.bar(labels, [train_times["static"], train_times["tabular"], train_times["dqn"]], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.0f}s")
ax.set_ylabel("Seconds per 10k")
ax.set_title("Avg Training Time (per 10k)")

ax = axes[1]
bars = ax.bar(labels, [static_avg, tabular_avg, dqn_avg], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.1f}")
ax.set_ylabel("Avg Reward")
ax.set_title("Average Reward")

ax = axes[2]
bars = ax.bar(labels, [static_win, tabular_win, dqn_win], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 110)
ax.set_title("Win Rate")

plt.tight_layout()
plt.show()
