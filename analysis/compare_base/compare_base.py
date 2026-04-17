import sys, os, time
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
from gym_core.reward_config import RewardConfig

DECAY_RATE = 0.999
EVAL_EPISODES = 10_000

N_OPPONENTS = 10
STARS = 3
GRID_SIZE = 12
BUDGET = {"rock_total": 3, "paper_total": 3, "scissors_total": 3}
NAV_REWARD_CONFIG = RewardConfig(
    win_matchup=100,
    lose_matchup=-100,
    tie_matchup=0,
    eliminated=-2000,
    victory=2000,
    invalid_move=-10,
    within_challenge_range=1,
    approach_opponent=0.5,
)

# ── Static Q-Learn ────────────────────────────────────────────────────────────

static_env = StaticRRPSEnv(
    n_opponents=N_OPPONENTS,
    agent_budget=BUDGET,
    player_budget=BUDGET,
)
STATIC_EPISODES = 20_000
static_agent = QLearnStatic(agent_name=os.path.join(HERE, "compare_static"), env=static_env)
t0 = time.time()
static_agent.tabular_train(gamma=0.9, train_episodes=STATIC_EPISODES, decay_rate=DECAY_RATE)
static_train_time = (time.time() - t0) / (STATIC_EPISODES / 10_000)

# ── Tabular Nav ───────────────────────────────────────────────────────────────

tabular_env = TabularEnv(
    n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
    agent_budget=BUDGET, player_budget=BUDGET,
    reward_config=NAV_REWARD_CONFIG,
)
TABULAR_EPISODES = 20_000
tabular_agent = QLearnTabularNav(agent_name=os.path.join(HERE, "compare_tabular"), env=tabular_env)
t0 = time.time()
tabular_agent.tabular_train(gamma=0.9, train_episodes=TABULAR_EPISODES, decay_rate=DECAY_RATE)
tabular_train_time = (time.time() - t0) / (TABULAR_EPISODES / 10_000)

# ── DQN Nav ───────────────────────────────────────────────────────────────────

dqn_env = DQNEnv(n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
                 n_obs_opponents=4, agent_budget=BUDGET, player_budget=BUDGET,
                 reward_config=NAV_REWARD_CONFIG)
DQN_TIMESTEPS = 2_000_000
dqn_agent = QLearnDQNNav(agent_name=os.path.join(HERE, "compare_dqn"), env=dqn_env)
t0 = time.time()
dqn_agent.train(total_timesteps=DQN_TIMESTEPS)
dqn_train_time = (time.time() - t0) / (DQN_TIMESTEPS / 10_000)

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
bars = ax.bar(labels, [static_train_time, tabular_train_time, dqn_train_time], color=colors)
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
