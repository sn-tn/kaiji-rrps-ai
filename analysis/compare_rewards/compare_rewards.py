import sys, os, time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from environment_dqn_nav.Q_learn import QLearnDQNNav
from environment_dqn_nav.rrps_gym import RestrictedRPSEnv as DQNEnv
from rrps_core.reward_config import RewardConfig

EVAL_EPISODES = 10_000

N_OPPONENTS = 10
STARS = 3
GRID_SIZE = 12
BUDGET = {"rock_total": 3, "paper_total": 3, "scissors_total": 3}

BASE_REWARDS = RewardConfig(
    win_matchup=100,
    lose_matchup=-100,
    tie_matchup=0,
    eliminated=-2000,
    victory=2000,
    invalid_move=-10,
    within_challenge_range=0,
    approach_opponent=0,
)
SHAPED_REWARDS = RewardConfig(
    win_matchup=100,
    lose_matchup=-100,
    tie_matchup=0,
    eliminated=-2000,
    victory=2000,
    invalid_move=-10,
    within_challenge_range=1,
    approach_opponent=0.5,
)

# ── DQN (no shaping) ──────────────────────────────────────────────────────────

dqn_base_env = DQNEnv(
    n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
    n_obs_opponents=4, agent_budget=BUDGET, player_budget=BUDGET,
    reward_config=BASE_REWARDS,
)
dqn_base = QLearnDQNNav(agent_name=os.path.join(HERE, "dqn_base"), env=dqn_base_env)
t0 = time.time()
dqn_base.train(total_timesteps=2_000_000)
dqn_base_train_time = time.time() - t0

# ── DQN (with shaping) ────────────────────────────────────────────────────────

dqn_shaped_env = DQNEnv(
    n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
    n_obs_opponents=4, agent_budget=BUDGET, player_budget=BUDGET,
    reward_config=SHAPED_REWARDS,
)
dqn_shaped = QLearnDQNNav(agent_name=os.path.join(HERE, "dqn_shaped"), env=dqn_shaped_env)
t0 = time.time()
dqn_shaped.train(total_timesteps=2_000_000)
dqn_shaped_train_time = time.time() - t0

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

base_avg, base_win = evaluate(dqn_base)
shaped_avg, shaped_win = evaluate(dqn_shaped)

# ── Plot ──────────────────────────────────────────────────────────────────────

labels = ["DQN\n(no shaping)", "DQN\n(shaped)"]
colors = ["steelblue", "darkorange"]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle(f"Reward Shaping Comparison  ({EVAL_EPISODES:,} eval episodes each)", fontsize=13)

ax = axes[0]
bars = ax.bar(labels, [dqn_base_train_time, dqn_shaped_train_time], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.0f}s")
ax.set_ylabel("Seconds")
ax.set_title("Training Time")

ax = axes[1]
bars = ax.bar(labels, [base_avg, shaped_avg], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.1f}")
ax.set_ylabel("Avg Reward")
ax.set_title("Average Reward")

ax = axes[2]
bars = ax.bar(labels, [base_win, shaped_win], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 110)
ax.set_title("Win Rate")

plt.tight_layout()
plt.show()
