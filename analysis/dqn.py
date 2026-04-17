import sys, os, argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment_dqn_nav.Q_learn import QLearnDQNNav
from environment_dqn_nav.rrps_gym import RestrictedRPSEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", type=int, default=2_000_000, help="number of training timesteps"
)
parser.add_argument(
    "--gui", action="store_true", help="enable GUI during evaluation"
)
parser.add_argument(
    "--file",
    type=str,
    default="dqn_nav",
    help="agent name used for save filename",
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="path to a saved model to skip training",
)
args = parser.parse_args()

env = RestrictedRPSEnv(
    n_opponents=10, stars=3, n_obs_opponents=4, grid_size=14
)
agent = QLearnDQNNav(agent_name=args.file, env=env)

if args.load:
    agent.load(args.load)
else:
    agent.train(total_timesteps=args.train)

rewards = []
wins = 0
losses = 0

for _ in tqdm(range(10_000)):
    total_reward = 0.0
    for obs, reward, terminated, truncated, info in agent.play_agent(
        gui=args.gui
    ):
        total_reward += reward
    rewards.append(total_reward)
    if info["game_status"] == "victory":
        wins += 1
    elif info["game_status"] == "eliminated":
        losses += 1

total = len(rewards)
avg_reward = sum(rewards) / total

labels = ["Win", "Loss"]
counts = [wins, losses]
colors = ["steelblue", "tomato"]

fig, ax = plt.subplots()
bars = ax.bar(labels, [c / total * 100 for c in counts], color=colors)
ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%")
ax.set_ylabel("Rate (%)")
ax.set_title(
    f"Results over {total:,} episodes  |  avg reward: {avg_reward:.2f}"
)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()
