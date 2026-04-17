import sys, os, argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment_tabular_nav.Q_learn import QLearnTabularNav
from environment_tabular_nav.rps_gym import RestrictedRPSEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", type=int, default=20_000, help="number of training episodes"
)
parser.add_argument(
    "--gui", action="store_true", help="enable GUI during training"
)
parser.add_argument(
    "--file",
    type=str,
    default="tabular_nav",
    help="agent name used for save filename",
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="path to a saved agent pickle to skip training",
)
args = parser.parse_args()

DECAY_RATE = 0.999

env = RestrictedRPSEnv(
    n_opponents=30,
    stars=3,
    grid_size=12,
)
agent = QLearnTabularNav(agent_name=args.file, env=env)

if args.load:
    agent.load_from_path(args.load)
else:
    agent.tabular_train(
        gamma=0.9,
        train_episodes=args.train,
        decay_rate=DECAY_RATE,
        gui=args.gui,
    )

rewards = []
wins = 0
losses = 0
truncations = 0

for _ in tqdm(range(10_000)):
    total_reward = 0
    for obs, reward, terminated, truncated, info in agent.play_agent():
        total_reward += reward
    rewards.append(total_reward)
    if info["game_status"] == "victory":
        wins += 1
    elif info["game_status"] == "eliminated":
        losses += 1
    else:
        truncations += 1

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
