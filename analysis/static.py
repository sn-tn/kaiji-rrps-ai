import sys, os, argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment_static.Q_learn import QLearnStatic
from environment_static.rrps_gym import StaticRRPSEnv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", type=int, default=100_000, help="number of training episodes"
)
parser.add_argument(
    "--gui", action="store_true", help="enable GUI during training"
)
parser.add_argument(
    "--file",
    type=str,
    default="static",
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

env = StaticRRPSEnv(
    n_opponents=1,
    agent_budget={"paper_total": 3, "rock_total": 3, "scissors_total": 3},
    player_budget={"paper_total": 3, "rock_total": 3, "scissors_total": 3},
)
static = QLearnStatic(agent_name=args.file, env=env)

if args.load:
    static.load_from_path(args.load)
else:
    static.tabular_train(
        gamma=0.9,
        train_episodes=args.train,
        decay_rate=DECAY_RATE,
        gui=args.gui,
    )

CARDS = ["rock_total", "paper_total", "scissors_total"]
CARD_LABELS = ["Rock", "Paper", "Scissors"]

rewards = []
wins = 0
losses = 0
heatmap_wins = [[0] * 3 for _ in range(3)]
heatmap_counts = [[0] * 3 for _ in range(3)]

for _ in tqdm(range(100_000)):
    total_reward = 0
    first_matchup = None
    for obs, reward, terminated, truncated, info in static.play_agent(
        gui=args.gui
    ):
        total_reward += reward
        if first_matchup is None and info["matchup_dict"]:
            for (pid1, pid2), (c1, c2) in info["matchup_dict"].items():
                if pid1 == 0:
                    first_matchup = (c1.value, c2.value)
                elif pid2 == 0:
                    first_matchup = (c2.value, c1.value)
                if first_matchup:
                    break
    rewards.append(total_reward)
    won = info["game_status"] == "victory"
    if won:
        wins += 1
    elif info["game_status"] == "eliminated":
        losses += 1
    if first_matchup:
        ai, oi = CARDS.index(first_matchup[0]), CARDS.index(first_matchup[1])
        heatmap_counts[ai][oi] += 1
        if won:
            heatmap_wins[ai][oi] += 1

total = len(rewards)
avg_reward = sum(rewards) / total

win_rate_map = [
    [heatmap_wins[r][c] / heatmap_counts[r][c] * 100 if heatmap_counts[r][c] else 0
     for c in range(3)] for r in range(3)
]

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(win_rate_map, vmin=0, vmax=100, cmap="RdYlGn")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CARD_LABELS)
ax.set_yticklabels(CARD_LABELS)
ax.set_xlabel("Opponent card")
ax.set_ylabel("Agent card")
ax.set_title(f"First-round matchup win rate  |  overall: {wins/total*100:.1f}% win  |  avg reward: {avg_reward:.2f}")
for r in range(3):
    for c in range(3):
        ax.text(c, r, f"{win_rate_map[r][c]:.0f}%", ha="center", va="center", fontsize=11)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
