import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from enviroment_static.rrps_gym import RestrictedRPSEnv
from enviroment_static.Q_learn import obs_to_key, softmax
from gym_core.cards import Card

# ── config ────────────────────────────────────────────────────────────────────
NUM_EPISODES = 10_000
DECAY_RATE   = 0.999
EVAL_EPISODES = 10_000

env = RestrictedRPSEnv(n_opponents=3, stars=3)

filename = f"Q_table_{NUM_EPISODES}_{DECAY_RATE}.pickle"
with open(filename, "rb") as f:
    Q_table = pickle.load(f)
print(f"Loaded Q-table: {len(Q_table)} states")

# ── collect matchup data ───────────────────────────────────────────────────────
CARDS = [Card.rock, Card.paper, Card.scissors]
CARD_LABELS = ["Rock", "Paper", "Scissors"]
card_idx = {c: i for i, c in enumerate(CARDS)}

# response_matrix[opp_idx, agent_idx] = raw count
response_matrix = np.zeros((3, 3), dtype=int)

for _ in tqdm(range(EVAL_EPISODES), desc="Evaluating"):
    obs, _ = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = obs_to_key(obs)
        agent_has_no_rock = obs["player_dict"][0]["rock_total"] == 0
        try:
            action = np.random.choice(
                env.action_space.n, p=softmax(Q_table[state])
            )
        except KeyError:
            action = env.action_space.sample()

        obs, _, terminated, truncated, info = env.step(action)

        if not agent_has_no_rock:
            continue

        matchup_dict = info.get("matchup_dict") or {}
        for (pid1, pid2), (card1, card2) in matchup_dict.items():
            if pid1 == 0:
                agent_card, opp_card = card1, card2
            elif pid2 == 0:
                agent_card, opp_card = card2, card1
            else:
                continue
            if agent_card in card_idx and opp_card in card_idx:
                response_matrix[card_idx[opp_card], card_idx[agent_card]] += 1

# ── normalise rows → conditional probabilities ────────────────────────────────
row_sums = response_matrix.sum(axis=1, keepdims=True)
response_probs = response_matrix / np.where(row_sums > 0, row_sums, 1)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(response_probs, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax, label="Probability")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CARD_LABELS)
ax.set_yticklabels(CARD_LABELS)
ax.set_xlabel("Agent's card")
ax.set_ylabel("Opponent's card")
ax.set_title("Agent response heatmap\nP(agent plays X  |  opponent played Y,  agent has no rock)")

# annotate cells
for row in range(3):
    for col in range(3):
        p  = response_probs[row, col]
        n  = response_matrix[row, col]
        text_color = "white" if p > 0.6 else "black"
        ax.text(col, row, f"{p:.2f}\n(n={n:,})",
                ha="center", va="center", fontsize=9, color=text_color)

plt.tight_layout()
out = "response_heatmap.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
