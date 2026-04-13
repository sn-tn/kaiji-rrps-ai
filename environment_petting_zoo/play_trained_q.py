from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from environment_petting_zoo.rrps_pz import parallel_env


StateKey = Tuple[int, ...]


def state_key(observation: dict) -> StateKey:
    return tuple(int(x) for x in observation["observation"])


def legal_actions(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def is_independent_bundle(obj: object) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(k, str) and k.startswith("player_") and isinstance(v, dict) for k, v in obj.items())


def ensure_state(table: dict, key: StateKey, action_size: int) -> None:
    if key not in table:
        table[key] = np.zeros(action_size, dtype=np.float32)


def choose_shared_action(q_table: dict, obs: dict, action_size: int, rng: np.random.Generator) -> int:
    mask = obs["action_mask"]
    legal = legal_actions(mask)
    if legal.size == 0:
        return 0
    key = state_key(obs)
    ensure_state(q_table, key, action_size)
    q_values = q_table[key]
    legal_q = q_values[legal]
    best = np.max(legal_q)
    best_legal = legal[legal_q == best]
    return int(rng.choice(best_legal))


def choose_independent_action(q_tables: dict, agent: str, obs: dict, action_size: int, rng: np.random.Generator) -> int:
    mask = obs["action_mask"]
    legal = legal_actions(mask)
    if legal.size == 0:
        return 0
    table = q_tables[agent]
    key = state_key(obs)
    ensure_state(table, key, action_size)
    q_values = table[key]
    legal_q = q_values[legal]
    best = np.max(legal_q)
    best_legal = legal[legal_q == best]
    return int(rng.choice(best_legal))


def main(args: argparse.Namespace) -> None:
    policy_path = Path(args.policy)
    with policy_path.open("rb") as f:
        policy = pickle.load(f)

    env = parallel_env(
        n_players=args.players,
        stars=args.stars,
        max_rounds=args.max_rounds,
        render_mode="human",
    )
    rng = np.random.default_rng(args.seed)

    if is_independent_bundle(policy):
        mode = "independent"
    else:
        mode = "shared"

    print(f"loaded {mode} policy from {policy_path.resolve()}")

    reward_history = []
    per_agent_totals = defaultdict(float)
    rounds_history = []

    for episode in range(1, args.episodes + 1):
        observations, infos = env.reset(seed=args.seed + episode)
        episode_rewards = defaultdict(float)

        print(f"\nepisode {episode}")
        while env.agents:
            actions = {}
            for agent in env.agents:
                if mode == "independent":
                    actions[agent] = choose_independent_action(
                        q_tables=policy,
                        agent=agent,
                        obs=observations[agent],
                        action_size=env._action_size,
                        rng=rng,
                    )
                else:
                    actions[agent] = choose_shared_action(
                        q_table=policy,
                        obs=observations[agent],
                        action_size=env._action_size,
                        rng=rng,
                    )

            observations, rewards, terminations, truncations, infos = env.step(actions)
            print(f"rewards: {rewards}")
            print(f"terminations: {terminations}")
            print(f"truncations: {truncations}")

            for agent, reward in rewards.items():
                episode_rewards[agent] += reward
                per_agent_totals[agent] += reward

        rounds_history.append(env.round_count)
        reward_history.append(sum(episode_rewards.values()) / args.players)
        print(f"episode_reward_by_agent={dict(episode_rewards)}")
        print(f"episode_mean_reward_per_agent={reward_history[-1]:.3f}")
        print(f"episode_rounds={env.round_count}")

    print("\nsummary")
    print(f"episodes={args.episodes}")
    print(f"mean_reward_per_agent={float(np.mean(reward_history) if reward_history else 0.0):.3f}")
    print(f"mean_rounds={float(np.mean(rounds_history) if rounds_history else 0.0):.2f}")
    print(
        "mean_reward_by_agent="
        + str({agent: total / args.episodes for agent, total in per_agent_totals.items()})
    )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a saved shared or independent tabular Q policy."
    )
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--stars", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
