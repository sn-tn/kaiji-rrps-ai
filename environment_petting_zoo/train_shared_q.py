from __future__ import annotations

import argparse
import json
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


def choose_action(
    q_table: Dict[StateKey, np.ndarray],
    obs: dict,
    epsilon: float,
    action_size: int,
    rng: np.random.Generator,
) -> int:
    mask = obs["action_mask"]
    legal = legal_actions(mask)
    if legal.size == 0:
        return 0

    key = state_key(obs)
    if key not in q_table:
        q_table[key] = np.zeros(action_size, dtype=np.float32)

    if rng.random() < epsilon:
        return int(rng.choice(legal))

    q_values = q_table[key]
    legal_q = q_values[legal]
    best = np.max(legal_q)
    best_legal = legal[legal_q == best]
    return int(rng.choice(best_legal))


def max_next_q(
    q_table: Dict[StateKey, np.ndarray], next_obs: dict, action_size: int
) -> float:
    mask = next_obs["action_mask"]
    legal = legal_actions(mask)
    if legal.size == 0:
        return 0.0

    key = state_key(next_obs)
    if key not in q_table:
        q_table[key] = np.zeros(action_size, dtype=np.float32)
    return float(np.max(q_table[key][legal]))


def evaluate(
    q_table: Dict[StateKey, np.ndarray],
    n_players: int,
    episodes: int,
    max_rounds: int,
    seed: int,
) -> dict:
    env = parallel_env(n_players=n_players, max_rounds=max_rounds)
    rng = np.random.default_rng(seed)
    rewards = []
    clears = 0
    last_agent_standing = 0
    avg_rounds = []

    for ep in range(episodes):
        observations, infos = env.reset(seed=seed + ep)
        episode_reward = defaultdict(float)

        while env.agents:
            actions = {}
            for agent in env.agents:
                obs = observations[agent]
                mask = obs["action_mask"]
                legal = legal_actions(mask)
                if legal.size == 0:
                    actions[agent] = 0
                    continue

                key = state_key(obs)
                if key not in q_table:
                    q_table[key] = np.zeros(env._action_size, dtype=np.float32)
                q_values = q_table[key]
                legal_q = q_values[legal]
                best = np.max(legal_q)
                best_legal = legal[legal_q == best]
                actions[agent] = int(rng.choice(best_legal))

            observations, step_rewards, terminations, truncations, infos = env.step(actions)
            for agent, reward in step_rewards.items():
                episode_reward[agent] += reward
            for agent, info in infos.items():
                if info.get("result") == "cleared_game":
                    clears += 1
                elif info.get("result") == "last_agent_standing":
                    last_agent_standing += 1

        rewards.append(sum(episode_reward.values()) / max(1, n_players))
        avg_rounds.append(env.round_count)

    env.close()
    return {
        "episodes": episodes,
        "mean_reward_per_agent": float(np.mean(rewards) if rewards else 0.0),
        "mean_rounds": float(np.mean(avg_rounds) if avg_rounds else 0.0),
        "cleared_game_events": int(clears),
        "last_agent_standing_events": int(last_agent_standing),
    }


def train(args: argparse.Namespace) -> tuple[dict, dict]:
    env = parallel_env(
        n_players=args.players,
        max_rounds=args.max_rounds,
        stars=args.stars,
        render_mode="human" if args.render else None,
    )
    rng = np.random.default_rng(args.seed)
    q_table: Dict[StateKey, np.ndarray] = {}

    epsilon = args.epsilon_start
    rewards_history = []
    round_history = []

    for episode in range(1, args.episodes + 1):
        observations, infos = env.reset(seed=args.seed + episode)
        done_agents = set()
        episode_reward = defaultdict(float)

        while env.agents:
            prev_obs = {agent: observations[agent] for agent in env.agents}
            actions = {}
            for agent in env.agents:
                actions[agent] = choose_action(
                    q_table=q_table,
                    obs=observations[agent],
                    epsilon=epsilon,
                    action_size=env._action_size,
                    rng=rng,
                )

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent, prev in prev_obs.items():
                s_key = state_key(prev)
                if s_key not in q_table:
                    q_table[s_key] = np.zeros(env._action_size, dtype=np.float32)
                a = actions[agent]
                r = float(rewards.get(agent, 0.0))
                episode_reward[agent] += r
                terminal = bool(terminations.get(agent, False) or truncations.get(agent, False))

                if terminal or agent not in next_observations:
                    target = r
                else:
                    target = r + args.gamma * max_next_q(
                        q_table, next_observations[agent], env._action_size
                    )

                q_old = q_table[s_key][a]
                q_table[s_key][a] = q_old + args.alpha * (target - q_old)

                if terminal:
                    done_agents.add(agent)

            observations = next_observations

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        rewards_history.append(sum(episode_reward.values()) / args.players)
        round_history.append(env.round_count)

        if args.log_every and episode % args.log_every == 0:
            recent_rewards = rewards_history[-args.log_every :]
            recent_rounds = round_history[-args.log_every :]
            print(
                f"episode={episode} "
                f"epsilon={epsilon:.4f} "
                f"avg_reward_per_agent={np.mean(recent_rewards):.3f} "
                f"avg_rounds={np.mean(recent_rounds):.2f} "
                f"states={len(q_table)}"
            )

    train_metrics = {
        "episodes": args.episodes,
        "players": args.players,
        "stars": args.stars,
        "max_rounds": args.max_rounds,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "seed": args.seed,
        "q_table_states": len(q_table),
        "final_epsilon": float(epsilon),
        "mean_training_reward_per_agent": float(np.mean(rewards_history) if rewards_history else 0.0),
        "mean_training_rounds": float(np.mean(round_history) if round_history else 0.0),
    }

    eval_metrics = evaluate(
        q_table=q_table,
        n_players=args.players,
        episodes=args.eval_episodes,
        max_rounds=args.max_rounds,
        seed=args.seed + 100_000,
    )

    bundle = {
        "q_table": {k: v.tolist() for k, v in q_table.items()},
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"shared_q_p{args.players}_ep{args.episodes}_seed{args.seed}"
    q_path = out_dir / f"{stem}.pkl"
    metrics_path = out_dir / f"{stem}_metrics.json"

    with q_path.open("wb") as f:
        pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_metrics, "eval_metrics": eval_metrics}, f, indent=2)

    print(f"saved q-table to {q_path.resolve()}")
    print(f"saved metrics to {metrics_path.resolve()}")
    print(json.dumps({"train_metrics": train_metrics, "eval_metrics": eval_metrics}, indent=2))

    env.close()
    return train_metrics, eval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared tabular Q-learning baseline for environment_petting_zoo."
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--stars", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="pettingzoo_runs")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
