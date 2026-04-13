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
QTable = Dict[StateKey, np.ndarray]


def state_key(observation: dict) -> StateKey:
    return tuple(int(x) for x in observation["observation"])


def legal_actions(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def ensure_state(table: QTable, key: StateKey, action_size: int) -> None:
    if key not in table:
        table[key] = np.zeros(action_size, dtype=np.float32)


def choose_action(
    q_tables: Dict[str, QTable],
    agent: str,
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
    table = q_tables[agent]
    ensure_state(table, key, action_size)

    if rng.random() < epsilon:
        return int(rng.choice(legal))

    q_values = table[key]
    legal_q = q_values[legal]
    best = np.max(legal_q)
    best_legal = legal[legal_q == best]
    return int(rng.choice(best_legal))


def max_next_q(
    q_tables: Dict[str, QTable],
    agent: str,
    next_obs: dict,
    action_size: int,
) -> float:
    mask = next_obs["action_mask"]
    legal = legal_actions(mask)
    if legal.size == 0:
        return 0.0

    key = state_key(next_obs)
    table = q_tables[agent]
    ensure_state(table, key, action_size)
    return float(np.max(table[key][legal]))


def evaluate(
    q_tables: Dict[str, QTable],
    n_players: int,
    episodes: int,
    max_rounds: int,
    seed: int,
) -> dict:
    env = parallel_env(n_players=n_players, max_rounds=max_rounds)
    rng = np.random.default_rng(seed)
    rewards = []
    per_agent_rewards = defaultdict(list)
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
                ensure_state(q_tables[agent], key, env._action_size)
                q_values = q_tables[agent][key]
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
        for agent in env.possible_agents:
            per_agent_rewards[agent].append(float(episode_reward.get(agent, 0.0)))
        avg_rounds.append(env.round_count)

    env.close()
    return {
        "episodes": episodes,
        "mean_reward_per_agent": float(np.mean(rewards) if rewards else 0.0),
        "mean_rounds": float(np.mean(avg_rounds) if avg_rounds else 0.0),
        "cleared_game_events": int(clears),
        "last_agent_standing_events": int(last_agent_standing),
        "mean_reward_by_agent": {
            agent: float(np.mean(values) if values else 0.0)
            for agent, values in per_agent_rewards.items()
        },
    }


def train(args: argparse.Namespace) -> tuple[dict, dict]:
    env = parallel_env(
        n_players=args.players,
        max_rounds=args.max_rounds,
        stars=args.stars,
        render_mode="human" if args.render else None,
    )
    rng = np.random.default_rng(args.seed)
    q_tables: Dict[str, QTable] = {
        f"player_{i}": {} for i in range(args.players)
    }

    epsilon = args.epsilon_start
    rewards_history = []
    round_history = []
    per_agent_training_rewards = defaultdict(list)

    for episode in range(1, args.episodes + 1):
        observations, infos = env.reset(seed=args.seed + episode)
        episode_reward = defaultdict(float)

        while env.agents:
            prev_obs = {agent: observations[agent] for agent in env.agents}
            actions = {}
            for agent in env.agents:
                actions[agent] = choose_action(
                    q_tables=q_tables,
                    agent=agent,
                    obs=observations[agent],
                    epsilon=epsilon,
                    action_size=env._action_size,
                    rng=rng,
                )

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent, prev in prev_obs.items():
                table = q_tables[agent]
                s_key = state_key(prev)
                ensure_state(table, s_key, env._action_size)
                a = actions[agent]
                r = float(rewards.get(agent, 0.0))
                episode_reward[agent] += r
                terminal = bool(terminations.get(agent, False) or truncations.get(agent, False))

                if terminal or agent not in next_observations:
                    target = r
                else:
                    target = r + args.gamma * max_next_q(
                        q_tables=q_tables,
                        agent=agent,
                        next_obs=next_observations[agent],
                        action_size=env._action_size,
                    )

                q_old = table[s_key][a]
                table[s_key][a] = q_old + args.alpha * (target - q_old)

            observations = next_observations

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        rewards_history.append(sum(episode_reward.values()) / args.players)
        round_history.append(env.round_count)
        for agent in env.possible_agents:
            per_agent_training_rewards[agent].append(float(episode_reward.get(agent, 0.0)))

        if args.log_every and episode % args.log_every == 0:
            recent_rewards = rewards_history[-args.log_every :]
            recent_rounds = round_history[-args.log_every :]
            states_by_agent = {
                agent: len(table) for agent, table in q_tables.items()
            }
            print(
                f"episode={episode} "
                f"epsilon={epsilon:.4f} "
                f"avg_reward_per_agent={np.mean(recent_rewards):.3f} "
                f"avg_rounds={np.mean(recent_rounds):.2f} "
                f"states_by_agent={states_by_agent}"
            )

    states_by_agent = {agent: len(table) for agent, table in q_tables.items()}
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
        "q_table_states_by_agent": states_by_agent,
        "q_table_states_total": int(sum(states_by_agent.values())),
        "final_epsilon": float(epsilon),
        "mean_training_reward_per_agent": float(np.mean(rewards_history) if rewards_history else 0.0),
        "mean_training_rounds": float(np.mean(round_history) if round_history else 0.0),
        "mean_training_reward_by_agent": {
            agent: float(np.mean(values) if values else 0.0)
            for agent, values in per_agent_training_rewards.items()
        },
    }

    eval_metrics = evaluate(
        q_tables=q_tables,
        n_players=args.players,
        episodes=args.eval_episodes,
        max_rounds=args.max_rounds,
        seed=args.seed + 100_000,
    )

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"independent_q_p{args.players}_ep{args.episodes}_seed{args.seed}"
    q_path = out_dir / f"{stem}.pkl"
    metrics_path = out_dir / f"{stem}_metrics.json"

    with q_path.open("wb") as f:
        pickle.dump(q_tables, f, protocol=pickle.HIGHEST_PROTOCOL)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_metrics, "eval_metrics": eval_metrics}, f, indent=2)

    print(f"saved q-tables to {q_path.resolve()}")
    print(f"saved metrics to {metrics_path.resolve()}")
    print(json.dumps({"train_metrics": train_metrics, "eval_metrics": eval_metrics}, indent=2))

    env.close()
    return train_metrics, eval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent tabular Q-learning baseline for environment_petting_zoo."
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
