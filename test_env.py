import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environment.rps_gym import RestrictedRPSEnv

# action labels for logging
_ACTION_LABELS = {
    0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT",
    4: "ROCK", 5: "PAPER", 6: "SCISSORS",
}

def run_episode(env: RestrictedRPSEnv, policy: str = "random", verbose: bool = True) -> dict:
    """
    Runs a single episode and returns summary stats.

    policy:
        "random"  — sample uniformly from all 7 actions
        "legal"   — sample from movement actions + non-exhausted RPS moves only
    """
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    if verbose:
        print(f"\n{'='*50}")
        print(f"Episode start  (policy={policy})")
        print(f"{'='*50}")

    while True:
        if policy == "legal":
            # movement actions are always valid
            legal_actions = list(env._MOVE_ACTIONS.keys())
            # add RPS actions for non-exhausted moves
            for rps_action, move in env._RPS_ACTIONS.items():
                if move in env._agent.available_moves():
                    legal_actions.append(rps_action)
            action = int(np.random.choice(legal_actions))
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            label = _ACTION_LABELS.get(action, str(action))
            print(f"  Step {steps:>3} | action={label:<8} reward={reward:+.1f} "
                  f"total={total_reward:+.1f} | alive={obs['opponents_alive']}")

        if terminated or truncated:
            break

    result = info.get("result", "unknown")
    if verbose:
        print(f"\n  Result: {result.upper()}  |  steps={steps}  total_reward={total_reward:+.1f}")

    return {"result": result, "steps": steps, "total_reward": total_reward}


def run_suite(n_episodes: int = 100, policy: str = "legal", **env_kwargs):
    """Runs multiple quiet episodes and prints aggregate stats."""
    env = RestrictedRPSEnv(**env_kwargs)

    results = {"victory": 0, "eliminated": 0, "unknown": 0}
    all_rewards = []
    all_steps = []

    for _ in range(n_episodes):
        stats = run_episode(env, policy=policy, verbose=False)
        results[stats["result"]] += 1
        all_rewards.append(stats["total_reward"])
        all_steps.append(stats["steps"])

    env.close()

    print(f"\n{'='*50}")
    print(f"Suite results  ({n_episodes} episodes, policy={policy})")
    print(f"{'='*50}")
    print(f"  Victories  : {results['victory']:>4}  ({results['victory']/n_episodes*100:.1f}%)")
    print(f"  Eliminated : {results['eliminated']:>4}  ({results['eliminated']/n_episodes*100:.1f}%)")
    print(f"  Avg reward : {np.mean(all_rewards):>+.2f}  (std={np.std(all_rewards):.2f})")
    print(f"  Avg steps  : {np.mean(all_steps):>.1f}  (std={np.std(all_steps):.2f})")


def check_spaces(env: RestrictedRPSEnv):
    """Verifies observations always stay within declared bounds."""
    violations = 0
    for _ in range(20):
        obs, _ = env.reset()
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if not env.observation_space.contains(obs):
                print(f"  [!] Obs out of bounds: {obs}")
                violations += 1
            if terminated or truncated:
                break

    print(f"\n{'='*50}")
    print("Observation space check (20 episodes)")
    print(f"{'='*50}")
    print(f"  Violations: {violations}")


if __name__ == "__main__":
    # ── 1. single verbose episode to visually inspect behaviour ──────────────
    env = RestrictedRPSEnv(n_opponents=3, stars=3, budget=4, render_mode="human")
    run_episode(env, policy="legal", verbose=True)
    env.close()

    # ── 2. observation space sanity check ────────────────────────────────────
    env = RestrictedRPSEnv(n_opponents=3, stars=3, budget=4)
    check_spaces(env)
    env.close()

    # ── 3. aggregate stats over 200 episodes ─────────────────────────────────
    run_suite(n_episodes=200, policy="legal", n_opponents=3, stars=3, budget=4)