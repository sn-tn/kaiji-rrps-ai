from __future__ import annotations

from pathlib import Path

import numpy as np

from environment_petting_zoo.rrps_pz import parallel_env


def sample_legal_action(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    return int(np.random.choice(legal))


def main() -> None:
    env = parallel_env(n_players=4, max_rounds=25, render_mode="human")
    observations, infos = env.reset(seed=42)

    while env.agents:
        actions = {
            agent: sample_legal_action(observations[agent]["action_mask"])
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("rewards:", rewards)
        print("terminations:", terminations)
        print("truncations:", truncations)

    out_path = Path("pettingzoo_random_episode.json")
    env.save_episode(out_path)
    print(f"saved episode log to {out_path.resolve()}")
    env.close()


if __name__ == "__main__":
    main()
