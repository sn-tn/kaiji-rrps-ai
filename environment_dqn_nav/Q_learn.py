from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback
from rrps_core.rrps_gym import RRPSEnvCore
from typing import Generator
import rrps_core.visualizer as vis


class QLearnDQNNav:
    def __init__(self, env: RRPSEnvCore, agent_name: str | None = None):
        self.env = env
        self.agent_name = agent_name
        self._model: DQN | None = None

    def train(
        self,
        total_timesteps: int = 2_000_000,
        gamma: float = 0.9,
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 2_000,
        batch_size: int = 64,
        target_update_interval: int = 500,
        exploration_fraction: float = 0.5,
        exploration_final_eps: float = 0.05,
        hidden_size: int = 128,
        agent_name: str | None = None,
    ):
        name = agent_name or self.agent_name
        if name is None:
            raise ValueError("agent_name must be provided")
        self._model = DQN(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=dict(net_arch=[hidden_size, hidden_size]),
        )
        self._model.learn(
            total_timesteps=total_timesteps, callback=ProgressBarCallback()
        )
        self._model.save(name)
        self.agent_name = name
        return self

    def load(self, path: str):
        self._model = DQN.load(path, env=self.env)
        self.agent_name = path
        return self

    def play_agent(
        self, gui: bool = False
    ) -> Generator[tuple, None, None]:
        if self._model is None:
            raise ValueError("No model loaded")
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            yield obs, reward, terminated, truncated, info
            if gui:
                vis.refresh(terminated, truncated, info)
