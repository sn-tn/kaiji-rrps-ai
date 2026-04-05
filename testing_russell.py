import gymnasium as gym
from environment.rps_gym import RestrictedRPSEnv
import environment.vis_rps as vis

env = RestrictedRPSEnv(n_opponents=50, lives=3, budget=4)
vis.game = env
vis.setup()

obs, info = env.reset(seed=0)

terminated = False
total_reward = 0.0

while not terminated:
    action = env.action_space.sample()  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    vis.refresh(obs, reward, terminated, info, delay=0.1)

print(
    f"Episode over. Result: {info.get('result')} | Total reward:"
    f" {total_reward}"
)
env.close()
