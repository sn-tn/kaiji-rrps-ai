import gymnasium as gym
from environment.rps_gym import RestrictedRPSEnv, Observation
import environment.vis_rps as vis
from tqdm import tqdm
import pickle
import numpy as np
import sys

env = RestrictedRPSEnv(n_opponents=4, stars=3, budget=4, grid_size=5)

BOLD = "\033[1m"  # ANSI escape sequence for bold text
RESET = "\033[0m"  # ANSI escape sequence to reset text formatting

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv
if gui_flag:
    vis.game = env
    vis.setup()
obs, info = env.reset(seed=0)

terminated = False
total_reward = 0.0


def hash(obs: Observation) -> tuple:
    ag = obs["agent"]
    opp = obs["opponent"]
    manhattan_dist = abs(ag["position"][0] - opp["position"][0]) + abs(
        ag["position"][1] - opp["position"][1]
    )
    key = (
        ag["stars"],
        ag["budget"]["rock"],
        ag["budget"]["paper"],
        ag["budget"]["scissors"],
        opp["stars"],
        opp["budget"]["rock"],
        opp["budget"]["paper"],
        opp["budget"]["scissors"],
        manhattan_dist
        <= 1,  # within 1 of agent
        obs["opponents_alive"],
    )
    # print("key", key)
    return key


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    Q_update_counts = {}

    for _ in tqdm(range(num_episodes)):
        start_obs = env.reset()
        prev_state_key = hash(start_obs[0])
        if prev_state_key not in Q_update_counts:
            Q_update_counts[prev_state_key] = np.zeros(env.action_space.n)
        if prev_state_key not in Q_table:
            Q_table[prev_state_key] = np.zeros(env.action_space.n)
        while True:
            # initialize prev state keys if not already
            action = None

            # want random action w/ prob P = epsilon
            if np.random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[prev_state_key])

            # transition to state s'
            new_obs, reward, terminated, truncated, info = env.step(action)
            if gui_flag:
                vis.refresh(obs, reward, terminated, info, delay=0.1)
            new_state_key = hash(new_obs)

            ## initalize state action if not already
            if new_state_key not in Q_update_counts:
                Q_update_counts[new_state_key] = np.zeros(env.action_space.n)
            if new_state_key not in Q_table:
                Q_table[new_state_key] = np.zeros(env.action_space.n)

            # calculate Q vals
            Q_old_update_counts = Q_update_counts[prev_state_key][action]
            Q_old = Q_table[prev_state_key][action]
            V_opt_old = np.max(Q_table[new_state_key])
            eta = 1 / (1 + Q_old_update_counts)
            Q_new = (1 - eta) * Q_old + eta * (reward + gamma * V_opt_old)

            # update table
            Q_table[prev_state_key][action] = Q_new
            Q_update_counts[prev_state_key][action] += 1

            # update epsilon and end or continue w/ new step as prev
            if terminated:
                epsilon *= decay_rate
                break
            else:
                prev_state_key = new_state_key
    return Q_table


"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""

num_episodes = 1_000_000
decay_rate = 0.999999
if train_flag:
    Q_table = Q_learning(
        num_episodes=num_episodes,
        gamma=0.9,
        epsilon=1,
        decay_rate=decay_rate,
    )  # Run Q-learning

    # Save the Q-table dict to a file
    with open(
        "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle",
        "wb",
    ) as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Completed", num_episodes, decay_rate)


def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)


if not train_flag:

    rewards = []

    filename = (
        "Q_table_" + str(num_episodes) + "_" + str(decay_rate) + ".pickle"
    )
    input(
        f"\n{BOLD}Currently loading Q-table from "
        + filename
        + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load"
        " a different Q-table file.\n(set num_episodes and decay_rate in"
        " Q_learning.py)."
    )
    with open(filename, "rb") as f:
        Q_table = pickle.load(f)
    print(
        f"Q_table: {len(Q_table)} states, {next(iter(Q_table.values())).shape}"
        " actions per state"
    )
    for episode in tqdm(range(10000)):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            state = hash(obs)
            try:
                action = np.random.choice(
                    env.action_space.n, p=softmax(Q_table[state])
                )  # Select action using softmax over Q-values
            except KeyError:
                action = (
                    env.action_space.sample()
                )  # Fallback to random action if state not in Q-table

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            if gui_flag:
                vis.refresh(
                    obs, reward, terminated, info, delay=0.1
                )  # Update the game screen [GUI only]

        # print("Total reward:", total_reward)
        rewards.append(total_reward)
    avg_reward = sum(rewards) / len(rewards)
    print("avg_reward", avg_reward)
