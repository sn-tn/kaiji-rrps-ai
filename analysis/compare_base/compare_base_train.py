import sys, os, time, json
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from environment_static.Q_learn import QLearnStatic
from environment_static.rrps_gym import StaticRRPSEnv
from environment_tabular_nav.Q_learn import QLearnTabularNav
from environment_tabular_nav.rps_gym import RestrictedRPSEnv as TabularEnv
from environment_dqn_nav.Q_learn import QLearnDQNNav
from environment_dqn_nav.rrps_gym import RestrictedRPSEnv as DQNEnv
from rrps_core.reward_config import RewardConfig

DECAY_RATE = 0.999
N_OPPONENTS = 10
STARS = 3
GRID_SIZE = 12
BUDGET = {"rock_total": 3, "paper_total": 3, "scissors_total": 3}
NAV_REWARD_CONFIG = RewardConfig(
    win_matchup=100,
    lose_matchup=-100,
    tie_matchup=0,
    eliminated=-2000,
    victory=2000,
    invalid_move=-10,
    within_challenge_range=1,
    approach_opponent=0.5,
)

# ── Static Q-Learn ────────────────────────────────────────────────────────────

STATIC_EPISODES = 20_000
static_env = StaticRRPSEnv(n_opponents=N_OPPONENTS, agent_budget=BUDGET, player_budget=BUDGET)
static_agent = QLearnStatic(agent_name=os.path.join(HERE, "compare_static"), env=static_env)
t0 = time.time()
static_agent.tabular_train(gamma=0.9, train_episodes=STATIC_EPISODES, decay_rate=DECAY_RATE)
static_train_time = (time.time() - t0) / (STATIC_EPISODES / 10_000)

# ── Tabular Nav ───────────────────────────────────────────────────────────────

TABULAR_EPISODES = 20_000
tabular_env = TabularEnv(
    n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
    agent_budget=BUDGET, player_budget=BUDGET, reward_config=NAV_REWARD_CONFIG,
)
tabular_agent = QLearnTabularNav(agent_name=os.path.join(HERE, "compare_tabular"), env=tabular_env)
t0 = time.time()
tabular_agent.tabular_train(gamma=0.9, train_episodes=TABULAR_EPISODES, decay_rate=DECAY_RATE)
tabular_train_time = (time.time() - t0) / (TABULAR_EPISODES / 10_000)

# ── DQN Nav ───────────────────────────────────────────────────────────────────

DQN_TIMESTEPS = 1_000_000
dqn_env = DQNEnv(
    n_opponents=N_OPPONENTS, stars=STARS, grid_size=GRID_SIZE,
    n_obs_opponents=4, agent_budget=BUDGET, player_budget=BUDGET, reward_config=NAV_REWARD_CONFIG,
)
dqn_agent = QLearnDQNNav(agent_name=os.path.join(HERE, "compare_dqn"), env=dqn_env)
t0 = time.time()
dqn_agent.train(total_timesteps=DQN_TIMESTEPS)
dqn_train_time = (time.time() - t0) / (DQN_TIMESTEPS / 10_000)

# ── Save train times ──────────────────────────────────────────────────────────

with open(os.path.join(HERE, "train_times.json"), "w") as f:
    json.dump({
        "static": static_train_time,
        "tabular": tabular_train_time,
        "dqn": dqn_train_time,
    }, f)

print("Training complete. Run compare_base_eval.py to evaluate and plot.")
