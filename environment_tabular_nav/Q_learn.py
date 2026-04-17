from gym_core.Q_learn import RRPSQLearnCore
from environment_tabular_nav.rps_gym import Observation
import numpy as np


class QLearnTabularNav(RRPSQLearnCore[Observation]):
    def hash(self, obs):
        agent = obs["player_dict"][0]
        agent_pos = agent["position"]
        initial = self.env.initial_player_budget

        nearest_pid = self.env._nearest(0, self.env._alive_opponents())
        if nearest_pid is not None:
            opp_state = self.env.player_dict[nearest_pid]
            opp_pos = opp_state["position"]
            rel_dx = int(np.sign(opp_pos[0] - agent_pos[0]))
            rel_dy = int(np.sign(opp_pos[1] - agent_pos[1]))
            nearest_state = (
                opp_state["stars_total"],
                initial["rock_total"] - opp_state["rock_total"],
                initial["paper_total"] - opp_state["paper_total"],
                initial["scissors_total"] - opp_state["scissors_total"],
            )
        else:
            rel_dx, rel_dy = 0, 0
            nearest_state = (0, 0, 0, 0)

        return (
            agent["stars_total"],
            agent["rock_total"],
            agent["paper_total"],
            agent["scissors_total"],
            nearest_state,
            rel_dx,
            rel_dy,
        )
