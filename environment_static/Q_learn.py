from rrps_core.Q_learn import RRPSQLearnCore

from environment_static.rrps_gym import Observation


class QLearnStatic(RRPSQLearnCore[Observation]):

    def hash(self, obs):
        agent = obs["player_dict"][0]
        initial = self.env.initial_player_budget
        opponents = sorted(
            ((pid, p) for pid, p in obs["player_dict"].items() if pid != 0),
            key=lambda x: x[0],
        )
        opponent_state = tuple(
            (
                p["stars_total"],
                initial["rock_total"] - p["rock_total"],
                initial["paper_total"] - p["paper_total"],
                initial["scissors_total"] - p["scissors_total"],
            )
            for _, p in opponents
        )

        return (
            agent["stars_total"],
            agent["rock_total"],
            agent["paper_total"],
            agent["scissors_total"],
            opponent_state,
        )
