import math
import random
from collections import defaultdict

import pandas as pd

from gym_PBN.envs.bittner.pbn_graph import PBNNode, PBNGraph
from gym_PBN.envs.pbn_target_multi import PBNTargetMultiEnv


class PBNEnv(PBNTargetMultiEnv):

    NAME = "PBN"

    def __init__(
        self,
        N=72,
        render_mode: str = "human",
        render_no_cache: bool = False,
        name: str = NAME,
        horizon: int = 100,
        end_episode_on_success: bool = True,
        logic_functions=None,
        genes=None,
    ):
        self.N = N
        print(f"its me, PBN-{self.N}")
        if not name:
            self.NAME = f"{self.NAME}-{N}"
            name = self.NAME

        graph = PBNGraph(genes, logic_functions)

        super().__init__(
            graph,
            {},
            render_mode,
            render_no_cache,
            name,
            None,
            end_episode_on_success,
        )

        self.all_attractors = [[s] for s in self.statistical_attractors()]
        self.attracting_states.update([s[0] for s in self.all_attractors])

        self.attractor_count = len(self.all_attractors)
        self.probabilities = [1 / self.attractor_count] * self.attractor_count

        print(self.all_attractors)

        # self.target_nodes = sorted(self.includeIDs)
        # self.target_node_values = self.all_attractors[-1]

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attracting_states
