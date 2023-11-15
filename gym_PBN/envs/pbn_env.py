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

        self.attractor_count = len(self.all_attractors)
        self.probabilities = [1 / self.attractor_count] * self.attractor_count

        print(self.all_attractors)

        # self.target_nodes = sorted(self.includeIDs)
        # self.target_node_values = self.all_attractors[-1]

    def statistical_attractors(self):
        print(f"Calculating state statistics for N = {self.N}")
        print(f"it should take {10 ** 4} steps")
        state_log = defaultdict(int)

        self.setTarget([[0] * self.N])

        steps = 1000
        simulations = 10 ** 4
        for i in range(simulations):
            if i % 10 ** 3 == 0:
                print(i)
            s = [random.randint(0, 1) for _ in range(self.N)]
            self.graph.setState(s)
            for j in range(steps):
                state = tuple(self.render())
                state_log[state] += 1
                _ = self.step([], force=True)

        states = sorted(state_log.items(), key=lambda kv: kv[1], reverse=True)

        statistial_attractors = [node for node, frequency in states if frequency > 0.15 * steps * simulations]

        if len(statistial_attractors) < 10:
            statistial_attractors = [node for node, frequency in states if frequency > 1000]

        if len(statistial_attractors) < 10:
            statistial_attractors = [node for node, frequency in states[:10]]

        print(f"got {statistial_attractors}")
        return statistial_attractors

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attracting_states
