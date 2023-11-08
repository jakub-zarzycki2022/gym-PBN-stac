import math
import random
from collections import defaultdict

import pandas as pd

from .bittner import base
from gym_PBN.envs.pbn_target_multi import PBNTargetMultiEnv


def graph_from_predictors(df):
    graph = base.Graph(2)  # BN = base 2 values

    nodes = []
    for i, row in df.iterrows():
        promoters = row["Promoters"]
        inhibitors = row["Inhibitors"]

        # check for nans
        promoters = promoters.replace(' ', '').split(',')
        inhibitors = inhibitors.replace(' ', '').split(',')

        promoters = [] if '' in promoters else promoters
        inhibitors = [] if '' in inhibitors else inhibitors

        predictors = promoters + inhibitors
        for p in predictors:
            if ' ' in p:
                print(predictors)
                raise ValueError
        A = [1] * len(promoters) + [-1] * len(inhibitors)
        A.append(1)

        node = base.Node(i, -1, row["Node"], row["Node"])
        node.add_predictors([(1, A, predictors)])
        nodes.append(node)

    graph.add_nodes(nodes)
    return graph


class PBNGTEx(PBNTargetMultiEnv):

    NAME = "GTEx-72"

    def __init__(
        self,
        N=72,
        render_mode: str = "human",
        render_no_cache: bool = False,
        name: str = NAME,
        horizon: int = 100,
        end_episode_on_success: bool = True,
    ):
        self.N = N
        print(f"its me, gtex-{self.N}")
        if not name:
            name = self.NAME

        df = pd.read_excel(io='~/Downloads/pnas.1722609115.sd02.xlsx')
        df.fillna('', inplace=True)
        graph = graph_from_predictors(df)

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
