import random
from typing import List, Tuple

from .base import Node, Graph
from gym_PBN.utils.converters import logic_funcs_to_PBN_data
from gym_PBN.utils.logic.eval import LogicExpressionEvaluator


class PBNNode(Node):
    def __init__(self, index, data, genes):
        input_mask, truth_table, node, _ = data
        super(PBNNode, self).__init__(index, -1, node, node, False)

        self.input_mask = input_mask
        self.truth_table = truth_table

        self.predictors = []
        for i in range(len(self.input_mask)):
            if self.input_mask[i]:
                self.predictors.append(i)

    def step(self, state, verbose=False):
        relevant_nodes = tuple(state[i] for i in self.predictors)
        proba = self.truth_table[relevant_nodes]
        self.value = int(random.random() < proba)
        return self.value


class PBNGraph(Graph):
    def __init__(self, genes, logic_functions):
        super(PBNGraph, self).__init__(2)
        self.genes = genes
        pbn_data = logic_funcs_to_PBN_data(self.genes, logic_functions)

        for i, data in enumerate(pbn_data):
            print(f'  spawning node {i}')
            self.nodes.append(PBNNode(i, data, self.genes))

    def step(self, changed_nodes: list = None, i=None):
        oldState = self.getState()
        i = random.randint(0, len(self.nodes) - 1) if i is None else i
        # while i in changed_nodes:
        #     i = random.randint(0, len(self.nodes) - 1)
        self.nodes[i].step(oldState)
        return self.getState()
