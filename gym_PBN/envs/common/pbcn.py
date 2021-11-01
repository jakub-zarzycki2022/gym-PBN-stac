import itertools
from typing import List, Union

import numpy as np
from gym_PBN.types import LOGIC_FUNC_DATA, PBN_DATA

from .node import Node
from .pbn import PBN


class PBCN(PBN):
    def __init__(
        self, PBN_data: PBN_DATA = [], logic_func_data: LOGIC_FUNC_DATA = None
    ):
        if len(PBN_data) == 0:
            PBN_data = self._logic_funcs_to_pbn_data(logic_func_data)

        # Filter nodes
        nodes, control_nodes = [], []
        for node in PBN_data:
            if node[3]:
                control_nodes.append(node)
            else:
                nodes.append(node)

        # Init the non-control part
        self._init_from_pbn_data(nodes)

        # Init control part
        self.M = len(control_nodes)
        self.control_nodes = np.array(
            [Node(*node_data) for node_data in control_nodes], dtype=object
        )
        self.control_state = np.empty((self.M), dtype=bool)

    def apply_control(self, control: List[Union[int, bool]]):
        if len(control) != len(self.control_nodes):
            raise ValueError(
                f"Control for {len(control)} control nodes provided, when there are {len(self.control_nodes)} in the network."
            )

        if type(control) != np.ndarray:
            control = np.array(control, dtype=bool)

        self.control_state = control

    def step(self):
        # Huge assumption: all control nodes are at the start of the input mask.
        combined_state = np.concatenate((self.control_state, self.state))
        self.state = np.array(
            [node.compute_next_value(combined_state) for node in self.nodes], dtype=bool
        )

    def reset(self, state: Union[List[Union[int, bool]], np.ndarray, None] = None):
        self.control_state = np.zeros((self.M), dtype=bool)
        return super().reset(state=state)

    def _compute_next_states(self, state):
        # Huge assumption: all control nodes are at the start of the input mask.
        combined_state = np.concatenate((self.control_state, state))

        probabilities = np.zeros((2, self.N), dtype=float)

        output = []
        for i in range(self.N):
            prob_true = self.nodes[i].get_next_value_prob(combined_state)
            probs = np.array([1 - prob_true, prob_true])
            probabilities[:, i] = probs

        prob_to_states = self._probs_to_states(probabilities)

        for prostate, proprob in prob_to_states:
            output.append((str(state.astype(int)), str(prostate.astype(int)), proprob))

        return output

    @property
    def control_actions(self):
        return map(list, itertools.product([0, 1], repeat=self.M))
