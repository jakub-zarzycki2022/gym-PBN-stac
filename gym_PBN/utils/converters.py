from networkx.algorithms.dag import transitive_reduction


import itertools

import numpy as np

from .logic.eval import LogicExpressionEvaluator

def logic_funcs_to_PBN_data(nodes: list[str], node_functions: list[tuple[str, int]]):
    logic_eval = LogicExpressionEvaluator({})  # Don't need a value dict yet
    PBN_data = []

    for i, node in enumerate(nodes):
        # Input Mask
        input_mask = np.zeros(len(nodes), dtype=bool)
        for function, _ in node_functions[i]:
            symbols = logic_eval.get_symbols(function)
            for symbol in symbols:
                j = nodes.index(symbol)
                input_mask[j] = True

        # Truth Table
        truth_table = np.zeros([2] * sum(input_mask))
        all_states = itertools.product([0, 1], repeat=sum(input_mask))
        input_nodes = np.array(nodes)[input_mask]

        for state in all_states:
            for function, prob in node_functions[i]:
                logic_eval.dictionary = {node: value for node, value in zip(input_nodes, state)}
                value = int(logic_eval.evaluate(function))
                if value == 1:
                    truth_table[state] += prob

        control = sum(input_mask) == 0
        
        PBN_data.append((input_mask, truth_table, node, control))

    return PBN_data
