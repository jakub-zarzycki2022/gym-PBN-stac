import os
import subprocess
from collections import defaultdict

import numpy as np
from itertools import product
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.stats import logistic
from sympy import symbols
from sympy.logic import SOPform

import networkx as nx
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_PBN


def translate(logic_function):
    """
    We need variable names to start with letter.

    """
    logic_function = logic_function.replace('(', "( ")
    tokens = logic_function.split(" ")
    for i, token in enumerate(tokens):
        # ~ -> not
        if token[0] == '~':
            tokens[i] = f"~x{token[1:]}"
        # | -> or
        elif token[0] in {'|', '&', '('}:
            pass
        # {num} -> x{num}
        else:
            tokens[i] = f"x{token}"
    res = " ".join(tokens)
    return res


def get_model(env):
    model_out_path = f"kaban/results/model_{env.name}.out"
    if os.path.isfile(model_out_path):
        print(f"reloading model form {model_out_path}")
        with open(f"{model_out_path}") as f:
            return f.read()

    g = env.graph

    jinja_env = Environment(
        loader=FileSystemLoader(searchpath="./"),
        autoescape=select_autoescape()
    )

    truth_tables = defaultdict(list)

    for node in g.nodes:
        predictors = node.predictors

        for predictor in predictors:
            IDs, A, _ = predictor
            num_genes = len(IDs)

            # matrix of # of len(state) + 1 x # of states
            truth_table = np.zeros((num_genes + 1 + 1, 2 ** (num_genes + 1)))
            for j, state in enumerate(product([0, 1], repeat=num_genes+1)):
                x = np.ones(num_genes + 1)
                for i in range(len(state)):
                    x[i] = state[i]
                    truth_table[i][j] = state[i]
                truth_table[num_genes+1][j] = 1 if np.dot(x, A) >= 0 else 0

            truth_tables[node.ID].append((IDs, truth_table))

    log_funcs = defaultdict(list)

    for gen in truth_tables:
        tts = truth_tables[gen]
        lf = []
        for IDs, tt in tts:
            minterms = [list(x)[:-1] for x in tt.T if list(x)[-1]]
            pred_ids = list(IDs)
            pred_ids.append(gen)

            if len(pred_ids) == 1:
                sym = symbols(pred_ids)
            else:
                sym = symbols(",".join([str(x) for x in pred_ids]))

            fun = str(SOPform(sym, minterms, []))
            print(fun)
            if fun == 'True':
                fun = f'{gen} | ~{gen}'
            lf.append((translate(fun)))
        log_funcs[gen] = lf

    template = jinja_env.get_template("model_template.jj2")

    out = template.render(log_funcs=log_funcs)

    with open(f"models/model_{env.name}.ispl", "w+") as f:
        f.write(out)

    out = subprocess.run(["cabean", f"models/model_{env.name}.ispl"], capture_output=True, encoding='utf-8')

    print(f"saving model to {model_out_path}")
    with open(f"{model_out_path}", "w") as f:
        f.write(out.stdout)

    return out.stdout
