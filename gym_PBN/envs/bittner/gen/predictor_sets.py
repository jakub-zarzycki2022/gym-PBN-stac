import itertools
import pickle
from pathlib import Path

import os
import psutil

import numpy as np
import pandas as pd
from tqdm import tqdm

PROCESS = psutil.Process(os.getpid())


def generate_predictor_sets(
    gene_data: pd.DataFrame,
    k=3,
    n_predictors=5,
    savepath="predictor_sets.pkl",
):
    gene_data = gene_data.drop("Name", axis=1)
    n_samples = len(gene_data.columns)

    predictor_sets = []

    # Load if exists
    if Path(savepath).exists():
        return pickle.load((open(savepath, "rb")))

    # Otherwise generate
    pbar = tqdm(
        gene_data.index.unique(),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}, {postfix}M]",
    )
    for gene in pbar:  # O(N)
        pbar.set_description(f"Calculating top {n_predictors} predictors for {gene}")
        pbar.set_postfix({"Memory usage": PROCESS.memory_info().rss / 1024**2})

        # Delete it from the data being looked at
        temp_data = gene_data.drop(gene)
        buff = np.empty((3, n_predictors), dtype=object)

        # All possible predictor combinations - O(k * (n choose k))
        for combination in itertools.combinations(temp_data.index.unique(), k):
            n_genes = len(combination)
            comb_gene_states = [
                np.atleast_2d(temp_data.loc[_gene]) for _gene in combination
            ]

            # List of indices for individual states for each gene in the combination
            n_combs = [list(range(len(states))) for states in comb_gene_states]

            # sample, gene, state_combo
            x_combos = np.empty((n_samples, n_genes, 0))

            for gene_state_combo in itertools.product(*n_combs):
                states = np.array(
                    [comb_gene_states[i][j] for i, j in enumerate(gene_state_combo)]
                )
                # sample, gene, value
                x = states.T.reshape(-1, n_genes, 1)
                x_combos = np.append(x_combos, x, axis=2)

            y = np.atleast_2d(gene_data.loc[gene])
            for _exp in y:
                _y = np.expand_dims(_exp, axis=1)
                for state_combo_idx in range(x_combos.shape[2]):
                    x = x_combos[:, :, state_combo_idx]
                    COD, A = gen_COD(x, _y)
                    add_to_buff(buff, (COD, A, combination))

        predictor_sets.append(buff)

    # Save
    pickle.dump(predictor_sets, open(savepath, "wb"))

    return predictor_sets


def add_to_buff(buff, data):
    n_predictors = buff.shape[1]
    COD, _, _ = data

    i = 0
    while True:
        # If there's an empty slot in the buffer
        if buff[0, i] == None:
            buff[:, i] = data  # Just add it
            break
        elif buff[0, i] < COD:  # If this is a better predictor
            temp = np.copy(buff[:, i])  # Copy existing data
            buff[:, i] = data  # Overwrite
            while i < n_predictors - 1:  # Move everything to the right
                temp2 = np.copy(buff[:, i + 1])
                buff[:, i + 1] = temp
                temp = temp2
                i += 1
            break
        elif i == n_predictors - 1:  # Stop if this is the last slot
            break
        else:  # Else go next
            i += 1
            # (?) Stop if this is the penultimate slot
            if i == n_predictors - 2:
                break


def gen_COD(X, Y):
    """Schmulevich's (?) method for generating a COD. The closed form solution."""

    def MSE(x, y):
        e = (x - y) ** 2 / x.shape[0]
        return e[0]

    def g(x):
        return np.array(x >= 0.5, dtype=int)

    ones = np.ones(Y.shape)
    X = np.append(X, ones, axis=1)
    R = np.matmul(X.T, X)
    Rp = np.linalg.pinv(R)
    C = np.matmul(X.T, Y)
    A = np.matmul(Rp, C)  # for comparison

    y_pred = g(np.matmul(X, A))
    y_pred_null = g(ones * np.mean(Y)) + 10**-8
    e_null = MSE(y_pred_null, Y)

    e = MSE(y_pred, Y)
    COD = (e_null - e) / e_null
    if COD < 0:
        COD = 10**-8
    return COD, A
