import itertools
import multiprocessing
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from tqdm.contrib.concurrent import process_map


def generate_predictor_sets(
    gene_data: pd.DataFrame,
    k=3,
    n_predictors=5,
    savepath="predictor_sets.pkl",
):
    gene_data = gene_data.drop("Name", axis=1)
    n_samples = len(gene_data.columns)

    # Load if exists
    if Path(savepath).exists():
        return pickle.load((open(savepath, "rb")))

    # Otherwise generate
    genes = gene_data.index.unique()
    _func = partial(_gen_predictor_sets_gene, gene_data, n_samples, n_predictors, k)

    predictor_sets = process_map(
        _func,
        genes,
        max_workers=multiprocessing.cpu_count(),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]",
    )

    # Save
    pickle.dump(predictor_sets, open(savepath, "wb"))

    return predictor_sets


def _gen_predictor_sets_gene(gene_data, n_samples, n_predictors, k, gene):
    # Data processing
    buff = np.empty((3, n_predictors), dtype=object)
    temp_data = gene_data.drop(gene)

    remaining_genes = temp_data.index.unique().to_numpy()
    remaining_genes_idx = range(len(remaining_genes))

    gene_states = np.atleast_2d(gene_data.loc[gene])
    remaining_genes_states = np.array(
        [np.atleast_2d(gene_data.loc[_gene]) for _gene in remaining_genes], dtype=object
    )

    # All possible predictor combinations - O(k * (n choose k))
    for combination in itertools.combinations(remaining_genes_idx, k):
        n_genes = len(combination)
        comb_gene_states = remaining_genes_states[list(combination)]

        # sample, gene, state_combo
        x_combos = np.empty((n_samples, n_genes, 0))

        for gene_state_combo in itertools.product(*comb_gene_states):
            states = np.array(gene_state_combo)
            # sample, gene, value
            x = states.T.reshape(-1, n_genes, 1)
            x_combos = np.append(x_combos, x, axis=2)

        for _exp in gene_states:
            _y = np.expand_dims(_exp, axis=1)
            for state_combo_idx in range(x_combos.shape[2]):
                x = x_combos[:, :, state_combo_idx]
                COD, A = gen_COD(x, _y)
                add_to_buff(buff, (COD, A, remaining_genes[list(combination)]))

    return buff


def add_to_buff(buff, data):
    n_predictors = buff.shape[1]
    COD, _, _ = data

    i = 0
    while i < n_predictors - 1:
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
        else:  # Else go next
            i += 1
            # if i < n_predictors - 2:
            #     break


@njit
def MSE(x, y):
    # Cannot have 1d arrays ret with Numba
    return np.mean((x - y) ** 2)


@njit
def g(x):
    # Numba does not work with the simple (x >= 0.5).astype(int)
    return np.where(x >= 0.5, 1, 0).astype(np.uint8)


@njit
def gen_COD(X, Y):
    """Schmulevich's (?) method for generating a COD. The closed form solution."""
    # Need float everywhere for Numba
    ones = np.ones(Y.shape, dtype=np.float_)
    X = np.append(X, ones, axis=1).astype(np.float_)
    Y = Y.astype(np.float_)
    R = np.dot(X.T, X)
    Rp = np.linalg.pinv(R)
    C = np.dot(X.T, Y)
    A = np.dot(Rp, C)  # for comparison

    y_pred = g(np.dot(X, A))
    y_pred_null = g(ones * np.mean(Y)) + 10**-8
    e_null = MSE(y_pred_null, Y)

    e = MSE(y_pred, Y)
    COD = (e_null - e) / e_null
    if COD < 0:
        COD = 10**-8
    return COD, A
