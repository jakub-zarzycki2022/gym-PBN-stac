from pathlib import Path

import numpy as np
from gym_PBN.envs.bittner.gen.binarise import binarise
from gym_PBN.envs.bittner.utils import extract_gene_data, pad_ids, spawn

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "gym_PBN" / "envs" / "bittner" / "data"
GENE_DATA_PATH = DATA_PATH / "genedata.xls"


def test_data_extraction():
    gene_data, weight_ids = extract_gene_data(GENE_DATA_PATH)

    assert gene_data.shape[0] == 8067
    assert gene_data.shape[1] == 32
    assert gene_data.index.name == "ID"
    assert len(weight_ids) == 276


def test_id_padding():
    _, weight_ids = extract_gene_data(GENE_DATA_PATH)
    include_ids = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    new_ids = pad_ids(include_ids, 70, weight_ids)

    # fmt: off
    assert new_ids == [234237, 324901, 759948, 25485, 266361, 108208, 130057, 357278, 39781, 49665, 39159, 23185, 417218, 31251, 343072, 142076, 128100, 376725, 112500, 241530, 44563, 36950, 812276, 51018, 897806, 809473, 754538, 813533, 161992, 306013, 418105, 841308, 53316, 427943, 45421, 471096, 44605, 471918, 280768, 510130, 470621, 38770, 130100, 24588, 50043, 485690, 230360, 283617, 244086, 898092, 51740, 26789, 288733, 44584, 768272, 134829, 51814, 363086, 364469, 770377, 110503, 193106, 25081, 767851, 244307, 254428, 142067, 25495, 526657, 50271]
    # fmt: on


def test_trimming():
    gene_data, weight_ids = extract_gene_data(GENE_DATA_PATH)
    include_ids = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    new_ids = pad_ids(include_ids, 70, weight_ids)

    trimmed_data = gene_data.loc[new_ids]
    assert trimmed_data.shape[0] == 85
    assert trimmed_data.shape[1] == 32


def test_binarisation():
    gene_data, weight_ids = extract_gene_data(GENE_DATA_PATH)
    include_ids = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    new_ids = pad_ids(include_ids, 70, weight_ids)
    trimmed_data = gene_data.loc[new_ids]

    t_cols = [f"T{i}" for i in range(1, 32)]

    binned_data = binarise(trimmed_data, "kmeans")

    # Smoke tests
    assert binned_data.shape[1] == 32
    assert binned_data.index.name == "ID"

    for col_type in binned_data.dtypes[t_cols]:
        assert col_type == np.int64

    for col in binned_data[t_cols].columns:
        assert binned_data[col].max() == 1
        assert binned_data[col].min() == 0

    # Save to investigate binarisation quality
    binned_data.drop("Name", axis=1).to_csv("binned_test.csv")


def test_loc():
    gene_data, weight_ids = extract_gene_data(GENE_DATA_PATH)
    include_ids = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    include_ids = pad_ids(include_ids, 70, weight_ids)
    trimmed_data = gene_data.loc[include_ids]

    l = []
    for i in trimmed_data.index:
        first_idx = np.where(trimmed_data.index == i)[0][0]
        if first_idx not in l:
            l.append(first_idx)

    assert len(l) == len(include_ids)


def test_generation():
    # fmt: off
    include_ids = [234237, 324901, 759948, 25485, 324700, 43129, 266361, 108208, 40764, 130057, 39781, 49665, 39159, 23185,417218, 31251, 343072, 142076, 128100, 376725, 112500, 241530, 44563, 36950, 812276, 51018, 306013, 418105]
    # fmt: on

    graph = spawn(
        file=GENE_DATA_PATH,
        total_genes=28,
        include_ids=include_ids,
        bin_method="median",
        n_predictors=15,
        predictor_sets_path=ROOT_PATH,
    )

    assert graph
