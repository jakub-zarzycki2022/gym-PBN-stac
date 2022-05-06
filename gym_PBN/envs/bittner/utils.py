"""
This file contains arcane magics.
"""
import pickle

import numpy as np
import xlrd
from gym_PBN.envs.bittner import gen


def spawn(includeIDs, Q_method, Q_axis, predictorN, file, predictorSetsPath, hack=28):
    # Loading xls
    genebook = xlrd.open_workbook(file)
    genesheet = genebook.sheet_by_name(genebook.sheets()[1].name)

    valfunc = lambda x: x.value  # helper function

    # Getting gene data

    geneIDs = np.empty((genesheet.nrows - 7, 1), dtype=int)
    geneNames = np.empty((genesheet.nrows - 7, 1), dtype=object)
    geneData = np.empty((genesheet.nrows - 7, 31), dtype=float)
    sDetected = np.empty((genesheet.nrows - 7), dtype=int)

    for i in range(2, genesheet.nrows - 5):
        geneIDs[i - 2] = int(valfunc(genesheet.cell(i, 0)))
        geneNames[i - 2] = valfunc(genesheet.cell(i, 2))
        sDetected[i - 2] = valfunc(genesheet.cell(i, 3))
        for j in range(5, 36):
            geneData[i - 2, j - 5] = valfunc(genesheet.cell(i, j))

    # Getting control data
    # NOTE: Not used

    controlIDs = np.empty((genesheet.nrows - 7, 1), dtype=int)
    controlNames = np.empty((genesheet.nrows - 7, 1), dtype=object)
    controlData = np.empty((genesheet.nrows - 7, 7), dtype=float)

    for i in range(2, genesheet.nrows - 5):
        controlIDs[i - 2] = int(valfunc(genesheet.cell(i, 0)))
        controlNames[i - 2] = valfunc(genesheet.cell(i, 2))
        for j in range(36, 43):
            controlData[i - 2, j - 36] = valfunc(genesheet.cell(i, j))

    # Getting weights

    weightsheet = genebook.sheet_by_name(genebook.sheets()[2].name)

    weightIDs = np.empty((weightsheet.nrows - 2, 1), dtype=int)
    weightValues = np.empty((weightsheet.nrows - 2, 1), dtype=float)

    for i in range(2, weightsheet.nrows):
        weightIDs[i - 2] = int(valfunc(weightsheet.cell(i, 0)))
        weightValues[i - 2] = valfunc(weightsheet.cell(i, 3))

    def appendExtraID(existing_list, tot_len, pool):
        n = len(existing_list)
        i = 0
        while n < tot_len:
            if not pool[i] in existing_list:
                existing_list.append(pool[i])
                n += 1
            i += 1

    # HACK
    if hack == 70:
        appendExtraID(includeIDs, 70, weightIDs)

    # Calling the function

    genes = (geneIDs, geneNames, geneData)

    dataGroup = gen.DataGroup(geneIDs, geneNames, geneData)
    # with open(
    #     f"{predictorSetsPath}/dataGroup{len(includeIDs)}_{predictorN}_{Q_method}.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(dataGroup, f)

    env = gen.generatePBNGenes(
        dataGroup,
        hack,  # HACK
        None,
        includeIDs,
        2,
        Q_method,
        1,
        predictorN,
        f"{predictorSetsPath}/predictorSets{len(includeIDs)}_{predictorN}_{Q_method}.p",
    )
    return env
