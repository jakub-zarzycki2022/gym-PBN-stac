"""
This file contains arcane magics.
"""
import copy
import itertools
import math
import pickle
from os import path

import numpy as np
from tqdm import tqdm
from gym_PBN.envs.bittner import base
from scipy.integrate import quad


# Helpful classes to group data
class DataGroup:
    def __init__(self, IDs, names, data):
        if IDs.shape[0] == names.shape[0] and IDs.shape[0] == data.shape[0]:
            self.IDs = IDs
            self.names = names
            self.data = data
        else:
            raise Exception("Data shape mismatch")

    def trimToIDs(self, IDs):
        newIDs = np.empty((0, 1), dtype=int)
        newNames = np.empty((0, 1), dtype=object)
        newData = np.empty((self.data.shape[1], 0), dtype=self.data.dtype)
        for i in IDs:
            index = np.where(self.IDs == i)
            for subi in index[0]:
                newIDs = np.append(newIDs, np.expand_dims(self.IDs[subi], 1), axis=0)
                newNames = np.append(
                    newNames, np.expand_dims(self.names[subi], 1), axis=0
                )
                newData = np.append(
                    newData, np.expand_dims(self.data[subi, :], 1), axis=1
                )
        newData = newData.T
        return DataGroup(newIDs, newNames, newData)

    def getIndexByID(self, ID):
        out = np.where(self.IDs == ID)
        if out[0].size == 0:
            raise Exception("Could not find entry with ID {0}".format(ID))
        return out[0]

    def printAll(self):
        for i in range(0, self.IDs.shape[0]):
            print(
                str(self.IDs[i]) + " " + str(self.names[i]) + " " + str(self.data[i, :])
            )

    def removeDuplicates(self):  # Function to remove rows with same IDs.
        newNames = np.empty((0, 1), dtype=object)
        newIDs = np.empty((0, 1), dtype=int)
        newData = np.empty((0, self.data.shape[1]), dtype=int)
        IDcopy = np.copy(self.IDs)
        for i in range(0, self.IDs.shape[0]):
            if IDcopy[i] != -1:
                indexes = self.getIndexByID(IDcopy[i])
                ID = np.expand_dims(self.IDs[indexes[0]], 1)
                IDcopy[indexes] = -1
                X = np.empty((newData.shape[1], 0), dtype=float)
                for index in indexes:
                    X = np.append(X, np.expand_dims(self.data[index], 1), axis=1)
                X = np.expand_dims(np.average(X, 1), 1)
                name = np.expand_dims(self.names[indexes[0]], 1)
                newNames = np.append(newNames, name, axis=0)
                newIDs = np.append(newIDs, ID, axis=0)
                newData = np.append(newData, X.T, axis=0)
        return DataGroup(newIDs, newNames, newData)


# Main function


def generatePBNGenes(
    dataGroup,
    n,
    discT,
    forceIncludeIDs,
    nary,
    discMethod,
    discAxis,
    predictorN,
    savepath,
    k=3,
):

    dataGroup = dataGroup.trimToIDs(forceIncludeIDs)
    binGroup = discretize(dataGroup, discMethod, discAxis, discT)
    graph = generateGraph(binGroup, k, nary, predictorN, savepath)

    return graph


def generatePBNGenesLUT(
    dataGroup,
    n,
    discT,
    forceIncludeIDs,
    nary,
    discMethod,
    discAxis,
    predictorN,
    savepath,
    k=3,
):
    dataGroup = dataGroup.trimToIDs(forceIncludeIDs)
    binGroup = discretize(dataGroup, discMethod, discAxis, discT)
    graph = generateGraphLUT(binGroup, k, nary)

    return graph


def generateGraphLUT(dataGroup, k, b):
    graph = base.Graph(b)
    IDcopy = np.copy(dataGroup.IDs)
    IDs = np.empty((0, 1), dtype=int)
    for i in range(0, IDcopy.shape[0]):
        if IDcopy[i] != -1:
            indexes = dataGroup.getIndexByID(IDcopy[i])
            ID = np.expand_dims(dataGroup.IDs[indexes[0]], 1)
            IDcopy[indexes] = -1
            IDs = np.append(IDs, ID, axis=0)
    expressions = []
    for i in range(IDs.size):
        indexes = dataGroup.getIndexByID(IDs[i])
        Y = dataGroup.data[indexes]
        expressions = expressions + [Y]
    expressions = np.asarray(expressions)
    nodelist = []
    for i in range(IDs.size):
        nodeID = IDs[i]
        globalIndex = dataGroup.getIndexByID(nodeID)[0]
        inputGenes = findInputGenes(i, expressions, IDs, k)
        inputGeneIndex = np.empty(3, dtype=object)
        inputExpressions = np.empty(3, dtype=object)
        for j in range(len(inputGenes)):
            inputGeneIndex[j] = dataGroup.getIndexByID(inputGenes[j])
            inputExpressions[j] = dataGroup.data[inputGeneIndex[j]]
        LUT = generateLUT(globalIndex, inputGeneIndex, inputExpressions, expressions[i])
        nodeName = dataGroup.names[globalIndex]
        node = base.Node(i, globalIndex, nodeName, nodeID, LUTflag=True)
        node.addLUT(LUT, inputGenes)
        nodelist = nodelist + [node]

    graph.addNodes(nodelist)
    return graph


def generateLUT(index, inIndex, expressions, outputExp):
    k = len(inIndex)
    LUT = np.empty((2**k, 2), dtype=float)
    for i in range(2**k):
        entry = np.zeros(2, dtype=int)
        inputList = genBoolList(i, k, 2)  # List representing the input for i
        # Iterate over each of the inputs.
        # If input matches, add
        nCombos = [None] * k
        for j in range(len(inIndex)):
            nCombos[j] = list(
                range(len(inIndex[j]))
            )  # Computing the number of possible combinations at each index.
        indexCombos = list(
            itertools.product(*nCombos)
        )  # Computing the possible input combinations for input indexes.

        Xcombos = np.empty((31, k, 0))  # NOTE: the 31 here is very bad. Sad!

        for indexCombo in indexCombos:
            X = np.empty((31, 0, 1))
            for l in range(k):
                x = expressions[l][indexCombo[l]]
                x = np.expand_dims(x, axis=1)
                x = np.expand_dims(x, axis=2)
                X = np.append(X, x, axis=1)
            Xcombos = np.append(Xcombos, X, axis=2)

        # Xcombos third axis holds possible combinations of input genes.

        for j in range(Xcombos.shape[2]):
            X = Xcombos[:, :, j]
            for m in range(Xcombos.shape[0]):
                x = np.expand_dims(X[m, :], axis=1)
                match = True
                for l in range(x.shape[0]):
                    if x[l] != inputList[l]:
                        match = False
                if match:
                    matchingOut = outputExp[:, m]
                    for mOut in matchingOut:
                        entry[mOut] += 1
        if (entry == 0).all():
            outputOff = outputExp == 0
            outputOn = outputExp == 1
            entry[0] = np.sum(outputOff)
            entry[1] = np.sum(outputOn)

        entry = np.divide(entry, np.sum(entry))

        LUT[i, :] = entry

    return LUT


def findInputGenes(index, expressions, IDs, k):
    ex = expressions[index]
    ID = IDs[index]
    IDcopy = np.copy(IDs)
    IDcopy = np.delete(IDcopy, index)
    EXcopy = np.copy(expressions)
    EXcopy = np.delete(EXcopy, index)
    indexes = list(range(IDcopy.size))
    combinations = []
    PREDbest = None
    for comb in itertools.combinations(indexes, k):
        combinations = combinations + [comb]
    combinations = np.asarray(combinations)
    for combination in combinations:
        nCombos = [None] * k
        for j in range(len(combination)):
            nCombos[j] = list(range(len(EXcopy[combination[j]])))
            if len(EXcopy[combination[j]]) > 1:
                uniques = np.unique(EXcopy[combination[j]])
                nCombos[j] = list(range(len(uniques)))
        indexCombos = list(itertools.product(*nCombos))
        Xcombos = np.empty((31, k, 0))  # NOTE: the 31 here is very bad. Sad!
        for indexCombo in indexCombos:
            X = np.empty((31, 0, 1))
            for l in range(len(combination)):
                x = np.expand_dims(
                    np.expand_dims(EXcopy[combination[l]][indexCombo[l]].T, axis=1),
                    axis=2,
                )
                X = np.append(X, x, axis=1)
            Xcombos = np.append(Xcombos, X, axis=2)
        Yt = ex
        if len(Yt) > 1:
            Yt = checkDuplicate(Yt)
        for Y in Yt:
            ypression = np.expand_dims(Y, axis=1)
            for j in range(Xcombos.shape[2]):
                xpression = Xcombos[:, :, j]
                COD, A = genSingleCOD(xpression, ypression)
                if PREDbest == None:
                    PREDbest = (IDcopy[combination], COD)
                elif PREDbest[1] < COD:
                    PREDbest = (IDcopy[combination], COD)
    return PREDbest[0]


def generateGraph(dataGroup, k, b, predictorN, savepath):
    graph = base.Graph(b)
    IDcopy = np.copy(dataGroup.IDs)
    IDs = np.empty((0, 1), dtype=int)
    for i in range(0, IDcopy.shape[0]):
        if IDcopy[i] != -1:
            indexes = dataGroup.getIndexByID(IDcopy[i])
            ID = np.expand_dims(dataGroup.IDs[indexes[0]], 1)
            IDcopy[indexes] = -1
            IDs = np.append(IDs, ID, axis=0)
    expressions = []
    for i in range(IDs.size):
        indexes = dataGroup.getIndexByID(IDs[i])
        Y = dataGroup.data[indexes]
        expressions = expressions + [Y]
    expressions = np.asarray(expressions, dtype=object)
    predictorSets = genPredictorSets(IDs, k, expressions, predictorN, savepath)
    nodelist = []
    for i in range(IDs.size):
        nodeID = IDs[i]
        predictorSet = predictorSets[i]
        globalIndex = dataGroup.getIndexByID(nodeID)[0]
        nodeName = dataGroup.names[globalIndex]
        node = base.Node(i, globalIndex, nodeName, nodeID)
        node.addPredictors(predictorSet)
        nodelist = nodelist + [node]

    graph.addNodes(nodelist)
    return graph


def generatePBNGates(nodes):
    graph = base.Graph(2)
    nNodes = len(nodes)
    nodelist = []
    for i in range(nNodes):
        entry = nodes[i]
        nodeName = entry[0]
        nodeFunctions = entry[1]
        nodeID = i
        globalIndex = i
        node = base.Node(i, globalIndex, nodeName, nodeID)
        predictorSet = gatesToPredictors(nodeFunctions)
        node.addPredictors(predictorSet)
        nodelist = nodelist + [node]

    graph.addNodes(nodelist)
    return graph


def gatesToPredictors(f):
    predictorSet = np.empty((3, 0), dtype=object)
    for tup in f:
        func = tup[0]
        COD = tup[1]
        nInputs = getInputN(func)
        X = genX(nInputs)
        Y = genY(X, func)
        confirm, A = genSingleCOD(np.array(X), np.expand_dims(np.array(Y), axis=1))
        if confirm < 0.95:
            raise Exception("Predictor can' represent the function {0}".format(func))
        inputIndex = genInputIndex(func)
        pred = (COD, A, inputIndex)
        pred = np.expand_dims(np.array(pred), axis=1)
        predictorSet = np.append(predictorSet, pred, axis=1)
    return predictorSet


def genY(X, func):
    Y = [None] * len(X)
    for i in range(len(X)):
        x = X[i]
        y = evalRPN(x, func)
        Y[i] = y
    return Y


# aaaaa help
def evalRPN(x, funcs):
    f = copy.deepcopy(funcs)
    inputIndex = genInputIndex(f)
    tempStack = []
    f.reverse()
    while len(f) > 0:
        e = f.pop()
        if not isinstance(e, type(lambda: 0)):
            tempStack += [x[inputIndex.index(e)]]
        elif e.__name__ == "NOT":
            a = tempStack.pop(len(tempStack) - 1)
            b = e(a)
            tempStack += [b]
        else:
            a = tempStack.pop(len(tempStack) - 1)
            b = tempStack.pop(len(tempStack) - 1)
            c = e(a, b)
            tempStack += [c]
    if len(tempStack) > 1:
        raise Exception("RPN evaluation may have goofed - multiple values in the stack")
    output = tempStack[0]
    return output


def genInputIndex(func):
    foundInputs = []
    for e in func:
        if not isinstance(e, type(lambda: 0)) and not e in foundInputs:
            foundInputs += [e]
    return foundInputs


def getInputN(func):
    inputN = 0
    foundInputs = []
    for e in func:
        if not isinstance(e, type(lambda: 0)) and not e in foundInputs:
            inputN += 1
            foundInputs += [e]
    return inputN


def genX(nInputs):
    output = [None] * (2**nInputs)
    for i in range(2**nInputs):
        output[i] = genBoolList(i, nInputs, 2)
    return output


# FUnction to generate a n-ary list representing a decimal value. Kinda
def genBoolList(n, length, b):
    output = np.zeros((length), dtype=int)
    for i in range(length):
        output[i] = n % b
        n = n // b
    return output


def genPredictorSets(IDs, k, expressions, predictorN, savepath="predictorSets.p"):
    predictorSets = []
    if path.exists(savepath):
        return pickle.load((open(savepath, "rb")))
    else:
        pbar = tqdm(range(IDs.size))
        for i in pbar:
            pbar.set_description(
                f"Calculating top {predictorN} predictors for gene of ID {IDs[i]}."
            )
            predictorSet = []
            IDcopy = np.copy(IDs)
            IDcopy = np.delete(IDcopy, i)
            EXcopy = np.copy(expressions)
            EXcopy = np.delete(EXcopy, i, axis=0)
            buff = np.empty((3, predictorN), dtype=object)
            indexes = list(range(IDcopy.size))
            combinations = []
            for comb in itertools.combinations(indexes, k):
                combinations = combinations + [comb]
            combinations = np.asarray(combinations)
            for combination in combinations:
                nGenes = len(combination)
                nCombos = [None] * nGenes
                for j in range(len(combination)):
                    nCombos[j] = list(range(len(EXcopy[combination[j]])))
                    if len(EXcopy[combination[j]]) > 1:
                        uniques = np.unique(EXcopy[combination[j]])
                        nCombos[j] = list(range(len(uniques)))
                indexCombos = list(itertools.product(*nCombos))
                Xcombos = np.empty(
                    (31, nGenes, 0)
                )  # NOTE: the 31 here is very bad. Sad!
                for indexCombo in indexCombos:
                    X = np.empty((31, 0, 1))
                    for l in range(len(combination)):
                        x = np.expand_dims(
                            np.expand_dims(
                                EXcopy[combination[l]][indexCombo[l]].T, axis=1
                            ),
                            axis=2,
                        )
                        X = np.append(X, x, axis=1)
                    Xcombos = np.append(Xcombos, X, axis=2)
                Yt = expressions[i]
                if len(Yt) > 1:
                    Yt = checkDuplicate(Yt)
                for Y in Yt:
                    ypression = np.expand_dims(Y, axis=1)
                    for j in range(Xcombos.shape[2]):
                        xpression = Xcombos[:, :, j]
                        COD, A = genSingleCOD(xpression, ypression)
                        buff = addToBuff(
                            buff, (COD, A, IDcopy[combination]), predictorN
                        )
            predictorSets = predictorSets + [buff]
        pickle.dump(predictorSets, open(savepath, "wb"))
        return predictorSets


def checkDuplicate(X):
    output = np.empty((0, X.shape[1]), dtype=int)
    for x in X:
        if not compareLists(output, x):
            output = np.append(output, np.expand_dims(x, axis=1).T, axis=0)
    return output


# returns true if lists are the same. false otherwise.
def compareLists(output, x):
    if output.shape[0] == 0:
        return False
    outputs = [False] * output.shape[0]
    for i in range(output.shape[0]):
        suboutput = True
        for j in range(output.shape[1]):
            if x[j] != output[i, j]:
                suboutput = False
        outputs[i] = suboutput
    o = False
    for e in outputs:
        o = e or o
    return o


def addToBuff(buff, tup, predictorN):
    COD, A, comb = tup

    cFlag = True
    i = 0
    while cFlag:
        if buff[0, i] == None:
            buff[0, i] = COD
            buff[1, i] = A
            buff[2, i] = comb
            cFlag = False
        else:
            if buff[0, i] < COD:
                temp = np.copy(buff[:, i])
                buff[0, i] = COD
                buff[1, i] = A
                buff[2, i] = comb
                while i < predictorN - 1:
                    temp2 = np.copy(buff[:, i + 1])
                    buff[:, i + 1] = np.copy(temp)
                    temp = np.copy(temp2)
                    i += 1
                cFlag = False
            else:
                i += 1
                if i == predictorN - 2:
                    cFlag = False
        if i == predictorN - 1:
            cFlag = False
    return buff


# Schmulevich's (?) method for generating a COD. The closed form solution.


def genSingleCOD(X, Y):
    ones = np.ones(Y.shape)
    X = np.append(X, ones, axis=1)
    R = np.matmul(X.T, X)
    Rp = np.linalg.pinv(R)
    C = np.matmul(X.T, Y)
    A = np.matmul(Rp, C)  # for comparison

    Ypred = g(np.matmul(X, A))
    YpredNull = g(ones * np.mean(Y)) + 10**-8
    eNull = MSE(YpredNull, Y)

    e = MSE(Ypred, Y)
    COD = (eNull - e) / eNull
    if COD < 0:
        COD = 10**-8
    return COD, A


def MSE(X, Y):
    e = 0
    for i in range(X.shape[0]):
        e += (X[i] - Y[i]) ** 2
    e = e / X.shape[0]
    return e[0]


def g(X):
    output = np.empty(X.shape)
    for i in range(X.shape[0]):
        if X[i] >= 0.5:
            output[i] = 1
        else:
            output[i] = 0
    return output.astype(int)


def discretize(dataGroup, method, discAxis, t, nIter=20, q=2, nRuns=10):
    if method == "average":
        output = nArize(dataGroup, discAxis, lambda x: np.mean(x, discAxis))
        return output
    if method == "median":
        output = nArize(dataGroup, discAxis, lambda x: np.median(x, discAxis))
        return output
    if method == "cutoff":
        output = nArize(dataGroup, discAxis, lambda x: thresholdF(x, t, discAxis))
        return output
    if method == "top":
        output = nArize(dataGroup, discAxis, lambda x: thresholdFRank(x, t, discAxis))
        return output
    if method == "k-means":
        output = nArize(
            dataGroup, discAxis, lambda x: kMeans(x, discAxis, nIter, nRuns, q)
        )
        return output
    raise Exception('"{0}" is not an implemented discretisation method.')


def kMeans(X, axis, nIter, nRuns, q):
    Xlog = np.log(np.copy(X))
    if axis == 0:
        axis = 1
    elif axis == 1:
        axis = 0
    # Creating clusters
    thresholds = []
    for i in range(nRuns):
        t = cluster(Xlog, axis, nIter, q)
        thresholds = thresholds + [t]
    # evaluating clusters to pick the best one. NOTE: I should probably evaluate each one. Oh well.
    errors = np.empty(len(thresholds))
    for i in range(len(thresholds)):
        errors[i] = evalCluster(Xlog, thresholds[i], axis)
    minT = np.argmin(errors)
    threshold = thresholds[minT]
    output = np.empty(threshold.shape, dtype=float)
    for i in range(output.size):
        output[i] = threshold[i][0]

    output = np.exp(output)
    return output


def cluster(X, axis, nIter, q):  # I'm only doing binary case for today I can't
    if axis == None:
        X = X.flatten()
        thresholds = np.empty((1), dtype=object)
    else:
        thresholds = np.empty(X.shape[axis], dtype=object)
    for k in range(thresholds.size):
        if axis == None:
            exp = X
        else:
            exp = np.take(X, k, axis=axis)
        mI = exp.min()
        mA = exp.max()
        means = np.random.rand(q) * (mA - mI) + mI
        for i in range(nIter):
            clusters = [[]] * q
            for x in exp:
                dists = np.empty(q)
                for j in range(means.size):
                    dists[j] = abs(x - means[j])
                clusters[np.argmin(dists)] = clusters[np.argmin(dists)] + [x]
            for j in range(len(clusters)):
                means[j] = sum(clusters[j]) / len(clusters[j])
        for i in range(len(clusters)):
            clusters[i].sort()
        clusterIndexSort = np.argsort(means)
        t = np.empty(q - 1)
        for i in range(clusterIndexSort.size - 1):
            r1 = clusters[clusterIndexSort[i]][-1]
            r2 = clusters[clusterIndexSort[i + 1]][0]
            t[i] = (r1 + r2) / 2
        thresholds[k] = (t, means[clusterIndexSort])
    return thresholds


def evalCluster(X, thresholds, axis):
    e = 0
    for j in range(thresholds.size):
        threshold = thresholds[j]
        if axis == None:
            exp = X.flatten()
        else:
            exp = np.take(X, j, axis=axis)

        (t, means) = threshold

        # Parameters for the gaussian.
        pMean = np.mean(exp)
        std = np.sqrt(np.var(exp))

        errors = np.zeros(len(means))

        # For the time being it's binary. TIme is of essence guys.
        # Splitting the data into two clusters.
        clusters = [[]] * 2  # NOTE: !!!!!!
        for x in exp:
            if x < t:
                clusters[0] = clusters[0] + [x]
            else:
                clusters[1] = clusters[1] + [x]

        for i in range(len(clusters)):
            if i == 0:
                tLower = min(clusters[i])
                tUpper = t[0]
            else:
                tLower = t[0]
                tUpper = max(clusters[i])
            I = quad(
                integrand, tLower, tUpper, args=(pMean, std, means[i])
            )  # Integration!
            errors[i] = I[0]

        e += np.sum(errors)
    return e


def gaussian(x, mean, std):
    a = 1 / (std * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    output = a * math.exp(exponent)
    return output


def integrand(x, mean, std, r):
    return ((x - r) ** 2) * gaussian(x, mean, std)


def thresholdF(expression, threshold, axis):
    top = np.amax(expression, axis)
    return top * (1 - threshold)


def thresholdFRank(expression, threshold, axis):
    sortd = np.sort(expression, axis)
    if axis == None:
        n = sortd.size
        index = math.floor(n * threshold)
        output = sortd[index]
    else:
        if axis == 1:
            sortd = sortd.T
        n = sortd.shape[axis]
        index = math.floor(n * threshold)
        output = sortd[index, :]
    return output


def nArize(dataGroup, axis, function):
    dataCopy = np.copy(dataGroup.data)

    newNames = np.empty(dataGroup.names.shape, dtype=object)
    newIDs = np.empty(dataGroup.IDs.shape, dtype=int)
    threshold = function(dataCopy)

    if axis == 1:
        dataCopy = dataCopy.T
    newData = np.empty(dataCopy.shape, dtype=int)
    newIDs = np.copy(dataGroup.IDs)
    newNames = np.copy(dataGroup.names)
    for i in range(dataCopy.shape[0]):
        expression = dataCopy[i, :]
        for j in range(dataCopy.shape[1]):
            if axis == None:
                t = threshold
            else:
                t = threshold[j]
            if expression[j] < t:
                newData[i, j] = 0
            else:
                newData[i, j] = 1
    if axis == 1:
        newData = newData.T
    output = DataGroup(newIDs, newNames, newData)
    return output


def birthPredictorSet(gates, conn):
    X = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]])
    Yor = np.empty((1, X.shape[1]), dtype=int)
    Yand = np.empty((1, X.shape[1]), dtype=int)
    for i in range(X.shape[1]):
        Yor[0, i] = X[0, i] or X[1, i]
    for i in range(X.shape[1]):
        Yand[0, i] = X[0, i] and X[1, i]
    ORCOD, ORA = genSingleCOD(X.T, Yor.T)
    ANDCOD, ANDA = genSingleCOD(X.T, Yand.T)

    predictorSet = np.empty((3, 2), dtype=object)
    predictorSet[0, 0] = gates[0][0]
    predictorSet[0, 1] = gates[1][0]
    predictorSet[1, 0] = ORA
    predictorSet[1, 1] = ANDA
    predictorSet[2, 0] = conn
    predictorSet[2, 1] = conn
    return predictorSet


# Gets the top COD predictor for the gene at ID
def getPredictors(ID, dataGroup, k):
    targetCandidates = dataGroup.getIndexByID(ID)
    highestCOD = 0
    highestPred = None
    for target in targetCandidates:
        predictors = np.empty((0), dtype=int)
        Y = dataGroup.data[target, :]
        for i in range(0, k):
            predictors, COD = appendNthPredictor(dataGroup, predictors, i, Y, ID)
        if COD > highestCOD:
            highestCOD = COD
            highestPred = predictors
    return highestPred, highestCOD


# Find the best next predictor given first n predictors
def appendNthPredictor(dataGroup, predictors, n, Y, targID):
    X = np.empty((Y.shape[0], 0))
    for i in range(0, n):
        predictorID = predictors[i]
        predictorCandidates = dataGroup.getIndexByID(predictorID)
        highestCOD = 0
        for predictorCandidate in predictorCandidates:
            candX = np.expand_dims(dataGroup.data[predictorCandidate], 1)
            COD = genSingleCOD(candX, Y)
            if COD > highestCOD:
                Xn = candX
                highestCOD = COD
        X = np.append(X, Xn, axis=1)

    highestCOD = 0
    highestID = None
    for predID in dataGroup.IDs:
        if predID != targID and not predID in predictors:
            predictorCandidates = dataGroup.getIndexByID(predID)
            for predictorCandidate in predictorCandidates:
                predictorGenes = np.expand_dims(dataGroup.data[predictorCandidate], 1)
                Xtemp = np.append(X, predictorGenes, axis=1)
                COD = genSingleCOD(Xtemp, Y)
                if COD > highestCOD:
                    highestCOD = COD
                    highestID = predID
    return np.append(predictors, highestID), highestCOD
