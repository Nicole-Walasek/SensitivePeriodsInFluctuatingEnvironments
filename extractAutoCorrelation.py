# approximate the autocorrelation of a markov chain

import numpy as np
import pandas as pd
import math
from MarkovChain import MarkovChain
from multiprocessing import Pool
import itertools
import math

def funcStar(allArgs):
    return generateSequence(*allArgs)


def generateSequence(numSeq, MC, startState, seqLength):
    return [MC.generate_states(startState, seqLength) for x in np.arange(numSeq)]


def autocorr(x, t):
    df = pd.concat([pd.DataFrame(x[:-t]), pd.DataFrame(x[t:])], axis=1)
    corrVal = df.corr().iloc[0, 1]
    if math.isnan(corrVal) and sum(x[:-t]) == sum(x[t:]):
        return 1
    elif math.isnan(corrVal):  # means that one variable had no variance
        return 0.9
    else:
        return corrVal


# simulate chains or calculate ALL chains of lnegth n
# weight average autocorrelation by likelihood of chain using the forward algorithm?
# but if a chain is simulated than it should be reflectinc the actual acutocorrelation
def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx


def calcStationaryDist(markov_chain):
    pE0E0, pE1E1 = markov_chain
    pE0E1 = 1 - pE0E0
    pE1E0 = 1 - pE1E1

    P = np.array([[pE0E0, pE0E1], [pE1E0, pE1E1]])
    w, v = np.linalg.eig(P.transpose())
    oneIdx = find_nearest(w, 1)
    # this corresponds to the overall prior
    # the starting distribution does however matter for the actual behavior over time
    pE0, pE1 = v[:, oneIdx] / float(sum(v[:, oneIdx]))
    return pE0


def calcAutocorrelation(markovProbabilities, lag):
    autocorrelationDict = {}
    for markovChain in markovProbabilities:
        pE0E0, pE1E1 = markovChain
        pE0E1 = 1 - pE0E0
        pE1E0 = 1 - pE1E1
        markov_chain = {
            'E0': {'E0': pE0E0, 'E1': pE0E1},
            'E1': {'E0': pE1E0, 'E1': pE1E1},
        }
        P = np.array(
            [[markov_chain['E0']['E0'], markov_chain['E0']['E1']],
             [markov_chain['E1']['E0'], markov_chain['E1']['E1']]])
        # formula for symmetric markov chains
        autoCorrelation = None
        if pE0E0 == pE1E1:
            autoCorrelation = (2 * pE0E0 - 1) ** lag
            autocorrelationDict[markovChain] = round(autoCorrelation,2)
        else:
            autocorrelationDict[markovChain] = autoCorrelation
    return autocorrelationDict


def calcAutocorrelationSim(markovProbabilities, lag, n, simNum):
    autocorrelationDict = {}
    for markovChain in markovProbabilities:
        pE0E0, pE1E1 = markovChain
        pE0E1 = 1 - pE0E0
        pE1E0 = 1 - pE1E1
        markov_chain = {
            0: {0: pE0E0, 1: pE0E1},
            1: {0: pE1E0, 1: pE1E1},
        }
        P = np.array(
            [[markov_chain[0][0], markov_chain[0][1]],
             [markov_chain[1][0], markov_chain[1][1]]])
        # create the simulation probabilities

        currMC = MarkovChain(markov_chain)

        simChunks = [math.ceil(simNum / float(12))] * 12
        pool = Pool(12)

        result = pool.map(funcStar,
                          itertools.izip(simChunks, itertools.repeat(currMC), itertools.repeat(0), itertools.repeat(n)))
        pool.close()
        pool.join()

        E0Sequences = list(itertools.chain.from_iterable(result))


        pool = Pool(12)

        result = pool.map(funcStar,
                          itertools.izip(simChunks, itertools.repeat(currMC), itertools.repeat(1), itertools.repeat(n)))
        pool.close()
        pool.join()

        E1Sequences = list(itertools.chain.from_iterable(result))

        autocorrStartE0 = np.mean([autocorr(row, lag) for row in E0Sequences])
        autocorrStartE1 = np.mean([autocorr(row, lag) for row in E1Sequences])

        pE0 = calcStationaryDist(markovChain)
        pE1 = 1 - pE0
        autocorrAvg = pE0 * autocorrStartE0 + pE1 * autocorrStartE1
        autocorrelationDict[markovChain] = round(autocorrAvg,2)

    return autocorrelationDict

#
# #print calcAutocorrelation(markovProbabilities, 1)
# a = calcAutocorrelationSim(markovProbabilitiesA, 1, 500, 2000)
# b = calcAutocorrelationSim(markovProbabilitiesA, 1, 10, 2000)
#
# print a.keys()
# print a.values()
# print b.keys()
# print b.values()
#
# a = calcAutocorrelationSim(markovProbabilitiesB, 1, 500, 2000)
# b = calcAutocorrelationSim(markovProbabilitiesB, 1, 10, 2000)
#
# print a.keys()
# print a.values()
# print b.keys()
# print b.values()
#
# a = calcAutocorrelationSim(markovProbabilitiesC, 1, 500, 2000)
# b = calcAutocorrelationSim(markovProbabilitiesC, 1, 10, 2000)
#
# print a.keys()
# print a.values()
# print b.keys()
# print b.values()