# replication of Willem's 2016 paper "The evolution of sensitive periods in a model of incremental development"
import sys
import numpy as np
import itertools
from numpy import random
import cPickle as pickle
import time
import os
import matplotlib
matplotlib.use('Agg')
import shelve
import shutil
import matplotlib.pyplot as plt
from multiprocessing import Pool
from progressbar import Bar, Percentage, ProgressBar, Timer, ETA
from Node2 import buildForwardTree
from PreparePlottingVarCueValidityParallel import preparePlotting
from PlottingParallel import policyPlotReduced, joinIndidividualResultFiles
from twinStudiesVarCue3NewFastSim import runPlots, runAggregatePlots, runAggregatePlotsMerge, runAggregatePlotsAssymetries
from decimal import Decimal as D
from extractAutoCorrelationNew import calcAutocorrelation, calcAutocorrelationSim


# Parameter description
# x0 : number of cues indicating Environment 0 at time t
# x1 : number of cues indicating Environment 1 at time t
# y0 : number of developmental steps at time t towards phenotype 0
# y1 : number of developmental steps at time t towards phenotype 1
# yw : number of developmental steps at time t in which the organism waited


# function to set global variables
def set_global_variables(aVar, bVar, probVar, funVar, probMax, probMin, invUPeakVar, stepCutoffVar, deprivationRangeVar,
                         cueRelMaxVar):
    # beta defines the curvature of the reward and penalty functions
    global aFT
    global bFT
    global probFT
    global funFT
    global probMaxFT
    global probMinFT
    global invUPeak
    global stepCutoff
    global Ft1
    global deprivationRange
    global cueRelMax

    aFT = aVar
    bFT = bVar
    probFT = probVar
    funFT = funVar
    probMaxFT = probMax
    probMinFT = probMin
    invUPeak = invUPeakVar
    stepCutoff = stepCutoffVar
    deprivationRange = deprivationRangeVar
    cueRelMax = cueRelMaxVar


def set_global_Ft1(setFt1):
    global Ft1
    Ft1 = setFt1


# helper functions
def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()


def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    rel_tol = D(str(rel_tol))
    abs_tol = D(str(abs_tol))
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx

def calcStationaryDist(markov_chain):
    pE0E0, pE1E1 = markov_chain
    pE0E1 = float(D(str(1)) - D(str(pE0E0)))
    pE1E0 = float(D(str(1)) - D(str(pE1E1)))

    P = np.array([[pE0E0, pE0E1], [pE1E0, pE1E1]])

    w, v = np.linalg.eig(P.transpose())
    oneIdx = find_nearest(w, 1)
    # this corresponds to the overall prior
    # the starting distribution does however matter for the actual behavior over time
    pE0, pE1 = v[:, oneIdx] / sum(v[:, oneIdx])

    return D(str(pE0))


def findOptimumClose(fitnesList):
    # fitnesList corresponds to [F0,F1,Fw]
    # determine whether list contains ties
    F0 = fitnesList[0]
    F1 = fitnesList[1]
    Fw = fitnesList[2]
    relTol = 1e-12
    if isclose(F0, F1, rel_tol=relTol) and isclose(F0, Fw, rel_tol=relTol) and isclose(F1, Fw, rel_tol=relTol):
        idx = random.choice([0, 1, 2])
        return fitnesList[idx], (int(0), int(1), int(2))

    elif (isclose(F0, F1, rel_tol=relTol) and F0 > Fw):
        idx = random.choice([0, 1])
        return fitnesList[idx], (int(0), int(1))

    elif (isclose(F0, Fw, rel_tol=relTol) and F0 > F1):
        idx = random.choice([0, 2])
        return fitnesList[idx], (int(0), int(2))

    elif (isclose(F1, Fw, rel_tol=relTol) and F1 > F0):
        idx = random.choice([1, 2])
        return fitnesList[idx], (int(1), int(2))
    else:
        maxVal = np.max(fitnesList)
        idx = np.argmax(fitnesList)
        return maxVal, int(idx)


def findOptimum(fitnesList):
    # fitnesList corresponds to [F0,F1,Fw]
    # determine whether list contains ties
    F0 = fitnesList[0]
    F1 = fitnesList[1]
    Fw = fitnesList[2]

    if F0 == F1 == Fw:
        idx = random.choice([0, 1, 2])
        return fitnesList[idx], (int(0), int(1), int(2))

    elif (F0 == F1 and F0 > Fw):
        idx = random.choice([0, 1])
        return fitnesList[idx], (int(0), int(1))

    elif (F0 == Fw and F0 > F1):
        idx = random.choice([0, 2])
        return fitnesList[idx], (int(0), int(2))

    elif (F1 == Fw and F1 > F0):
        idx = random.choice([1, 2])
        return fitnesList[idx], (int(1), int(2))
    else:
        maxVal = np.max(fitnesList)
        idx = np.argmax(fitnesList)
        return maxVal, int(idx)


def normData(data, MinFT, MaxFT):
    if min(data) < MinFT or max(data) > MaxFT:
        rangeArr = max(data) - min(data)
        newRangeArr = MaxFT - MinFT
        normalized = [((val - min(data)) / float(rangeArr)) * newRangeArr + MinFT for val in data]
        return normalized
    else:
        return data


def stepFun(t, a, b, prob, T):
    if t < a:
        return 0
    elif t <= b:
        merke = sum(np.arange(a, b + 1, 1))

        return t / (float(merke) / (1 - prob))
    else:
        return prob / float(T - b)


def probFinalTime(t, T, kw):
    # currently these are all summing to 1, but this is not a necessity?
    if kw == 'None':
        return 0
    elif kw == 'uniform':
        return 1 / float(T)
    elif kw == 'log':
        tVal = np.arange(1, T + 1, 1)
        return np.log(t) / float(sum(np.log(tVal)))
    elif kw == 'step':
        return stepFun(t, aFT, bFT, probFT, T)

    elif kw == 'fun':
        return funFT(t)
    else:
        print 'Unkown keyword for random final time'
        exit(1)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def plotCueReliability(tValues, cueReliabilityArr, kw):
    plt.figure()
    ax = plt.gca()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(tValues, cueReliabilityArr)
    plt.xlim(1, max(tValues))
    plt.title('Cue reliability as a function of time')
    plt.xlabel('Time step')
    plt.ylabel('Cue reliability')
    plt.savefig("CueReliability%s.png" % kw)
    plt.close("all")


def processCueValidity(pC0E0, tValues):
    if is_number(pC0E0):
        cueValidityE0Dict = {t: round(float(pC0E0), 3) for t in tValues}
        cueValidityE1Dict = cueValidityE0Dict
        return (cueValidityE0Dict, cueValidityE1Dict)

    elif pC0E0 == 'increasing':
        g = lambda x: x
        # assure that you get a proper probability function
        cueValidityArr = normData([g(t) for t in tValues], 0.55, cueRelMax)
        cueValidityArr = [round(float(v), 3) for v in cueValidityArr]
        print cueValidityArr
        cueValidityE0Dict = {t: v for t, v in zip(tValues, cueValidityArr)}
        cueValidityE1Dict = cueValidityE0Dict
        plotCueReliability(tValues, cueValidityArr, pC0E0)

        return (cueValidityE0Dict, cueValidityE1Dict)

    elif pC0E0 == 'decreasing':
        g = lambda x: -x
        # assure that you get a proper probability function
        cueValidityArr = normData([g(t) for t in tValues], 0.55, cueRelMax)
        cueValidityArr = [round(float(v), 3) for v in cueValidityArr]
        print cueValidityArr
        cueValidityE0Dict = {t: v for t, v in zip(tValues, cueValidityArr)}
        cueValidityE1Dict = cueValidityE0Dict
        plotCueReliability(tValues, cueValidityArr, pC0E0)

        return (cueValidityE0Dict, cueValidityE1Dict)

    elif pC0E0 == 'inverted U':
        g = lambda x: -10 * (x - invUPeak) ** 2
        # assure that you get a proper probability function
        cueValidityArr = normData([g(t) for t in tValues], 0.55, cueRelMax)
        cueValidityArr = [round(float(v), 3) for v in cueValidityArr]
        print cueValidityArr
        cueValidityE0Dict = {t: round(float(v), 3) for t, v in zip(tValues, cueValidityArr)}
        cueValidityE1Dict = cueValidityE0Dict
        plotCueReliability(tValues, cueValidityArr, pC0E0)

        return (cueValidityE0Dict, cueValidityE1Dict)

    elif pC0E0 == 'triangular':
        g1 = lambda x: x  # increasing
        g2 = lambda x: -x  # decreasing

        cueValidityArr1 = normData([g1(t) for t in tValues[0:int(invUPeakVar)]], 0.55, cueRelMax)
        cueValidityArr1 = [round(float(v), 3) for v in cueValidityArr1]

        cueValidityArr2 = normData([g2(t) for t in tValues[int(invUPeakVar) - 1:]], 0.55, cueRelMax)
        cueValidityArr2 = [round(float(v), 3) for v in cueValidityArr2]

        cueValidityArr = cueValidityArr1 + cueValidityArr2[1:]

        # to make sure that the areas are equal
        # assure that you get a proper probability function
        areaCheck = normData([g1(t) for t in tValues], 0.55, cueRelMax)
        areaCheck = [round(float(v), 3) for v in areaCheck]

        if sum(cueValidityArr) != sum(areaCheck):
            storageVal = round((sum(areaCheck) - sum(cueValidityArr)) / float(len(tValues) - 3), 3)
            copyCueValidityArr = np.copy(cueValidityArr)

            cueValidityArr[1:int(invUPeakVar - 1)] = copyCueValidityArr[1:int(invUPeakVar - 1)] + storageVal
            cueValidityArr[int(invUPeakVar):-1] = copyCueValidityArr[int(invUPeakVar):-1] + storageVal

        cueValidityE0Dict = {t: round(v, 3) for t, v in zip(tValues, cueValidityArr)}
        cueValidityE1Dict = cueValidityE0Dict

        plotCueReliability(tValues, cueValidityArr, pC0E0)

        return (cueValidityE0Dict, cueValidityE1Dict)
    else:
        print 'Wrong input argument for cue validity paramter'
        exit(1)


# Bayesian updating scheme
def BayesianUpdating(pE0, pE1, pDE0, pDE1):
    pE0, pE1, pDE0, pDE1 = float(pE0), float(pE1), float(pDE0), float(pDE1)
    # pE0 is the evolutionary prior vor environment 1
    # pE1 is the evolutionary prior for environment 2
    # pDE0 and pDE1 are the probabilities of obtaining the data given environment 0 or 1 respectively (likelihood)
    p_D = pDE0 * pE0 + pDE1 * pE1
    b0_D = (pDE0 * pE0) / p_D
    b1_D = (pDE1 * pE1) / p_D

    return b0_D, b1_D


# terminal fitness function
def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    x0, x1, y0, y1, yw = state
    if argumentR == 'linear':
        phiVar = b0_D * D(str(y0)) + b1_D * D(str(y1))

    elif argumentR == 'diminishing':
        alphaRD = D(str(T)) / (D(str(1)) - (np.exp(D(str(-beta)) * (D(str(T))))))
        phiVar = b0_D * alphaRD * (D(str(1)) - np.exp(-D(str(beta)) * D(str(y0)))) + b1_D * alphaRD * (D(str(1)) -
                                                                np.exp(-D(str(beta)) * D(str(y1))))

    elif argumentR == 'increasing':
        alphaRI = D(str(T)) / ((np.exp(D(str(beta)) * D(str(T)))) - D(str(1)))
        phiVar = b0_D * alphaRI * (np.exp(D(str(beta)) * D(str(y0))) - D(str(1))) + b1_D * alphaRI * (np.exp(D(str(beta)) * D(str(y1))) - D(str(1)))
    else:
        print 'Wrong input argument to additive fitness reward function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    if argumentP == 'linear':
        psiVar = -(b0_D * D(str(y1)) + b1_D * D(str(y0)))
    elif argumentP == 'diminishing':
        alphaPD = D(str(T)) / (D(str(1)) - (np.exp(D(str(-beta)) * (D(str(T))))))
        psiVar = -(b0_D * alphaPD * (D(str(1)) - np.exp(-D(str(beta)) * D(str(y1)))) + b1_D * alphaPD * (D(str(1)) -
                                                                                                       np.exp(-D(str(
                                                                                                           beta)) * D(
                                                                                                           str(y0)))))


    elif argumentP == 'increasing':

        alphaPI = D(str(T)) / ((np.exp(D(str(beta)) * D(str(T)))) - D(str(1)))
        psiVar = -(b0_D * alphaPI * (np.exp(D(str(beta)) * D(str(y1))) - D(str(1))) + b1_D * alphaPI * (
                    np.exp(D(str(beta)) * D(str(y0))) - D(str(1))))


    else:
        print 'Wrong input argument to additive fitness penalty function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    tf = D(str(0)) + phiVar + D(str(psi_weighting)) * psiVar


    return tf


def terminalFitness(state, pE0, pE1, argumentR, argumentP, T, adultT, markov_chain, beta, psi_weighting):
    b0_D, b1_D = pE0, pE1

    P = np.array(
        [[markov_chain['E0']['E0'], markov_chain['E0']['E1']], [markov_chain['E1']['E0'],markov_chain['E1']['E1']]])


    tfList = []  # this will hold all fitness values across the adult lifespan
    for t in np.arange(1, adultT + 1, 1):
        tfCurr = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)
        tfList.append(tfCurr)
        # recalculate the distribution in the markov chain after one time step
        b0_D, b1_D = np.dot([pE0, pE1], np.linalg.matrix_power(P, t))

    tfList = np.array(tfList)

    return np.sum(tfList)


# @profile
def calcFitness(identifier, trees, x0, x1, y0, y1, yw, T, beta, psi_weightingParam):
    tree = trees[identifier]
    posE0 = tree.pE0
    posE1 = tree.pE1

    return terminalFitness((x0, x1, y0, y1, yw), posE0, posE1, argumentR, argumentP, T, beta, psi_weightingParam)


def calcTerminalFitnessThreaded(tree, identifier, y0Values, y1Values, ywValues, T, adultT, markov_chain, beta,
                                psi_weightingParam):

    identifier = int(identifier)
    terminalF = {}
    x0, x1 = tree.state
    posE0 = tree.pE0
    posE1 = tree.pE1



    allCombinations = (itertools.product(*[y0Values, y1Values, ywValues]))
    phenotypes = list(((y0, y1, yw) for y0, y1, yw in allCombinations if sum([y0, y1, yw]) == T))

    for y0, y1, yw in phenotypes:
        currKey = '%s;%s;%s;%s;%s;%s;%s' % (identifier, x0, x1, y0, y1, yw, T)
        terminalF[currKey] = terminalFitness((x0, x1, y0, y1, yw), posE0, posE1, argumentR,
                                             argumentP, T, adultT, markov_chain, beta, psi_weightingParam), posE1

    return terminalF


def func_star(allArgs):
    return calcTerminalFitnessThreaded(*allArgs)


def calcTerminalFitness(tree_path, y0Values, y1Values, ywValues, T, adultT, markov_chain, beta, psi_weightingParam,
                        batchSize):
    # if it doesn't exist yet create a directory for the final timestep
    if not os.path.exists("fitness/%s" % T):
        os.makedirs("fitness/%s" % T)


    trees = shelve.open(os.path.join(tree_path, "tree%s" % (T)))
    print "Number of binary trees in the last step: " + str(len(trees))
    # to prevent flooding memory space we need to load work the results in batches
    if len(trees) > batchSize:
        stepsize = batchSize
    else:
        stepsize = len(trees)

    # this is just for the progress bar
    widgets = ['Calculate terminal fitness: ', Percentage(), ' ',
               Bar(marker=('-'), left='[', right=']'), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=int((len(trees) / float(stepsize)))).start()

    for i in pbar(xrange(stepsize, len(trees) + stepsize, stepsize)):
        if i >= len(trees):
            endCondition = len(trees)
        else:
            endCondition = i

        currTerminalF = {}
        posteriorTerminal = {}
        pool = Pool(12)

        key_vallist = {(a, trees[str(a)]) for a in xrange(i - stepsize, endCondition, 1)}

        identifiers, allTrees = zip(*key_vallist)
        results = pool.map(func_star,
                           itertools.izip(allTrees, identifiers, itertools.repeat(y0Values), itertools.repeat(y1Values),
                                          itertools.repeat(ywValues), itertools.repeat(T), itertools.repeat(adultT),
                                          itertools.repeat(markov_chain), itertools.repeat(beta),
                                          itertools.repeat(psi_weightingParam)))
        pool.close()
        pool.join()

        results = iter(results)
        for currF in results:  # you can save them separatley if memory is an issue
            currTerminalF.update(currF)

        # store the result for the current batch
        if i == len(trees):
            lastVal = int((i + batchSize) / float(batchSize)) * batchSize
            currentFile = os.path.join('fitness/%s' % T, "TF%s" % lastVal)

        else:
            currentFile = os.path.join('fitness/%s' % T, "TF%s" % i)

        # shelve is a data structure that is generally slower for lookup than a dict but has the advantage that
        # it is much more memory effcient
        myShelve = shelve.open(currentFile)
        myShelve.update(currTerminalF)
        myShelve.close()


        del currTerminalF


# @profile
def F(identifier, x0, x1, y0, y1, yw, posE0, posE1, pC1D, pC0D, t, argumentR, argumentP, T, finalTimeArr):
    """
    this is the core of SDP
    """

    # computes the optimal decision in each time step
    # computes the maximal expected fitness for decision made between time t and T for valid combinations
    # of environmental variables
    prFT = D(str(finalTimeArr[t]))
    TF = D(str(0))

    identifierx0 = identifier * 2
    identifierx1 = identifier * 2 + 1


    t1 = t + 1

    if t == T - 1:
        currKey1 = '%s;%s;%s;%s;%s;%s;%s' % (identifier,x0, x1, y0 + 1, y1, yw, t1)

        currKey3 = '%s;%s;%s;%s;%s;%s;%s' % (identifier,x0, x1, y0, y1 + 1, yw, t1)

        currKey5 = '%s;%s;%s;%s;%s;%s;%s' % (identifier,x0, x1, y0, y1, yw + 1, t1)

        F0 = Ft1[currKey1][0]
        F1 = Ft1[currKey3][0]
        Fw = Ft1[currKey5][0]
    else:

        currKey1 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx0, x0 + 1, x1, y0 + 1, y1, yw, t1)
        currKey2 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx1, x0, x1 + 1, y0 + 1, y1, yw, t1)

        currKey3 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx0, x0 + 1, x1, y0, y1 + 1, yw, t1)
        currKey4 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx1, x0, x1 + 1, y0, y1 + 1, yw, t1)

        currKey5 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx0, x0 + 1, x1, y0, y1, yw + 1, t1)
        currKey6 = '%s;%s;%s;%s;%s;%s;%s' % (identifierx1, x0, x1 + 1, y0, y1, yw + 1, t1)

        F0 = TF + (D(str(1)) - prFT) * (pC0D * Ft1[currKey1][0] + pC1D * Ft1[currKey2][0])
        F1 = TF + (D(str(1)) - prFT) * (pC0D * Ft1[currKey3][0] + pC1D * Ft1[currKey4][0])
        Fw = TF + (D(str(1)) - prFT) * (pC0D * Ft1[currKey5][0] + pC1D * Ft1[currKey6][0])


    maxF, maxDecision = findOptimumClose([F0, F1, Fw])

    return maxF, maxDecision  # in order to track the current beliefs


def func_star2(allArgs):
    return oneFitnessSweep(*allArgs)


def oneFitnessSweep(tree, identifier, t, phenotypes, finalTimeArr, T, argumentR, argumentP):
    policyStar = {}
    x0, x1 = tree.state
    posE1 = tree.pE1
    posE0 = tree.pE0
    pC1D = tree.pC1D
    pC0D = tree.pC0D

    # now for the other set of variables:
    identifierInt = int(identifier)
    for y0, y1, yw in phenotypes:
        fitness, optimalDecision = F(identifierInt, x0, x1, y0, y1, yw, posE0, posE1,
                                     pC1D, pC0D, t, argumentR, argumentP, T,
                                     finalTimeArr)

        currKey = '%s;%s;%s;%s;%s;%s;%s' % (identifier, x0, x1, y0, y1, yw, t)
        policyStar[currKey] = (
            fitness, optimalDecision, pC1D, posE1)

    return policyStar


def printKeys(myShelve):
    for key in myShelve.keys()[0:10]:
        print key
        print myShelve[key]


def run(pE0, pC0E0, T, adultT, markov_chain, transition_probability, argumentR, argumentP, kwUncertainTime, beta,
        psi_weightingParam, batchSize, dataPath):
    # this is set to global so that it can be accessed from all parallel workers
    global Ft1
    Ft1 = {}
    treeLengthDict = {}
    # if it doesn't exist yet create a fitness directory
    if not os.path.exists('fitness'):
        os.makedirs('fitness')

    tree_path = 'trees'

    cueSet = [0, 1]
    # the organism makes a decision in each of T-1 time periods to specialize towards P0, or P1 or wait
    T = int(T)
    # define input variable space
    # define input variable space
    # define input variable space
    y0Values = np.arange(0, T +1, 1)
    y1Values = np.arange(0, T +1, 1)
    ywValues = np.arange(0, T +1, 1)
    tValues = np.arange(T - 1, -1, -1)
    tValuesForward = np.arange(1, T+1, 1)

    pC0E0, pC1E1 = processCueValidity(pC0E0, np.arange(1, T + 1, 1))
    pE1 = D(str(1)) - D(str(pE0))

    # store the varying cue validities for later
    pickle.dump(pC0E0, open('pC0E0dict.p', 'wb'))
    pickle.dump(pC1E1, open('pC1E1dict.p', 'wb'))
    print pC0E0
    # # defines the random final time array
    # # not really using this at the moment
    # # should go back to this in the future
    finalTimeArrNormal = normData([probFinalTime(t, T, kwUncertainTime) for t in np.arange(0, T+1, 1)], probMinFT, probMaxFT)
    finalTimeArr = {t: v for t, v in zip(np.arange(0, T+1, 1), finalTimeArrNormal)}
    pickle.dump(finalTimeArr, open('finalTimeArr.p', 'wb'))

    """""
    FORWARD PASS
    before calculating fitness values need to perform a forward pass in order to calculate the updated posterior values
    """""
    print "start building tree"
    buildForwardTree(T, cueSet, pE0, pE1, pC0E0, pC1E1, tree_path, transition_probability)
    """""
    Calculate terminal fitness
    the next section section will call a function that uses the last terminal tree to calculate the fitness
    at the final time step
    """""
    startTime = time.clock()
    calcTerminalFitness(tree_path, y0Values, y1Values, ywValues, T, adultT, transition_probability, beta,
                        psi_weightingParam,
                        batchSize)
    elapsedTime = time.clock() - startTime
    # this just to give an indication how much time it takes to calculate terminal fitness (as this will be the
    # most demanding step)
    print 'Elapsed time for calculating terminal fitness: ' + str(elapsedTime)

    """""
    Stochastic dynamic programming via backwards induction 
    applies DP from T-1 to 1
    """""
    # developmental cycle; iterate from t = T-1 to t = 0 (backward induction)
    # step size parameter for loading the trees
    down()
    # this is again to display a progress bar
    widgets = ['Time steps left: ', Percentage(), ' ',
               Bar(marker=('-'), left='[', right=']'), ' ', ETA(), ' ']
    pbarTimer = ProgressBar(widgets=widgets, maxval=T - 1).start()
    for t in pbarTimer(tValues):
        up()
        # move files around to free up disk space
        if T - t >= 2 and t > 15:
            shutil.move('fitness/%s' % (t + 2), dataPath)

        t1 = t + 1
        if not os.path.exists("fitness/%s" % t):
            os.makedirs("fitness/%s" % t)

        allCombinations = (itertools.product(*[y0Values, y1Values, ywValues]))
        phenotypes = list(((y0, y1, yw) for y0, y1, yw in allCombinations if sum([y0, y1, yw]) == t))

        trees = shelve.open(os.path.join(tree_path, "tree%s" % (t1)))
        # store the value in the treeLengthDict for plotting purposes
        treeLengthDict[t1] = len(trees)
        # now I need to split up the trees again

        if len(trees) > batchSize:
            stepsize = batchSize
        else:
            stepsize = len(trees)
        widgets = ['Time step %s:' % t, Percentage(), ' ',
                   Bar(marker=('-'), left='[', right=']'), ' ', ETA(), ' ']
        pbarTimeStep = ProgressBar(widgets=widgets, maxval=int(len(trees) / float(stepsize))).start()

        for i in pbarTimeStep(xrange(stepsize, len(trees) + stepsize, stepsize)):
            if i >= len(trees):
                endCondition = len(trees)
            else:
                endCondition = i

            if len(trees) <= stepsize:
                identTimesTwo = batchSize
                identTimesTwoPlusOne = 2 * batchSize
            else:
                if t == T - 1:
                    identTimesTwo = i
                    identTimesTwoPlusOne = 2 * i
                else:
                    identTimesTwo = int((endCondition - 1) * 2 / float(batchSize)) * batchSize
                    identTimesTwoPlusOne = int(((endCondition - 1) * 2 + batchSize) / float(batchSize)) * batchSize

            if os.path.exists('fitness/%s/TF%s' % (t1, identTimesTwo)):
                currFt1 = shelve.open('fitness/%s/TF%s' % (t1, identTimesTwo))
                if os.path.exists('fitness/%s/TF%s' % (t1, identTimesTwoPlusOne)):
                    currFt1.update(shelve.open('fitness/%s/TF%s' % (t1, identTimesTwoPlusOne)))
                set_global_Ft1(currFt1)
            else:
                currFt1 = pickle.load(open('fitness/%s/TF%s.p' % (t1, identTimesTwo), 'rb'))
                if os.path.exists('fitness/%s/TF%s.p' % (t1, identTimesTwoPlusOne)):
                    currFt1.update(pickle.load(open('fitness/%s/TF%s.p' % (t1, identTimesTwoPlusOne), 'rb')))
                set_global_Ft1(currFt1)

            del currFt1

            # this is the bit that will do the actual fitness calculation
            currFt = {}  # watch out as Ft1 is currently a global variable
            pool = Pool(12)
            key_vallist = {(a, trees[str(a)]) for a in xrange(i - stepsize, endCondition, 1)}

            identifiers, allTrees = zip(*key_vallist)

            results = pool.map(func_star2,
                               itertools.izip(allTrees, identifiers,
                                              itertools.repeat(t), itertools.repeat(phenotypes),
                                              itertools.repeat(finalTimeArr),
                                              itertools.repeat(T), itertools.repeat(argumentR),
                                              itertools.repeat(argumentP)))

            pool.close()
            pool.join()
            for currF in results:
                currFt.update(currF)
            # the next part will store the results
            if t > 15:
                if i == len(trees):
                    lastVal = int((i + batchSize) / float(batchSize)) * batchSize
                    currentFile = os.path.join('fitness/%s' % t, "TF%s" % lastVal)
                else:
                    currentFile = os.path.join('fitness/%s' % t, "TF%s" % i)
                myShelve = shelve.open(currentFile)
                myShelve.update(currFt)
                myShelve.close()
            else:
                if i == len(trees):
                    lastVal = int((i + batchSize) / float(batchSize)) * batchSize
                    currentFile = os.path.join('fitness/%s' % t, "TF%s.p" % lastVal)
                else:
                    currentFile = os.path.join('fitness/%s' % t, "TF%s.p" % i)
                pickle.dump(currFt, open(currentFile, 'wb'))

            set_global_Ft1({})
            del currFt
            time.sleep(0.005)
    # save the treeLengthDict
    pickle.dump(treeLengthDict, open('treeLengthDict.p', 'wb'))

    elapsedTime = time.clock() - startTime
    print 'Elapsed time for the whole run function: ' + str(elapsedTime)
    del Ft1


if __name__ == '__main__':
    # specify parameters

    """
    T is an integer specifying the number of discrete time steps of the model
    T is the only parameter read in via the terminal
    to run the model, type the following in the terminal: python VarCueRelModel.py T
    """

    T = sys.argv[1]
    cueRelMax = 0.95
    optimalPolicyFlag = False  # would you like to compute the optimal policy
    preparePlottingFlag = False  # would you like to prepare the policy for plotting?
    plotFlag = True  # would you like to plot the aggregated data?
    performSimulation = False  # do you need to simulate data based on the "preapredPlotting data" for plotting?; i.e.,

    standardPlots = False
    advancedPlots = True
    # simulate twins and/or mature phenotypes
    autoCorrFlag = False  # do you want to compute autcorrelationDictionaries for the current markov chain
    merge = True  # can be used for merging asymmetric and symmetric probabilities, as well as priors
    exploreAssymetries = False

    # this is the directory where modeling results are stored
    mainPath = "/home/nicole/PhD/PorjectMerger/merge"
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    # depending on the memory capacity of your system you might consider choosing a smaller batch size
    batchSize = 10000  # how large should parallalizable units be

    argumentRArr = ['linear','increasing', 'diminishing']  # 'linear','increasing', 'diminishing'  # string specifying the additive fitness reward; can pass multiple arguments in here
    argumentPArr = ['linear','increasing', 'diminishing']  # , 'increasing', 'diminishing']  # string specifying the additive fitness penalty


    # continue with diminishing and everything else and symmetric

    # add additonal arguments for changing env
    states = ('E0', 'E1')
    markovProbabilities = [(0.95, 0.85),(0.8, 0.70), (0.65, 0.55)]
    #markovProbabilities = [(0.9, 0.9), (0.75, 0.75),(0.6, 0.6)] #

    # all asymmetries
    #markovProbabilities =[(0.9, 0.88),(0.75, 0.73),(0.6,0.58)] #02

    #markovProbabilities = [(0.9, 0.85), (0.75, 0.7),(0.6, 0.55) ]  # underestimate 05

    #markovProbabilities = [(0.95, 0.75),(0.85, 0.65),(0.7, 0.5)] #20

    cueValidityC0E0Arr = [0.55, 0.75,0.95] #0.55, 0.75,
    adultTList = [1,5,20] #1,5,20
    fitnessWeightingArg = 'equal'

    # parameters for the some of the internal functions (specifically the phenotype to fitness mapping),
    # change for specifc adaptations
    beta = 0.2  # beta determines the curvature of the fitness and reward functions
    # for uncertain time == step
    a = 10
    b = 18

    # for uncertain time == function:
    # define your own function
    kwUncertainTime = 'None'
    prob = 0.8  # the probability of dying after t = b
    funFT = lambda x: 0.5 * np.cos(x)  # (1-(x/float(21)))
    probMax = 0.5  # this will define the probability range for reaching the terminal state
    probMin = 0.0

    # parameters for the cue reliability
    if int(T) % 2 == 0:
        invUPeakVar = round(int(T) / 2.0) + 0.5
    else:
        invUPeakVar = round(int(T) / 2.0)  # this is defines where the maximum for the inverted U shape should be
    stepCutoff = 3  # this is a time value
    deprivationRange = 3

    # penalty weighting
    psi_weightingParam = 1.0

    # global policy dict
    set_global_variables(a, b, prob, funFT, probMax, probMin, invUPeakVar, stepCutoff, deprivationRange, cueRelMax)

    """
        first compute the autocorrelation dictionary for the current combinations
        - exact dict for symmetric autocorrelations
        - T dict that shows autocorrelations for simulated chains of environments of length T
        - accurateDict which shows autocorrelations for simulated chains of environments of length 300 
        """
    autoCorrPath = os.path.join(mainPath, 'autoCorrelationDictionaries')
    if autoCorrFlag:
        if not os.path.exists(autoCorrPath):
            os.makedirs(autoCorrPath)

        exactDict = calcAutocorrelation(markovProbabilities, 1)
        pickle.dump(exactDict, open(os.path.join(autoCorrPath, "exact_dict.p"), "wb"))
        accurateDict = calcAutocorrelationSim(markovProbabilities)
        pickle.dump(accurateDict, open(os.path.join(autoCorrPath, "accurate_dict.p"), "wb"))
        tDict = calcAutocorrelationSim(markovProbabilities)
        pickle.dump(tDict, open(os.path.join(autoCorrPath, "t_dict.p"), "wb"))


    # prepare results for different input parameters
    # first make sure that results are saved properly
    priorE0Arr = []#[0.5, 0.3, 0.1]
    # make sure the directories on the HDD (or wherever you have sufficient storage space) are in place for storage of large files

    dataPath = '/mnt/f28c380a-3bef-40eb-8061-de162309442d/programming/varying cue reliabilities'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)


    for adultT in adultTList:
        print adultT
        print "\n"
        if not os.path.exists(os.path.join(mainPath, "%s" % adultT)):
            os.makedirs(os.path.join(mainPath, "%s" % adultT))
        mainPathCurr = os.path.join(mainPath, "%s" % adultT)

        for argumentR in argumentRArr:
            for argumentP in argumentPArr:

                for i,markov_chain in enumerate(markovProbabilities):
                    pE0E0, pE1E1 = markov_chain


                    pE0E1 = D(str(1)) - D(str(pE0E0))
                    pE1E0 = D(str(1)) - D(str(pE1E1))
                    transition_probability = {
                        'E0': {'E0': D(str(pE0E0)), 'E1': D(str(pE0E1))},
                        'E1': {'E0': D(str(pE1E0)), 'E1': D(str(pE1E1))},
                    }
                    # calculate the stationary distribution
                    priorE0 = calcStationaryDist(markov_chain)
                    priorE0Arr.append(priorE0)
                    for cueValidityC0E0 in cueValidityC0E0Arr:
                        print "Run with prior " + str(priorE0) + " " + str(markov_chain)+ " and cue validity " + str(cueValidityC0E0)
                        print "Run with reward " + str(argumentR) + " and penalty " + str(argumentP)

                        # create a folder for that specific combination, also encode the reward and penalty argument
                        currPath = "runTest_%s%s_%s%s_%s" % (argumentR[0], argumentP[0], pE0E0, pE1E1, cueValidityC0E0)

                        if not os.path.exists(os.path.join(mainPathCurr, currPath)):
                            os.makedirs(os.path.join(mainPathCurr, currPath))
                            # set the working directory for this particular parameter combination
                        os.chdir(os.path.join(mainPathCurr, currPath))


                        # create the corresponding folder on the hdd
                        currDataPath = os.path.join(dataPath, currPath)
                        if not os.path.exists(currDataPath):
                            os.makedirs(currDataPath)

                        if optimalPolicyFlag:
                            # calculates the optimal policy
                            run(priorE0, cueValidityC0E0, T, adultT,
                                markov_chain, transition_probability, argumentR, argumentP, kwUncertainTime, beta,
                                psi_weightingParam, batchSize,
                                currDataPath)


                        if preparePlottingFlag:
                            print "prepare plotting"
                            T = int(T)
                            # probability of dying at any given state
                            finalTimeArr = pickle.load(open("finalTimeArr.p", "rb"))
                            pC0E0 = pickle.load(open("pC0E0dict.p", "rb"))
                            pC1E1 = pickle.load(open("pC1E1dict.p", "rb"))

                            pE1 = 1 - priorE0

                            # create plotting folder
                            if not os.path.exists('plotting'):
                                os.makedirs('plotting')

                            """
                            run code that prepares the optimal policy data for plotting
                            preparePlotting will prepare the results from the optimal policy for plotting and store them in the
                            folder for each parameter combination
                            """
                            # set the working directory for this particular parameter combination
                            # set the working directory for this particular parameter combination
                            preparePlotting(os.path.join(dataPath, currPath), T, priorE0, pE1, kwUncertainTime,
                                            finalTimeArr, pC0E0,
                                            pC1E1, batchSize)
                            """
                            Deleting large data files
                            - first from HDD location
                            - then SSD
                            """
                            # can delete he complete folder on the HDD
                            os.chdir(mainPathCurr)
                            shutil.rmtree(currDataPath, ignore_errors=True)
                            # next delete stuff from SSD
                            shutil.rmtree(os.path.join(currPath, 'trees'), ignore_errors=True)
                            shutil.rmtree(os.path.join(currPath, 'fitness'), ignore_errors=True)
                            shutil.rmtree(os.path.join(currPath, 'plotting/StateDistribution'), ignore_errors=True)
                            # empty the trash
                            os.system('trash-empty')
                            print 'Creating data for plots'

                            dataPath2 = 'plotting/aggregatedResults'
                            resultsPath = 'plotting/resultDataFrames'

                            os.chdir(os.path.join(mainPathCurr, currPath))
                            joinIndidividualResultFiles('raw', np.arange(1, T+1, 1), resultsPath)  # raw results
                            joinIndidividualResultFiles('aggregated', np.arange(1, T+1, 1), dataPath2)  # raw results
                            joinIndidividualResultFiles('plotting', np.arange(1, T+1, 1), dataPath2)  # raw results

        """
        Creating plots
        """
        if plotFlag:
            markovProbabilitiesTotal = []

            if len(markovProbabilities) > 5:

                firstHalf = markovProbabilities[0:5]
                secondHalf = markovProbabilities[5:]
                markovProbabilitiesTotal.append(firstHalf)
                markovProbabilitiesTotal.append(secondHalf)
            else:
                markovProbabilitiesTotal.append(markovProbabilities)

            # what kind of plots can we make:
            # 1. policy plot
            # 2. rank order stability
            # 3. twinstudy and experimental twin study with all treatments and lags
            # 4. mature phenotypes and fitness differences

            # possible plotting arguments: Twinstudy, ExperimentalTwinstudy, MaturePhenotypes, FitnessDifference
            dataPath2 = 'plotting/aggregatedResults'
            for argumentR in argumentRArr:
                for argumentP in argumentPArr:
                    for idx in range(len(markovProbabilitiesTotal)):
                        markovProbabilitiesCurr = markovProbabilitiesTotal[idx]

                        os.chdir(mainPathCurr)
                        print "Plot for reward " + str(argumentR) + " and penalty " + str(argumentP)
                        twinResultsPath = "PlottingResults_Half%s_%s_%sNew" % (idx, argumentR[0], argumentP[0])
                        if not os.path.exists(twinResultsPath):
                            os.makedirs(twinResultsPath)


                        r = 0.3  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
                        minProb = 0  # minimal probability of reaching a state for those that are displayed
                        startEnvList = [0, 1]  # ,1]  #1 specify the starting environment

                        numAgents = 5000
                        baselineFitness = 0
                        lag = [5]  # number of discrete time steps that twins are separated
                        endOfExposure = False
                        adoptionType = "yoked"
                        plotVar = False  # do you want to plot variance in phenotypic distances?

                        if standardPlots:

                            # policyPlotReduced(int(T) + 1, r, markovProbabilitiesCurr, cueValidityC0E0Arr,
                            #                   np.arange(1, int(T) + 1, 1), dataPath2, True,
                            #                   argumentR, argumentP, minProb,
                            #                   mainPathCurr, twinResultsPath)
                            plotArgs = ["FitnessDifference"]
                            runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents,
                                     twinResultsPath,
                                     baselineFitness,
                                     mainPathCurr, argumentR, argumentP, lag, adoptionType, endOfExposure, plotArgs,
                                     plotVar,
                                     True, adultT, 1)

                            for startEnv in startEnvList:

                                os.chdir(mainPathCurr)

                                # plotArgs = ["MaturePhenotypes"]  # "MaturePhenotypes",
                                # runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath,
                                #          baselineFitness,
                                #          mainPathCurr, argumentR, argumentP, lag, adoptionType, endOfExposure, plotArgs,
                                #          plotVar,
                                #          performSimulation, adultT, startEnv)

                                # '''
                                # The subsequent code block is specifically for variants of adoption studies
                                # '''
                                # plotArgs = ["Twinstudy"]  # , "BeliefTwinstudy"
                                # for adoptionType in ["yoked"]:  # , "oppositePatch", "deprivation"]:
                                #     print "adoptionType: " + str(adoptionType)
                                #     runPlots(markovProbabilities, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath, baselineFitness,
                                #              mainPathCurr, argumentR,
                                #              argumentP, lag, adoptionType, endOfExposure, plotArgs, plotVar,performSimulation,adultT, startEnv)


    if advancedPlots:
        markovProbabilitiesTotal = []

        if len(markovProbabilities) > 5:

            firstHalf = markovProbabilities[0:5]
            secondHalf = markovProbabilities[5:]
            markovProbabilitiesTotal.append(firstHalf)
            markovProbabilitiesTotal.append(secondHalf)
        else:
            markovProbabilitiesTotal.append(markovProbabilities)
        """
        New plots for analyzing the changing environments model
        1. for each T show three plots in a subplot structure and each plot shows either absolute or
        relative plasticity depending on the argument for each autocorrelation level (absolute vs.
        experienced depending on argument)

        TODO: add variable which indicates which plots you would like to produce
        """

        for argumentR in argumentRArr:
            for argumentP in argumentPArr:

                os.chdir(mainPath)
                twinResultsAggregatedPath = "PlottingResults_Aggr_%s_%s" % (argumentR[0], argumentP[0])
                if not os.path.exists(twinResultsAggregatedPath):
                    os.makedirs(twinResultsAggregatedPath)
                startEnvList = [0,1]  #0


                autoCorrArg = 'absolute'  # or "experienced'
                adoptionType = "yoked"
                lag = None
                endOfExposure = False
                studyArg = 'Twinstudy'
                coarseArg = True  # do you want to aggregate autoCorrelations into low, medium, high (is an aggregate better than just picking one?)
                normalize = True  # whether or not to normalize values in heatplots to makse differences easier visible

                if merge:
                    mergeArr = ['symmetric', 'asymmetric (E0)','asymmetric (E1)'] # [0.5, 0.7,0.9] #pE1
                    mergeName = 'transition probabilities & starting environment' # or "prior"

                    runAggregatePlotsMerge(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T), adultTList, startEnvList,
                                      "autoCorrelationDictionaries",
                                      twinResultsAggregatedPath, mainPath, argumentR, argumentP, autoCorrArg,
                                      adoptionType, lag, endOfExposure, studyArg, coarseArg,
                                      normalize, mergeArr, mergeName)

                elif exploreAssymetries:

                    mergeArr = ['0','02', '05', '10', '20']
                    mergeName = 'asymmetry' # or "prior"

                    runAggregatePlotsAssymetries(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T)+1, adultTList, startEnvList,
                                      "autoCorrelationDictionaries",
                                      twinResultsAggregatedPath, mainPath, argumentR, argumentP, autoCorrArg,
                                      adoptionType, lag, endOfExposure, studyArg, coarseArg,
                                      normalize, mergeArr, mergeName)

                else:
                    """
                    if necessary first look at the autocorrelation dictiornary
                    """
                    autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 'exact_dict.p'), "rb"))
                    autoCorrDictAccurate = pickle.load(open(os.path.join(autoCorrPath, 'accurate_dict.p'), "rb"))

                    for key, value in autoCorrDict.items():
                        if not value:
                            autoCorrDict[key] = autoCorrDictAccurate[key]
                    levelsAutoCorrToPlot = sorted(autoCorrDict.values())
                    # levelsAutoCorrToPlot =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  #or put it in None for an aggregate[0.1,0.5,0.9]

                    runAggregatePlots(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T), adultTList, startEnvList,
                                      autoCorrPath,
                                      twinResultsAggregatedPath, mainPath, argumentR, argumentP, autoCorrArg,
                                      adoptionType, lag, endOfExposure, studyArg, coarseArg, levelsAutoCorrToPlot,
                                      normalize)

