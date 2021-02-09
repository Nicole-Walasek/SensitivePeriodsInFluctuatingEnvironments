# replication of Willem's 2016 paper "The evolution of sensitive periods in a model of incremental development"
import sys
import numpy as np
import itertools
from numpy import random
import cPickle as pickle
import time
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from PreparePlottingVarCueValidityParallel import preparePlotting
from PlottingParallel import policyPlotReduced, joinIndidividualResultFiles
from twinStudiesVarCueFixedStartingEnv import runPlots, runAggregatePlots, runAggregatePlotsMerge, runAggregatePlotsAssymetries
from sympy.utilities.iterables import multiset_permutations
from extractAutoCorrelation import calcAutocorrelation, calcAutocorrelationSim
from decimal import Decimal as D


# Parameter description
# x0 : number of cues indicating Environment 0 at time t
# x1 : number of cues indicating Environment 1 at time t
# y0 : number of developmental steps at time t towards phenotype 0
# y1 : number of developmental steps at time t towards phenotype 1
# yw : number of developmental steps at time t in which the organism waited


# function to set global variables
def set_global_variables(aVar, bVar, probVar, funVar, probMax, probMin):
    # beta defines the curvature of the reward and penalty functions
    global aFT
    global bFT
    global probFT
    global funFT
    global probMaxFT
    global probMinFT
    global Ft1

    aFT = aVar
    bFT = bVar
    probFT = probVar
    funFT = funVar
    probMaxFT = probMax
    probMinFT = probMin


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
    rel_tol = float(rel_tol)
    abs_tol = float(abs_tol)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def findOptimumClose(fitnesList):
    # fitnesList corresponds to [F0,F1,Fw]
    # determine whether list contains ties
    F0 = float(fitnesList[0])
    F1 = float(fitnesList[1])
    Fw = float(fitnesList[2])
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
    else:
        print "cue reliability is not a number"
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


def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    x0, x1, y0, y1, yw = state

    if argumentR == 'linear':
        phiVar = b0_D * y0 + b1_D * y1

    elif argumentR == 'diminishing':
        alphaRD = (T) / float(1 - float(np.exp(-beta * (T))))
        alphaRD = alphaRD
        phiVar = b0_D * alphaRD * (1 - np.exp(-beta * y0)) + b1_D * alphaRD * (1 - np.exp(-beta * y1))

    elif argumentR == 'increasing':
        alphaRI = (T) / float(float(np.exp(beta * (T))) - 1)
        alphaRI = alphaRI
        phiVar = b0_D * alphaRI * (np.exp(beta * y0) - 1) + b1_D * alphaRI * (np.exp(beta * y1) - 1)
    else:
        print 'Wrong input argument to additive fitness reward function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    if argumentP == 'linear':
        psiVar = -(b0_D * y1 + b1_D * y0)

    elif argumentP == 'diminishing':
        alphaPD = (T) / float(1 - float(np.exp(-beta * (T))))
        alphaPD = alphaPD
        psiVar = -(b0_D * alphaPD * (1 - np.exp(-beta * y1)) + b1_D * alphaPD * (1 - np.exp(-beta * y0)))

    elif argumentP == 'increasing':
        alphaPI = (T) / float(float(np.exp(beta * (T))) - 1)
        alphaPI = alphaPI
        psiVar = -(b0_D * alphaPI * (np.exp(beta * y1) - 1) + b1_D * alphaPI * (np.exp(beta * y0) - 1))
    else:
        print 'Wrong input argument to additive fitness penalty function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    tf = 0 + phiVar + psi_weighting * psiVar

    return float(tf)


# terminal fitness function

def terminalFitness(state, adultT, markov_chain, fitnessWeighting, pE0, pE1, argumentR, argumentP, T, beta,
                    psi_weighting):
    """
    in the changing environments model we need to loop through the adult life span and calculate fitness at every time point
    """
    # transform the markov chain dictionary into a matrix
    b0_D = pE0
    b1_D = pE1

    P = np.array(
        [[markov_chain['E0']['E0'], markov_chain['E0']['E1']], [markov_chain['E1']['E0'], markov_chain['E1']['E1']]])
    tfList = []  # this will hold all fitness values across the adult lifespan
    for t in np.arange(1, adultT + 1, 1):
        tfCurr = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)
        tfList.append(float(tfCurr))
        # recalculate the distribution in the markov chain after one time step
        b0_D, b1_D = np.dot([float(pE0), float(pE1)], np.linalg.matrix_power(P, t))

    tfList = np.array(tfList)

    return np.sum(tfList)


def fwd(observations, states, start_prob, trans_prob, emm_prob):
    # forward part of the algorithm
    f_prev = {}  # these are joint probabilities

    for i, observation_i in enumerate(observations):
        f_curr = dict(zip(states, [
            emm_prob[st][observation_i] * start_prob[st] if i == 0 else emm_prob[st][observation_i] * sum(
                f_prev[k] * trans_prob[k][st] for k in states) for st in states]))

        f_prev = f_curr

    f_mat = np.array([f_curr[st] for st in states])

    return f_mat


def func_star2(allArgs):
    return posteriorUpdatingMarkov(*allArgs)


def posteriorUpdatingMarkov(data, states, start_prob, trans_prob, em_prob):
    posteriorTree = {}
    x0, x1 = data

    if x0 == x1 == 0:
        normalizedRes = np.array([start_prob['E0'], start_prob['E1']])

    else:
        observationsAll = multiset_permutations(['0'] * x0 + ['1'] * x1)
        result = [fwd(observation, states, start_prob, trans_prob, em_prob) for observation in observationsAll]
        result = sum(result)
        normalizedRes = result / (sum(result))  # sum(result) is the probability of D

    pC0D = em_prob["E0"]["0"] * normalizedRes[0] + em_prob["E1"]["0"] * normalizedRes[1]
    pC1D = em_prob["E0"]["1"] * normalizedRes[0] + em_prob["E1"]["1"] * normalizedRes[1]
    posteriorTree[data] = (tuple([float(normalizedRes[0]), float(normalizedRes[1])]), tuple([float(pC0D), float(pC1D)]))

    return posteriorTree


"""
- possible posteriors at the end of ontogeny won't change regardless of adult lifespan
- therefore store on tree-dictionary per time step 
"""


def buildPosteriorTree(x0Values, x1Values, emission_prob, states, start_probability, transition_probability, T,
                       tree_path):
    tValues = np.arange(T, -1, -1)
    for t in tValues:  # range of tValues is 0 till T-1, because we also want to store the prior
        posteriorTree = {}
        allCombinations = (itertools.product(*[x0Values, x1Values]))
        dataSets = list(((x0, x1) for x0, x1 in allCombinations if sum([x0, x1]) == t))

        pool = Pool(32)

        result = pool.map(func_star2,
                          itertools.izip(dataSets,
                                         itertools.repeat(states), itertools.repeat(start_probability),
                                         itertools.repeat(transition_probability),
                                         itertools.repeat(emission_prob)))
        pool.close()
        pool.join()

        for res in result:
            posteriorTree.update(res)

        pickle.dump(posteriorTree, open(os.path.join(tree_path, "tree%s.p" % (t)), 'wb'))
    del posteriorTree


def calcTerminalFitness(tree_path, y0Values, y1Values, ywValues, T, beta, argumentR, argumentP, psi_weightingParam,
                        adultT, markov_chain, fitnessWeighting):
    # if it doesn't exist yet create a directory for the final timestep

    trees = pickle.load(open(os.path.join(tree_path, "tree%s.p" % (T)), "rb"))
    print "Number of binary trees in the last step: " + str(len(trees))
    terminalF = {}
    for D in trees.keys():
        x0, x1 = D
        posE0, posE1 = trees[D][0]
        allCombinations = (itertools.product(*[y0Values, y1Values, ywValues]))
        phenotypes = list(((y0, y1, yw) for y0, y1, yw in allCombinations if sum([y0, y1, yw]) == T))

        for y0, y1, yw in phenotypes:
            currKey = '%s;%s;%s;%s;%s;%s' % (x0, x1, y0, y1, yw, T)

            terminalF[currKey] = terminalFitness((x0, x1, y0, y1, yw), adultT, markov_chain, fitnessWeighting, posE0,
                                                 posE1, argumentR,
                                                 argumentP, T, beta, psi_weightingParam)
    return terminalF


# @profile
def F(x0, x1, y0, y1, yw, posE0, posE1, pC1D, pC0D, t, argumentR, argumentP, T, finalTimeArr, adultT, markovChain,
      fitnessWeighting):
    """
    this is the core of SDP
    """

    posE0, posE1, pC1D, pC0D = float(posE0), float(posE1), float(pC1D), float(pC0D)
    # computes the optimal decision in each time step
    # computes the maximal expected fitness for decision made between time t and T for valid combinations
    # of environmental variables
    prFT = float(finalTimeArr[t + 1])
    # for generating agents
    # SECOND: calculate expected fitness at the end of lifetime associated with each of the developmental decisions
    if prFT != 0:
        TF = prFT * terminalFitness((x0, x1, y0, y1, yw), adultT, markovChain, fitnessWeighting, posE0, posE1,
                                    argumentR, argumentP, T)
    else:
        TF = 0

    t1 = t + 1

    if t == T - 1:
        currKey1 = '%s;%s;%s;%s;%s;%s' % (x0, x1, y0 + 1, y1, yw, t1)

        currKey3 = '%s;%s;%s;%s;%s;%s' % (x0, x1, y0, y1 + 1, yw, t1)

        currKey5 = '%s;%s;%s;%s;%s;%s' % (x0, x1, y0, y1, yw + 1, t1)

        F0 = Ft1[currKey1]
        F1 = Ft1[currKey3]
        Fw = Ft1[currKey5]
    else:
        currKey1 = '%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0 + 1, y1, yw, t1)
        currKey2 = '%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0 + 1, y1, yw, t1)

        currKey3 = '%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0, y1 + 1, yw, t1)
        currKey4 = '%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0, y1 + 1, yw, t1)

        currKey5 = '%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0, y1, yw + 1, t1)
        currKey6 = '%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0, y1, yw + 1, t1)

        F0 = ((pC0D) * (Ft1[currKey1][0]) + (pC1D) * (Ft1[currKey2][0]))
        F1 = ((pC0D) * (Ft1[currKey3][0]) + (pC1D) * (Ft1[currKey4][0]))
        Fw = ((pC0D) * (Ft1[currKey5][0]) + (pC1D) * (Ft1[currKey6][0]))

    maxF, maxDecision = findOptimumClose([F0, F1, Fw])
    return float(maxF), maxDecision  # in order to track the current beliefs


def oneFitnessSweep(tree, t, phenotypes, finalTimeArr, T, argumentR, argumentP, adultT, markovChain, fitnessWeighting):
    # iterate through D in tree; for each D and phenotype combo store  the oprimal decision and accordingly
    # create a new fitness function but this paramerter can be stored later

    policyStar = {}
    for D in tree.keys():

        x0, x1 = D
        posE0, posE1 = tree[D][0]
        pC0D, pC1D = tree[D][1]

        for y0, y1, yw in phenotypes:
            fitness, optimalDecision = F(x0, x1, y0, y1, yw, posE0, posE1,
                                         pC1D, pC0D, t, argumentR, argumentP, T,
                                         finalTimeArr, adultT, markovChain, fitnessWeighting)

            currKey = '%s;%s;%s;%s;%s;%s' % (x0, x1, y0, y1, yw, t)
            policyStar[currKey] = (
                fitness, optimalDecision, pC1D, posE1)

    return policyStar


def printKeys(myShelve):
    for key in myShelve.keys()[0:10]:
        print key
        print myShelve[key]


def run(priorE0, pC0E0, T, argumentR, argumentP, kwUncertainTime, beta, psi_weightingParam, states,
        transition_probability, buildPosteriorTreeFlag, adultT, fitnessWeighting):
    # this is set to global so that it can be accessed from all parallel workers
    global Ft1
    Ft1 = {}
    treeLengthDict = {}

    # if it doesn't exist yet create a fitness directory
    if not os.path.exists('fitness'):
        os.makedirs('fitness')

    tree_path = 'trees'
    if not os.path.exists(tree_path):
        os.makedirs(tree_path)

    # the organism makes a decision in each of T-1 time periods to specialize towards P0, or P1 or wait
    T = int(T)

    # define input variable space
    y0Values = np.arange(0, T + 1, 1)
    y1Values = np.arange(0, T + 1, 1)
    ywValues = np.arange(0, T + 1, 1)

    x0Values = np.arange(0, T + 1, 1)
    x1Values = np.arange(0, T + 1, 1)

    tValues = np.arange(T - 1, -1, -1)
    tValuesForward = np.arange(1, T + 1, 1)

    pC1E0 = float(D(1) - D(pC0E0))  # TODO with this model we can also explore assymteric cue reiabilities
    pC1E0 = float(D(1) - D(pC0E0))  # TODO with this model we can also explore assymteric cue reiabilities

    """
    prepare the required parameters for the HMM algorithm 
    """
    pE1 = float(D(1) - D(priorE0))
    start_probability = {'E0': priorE0,
                         'E1': pE1}
    emission_probability = {  # these are the cue reliabilities
        'E0': {'0': pC0E0, '1': pC1E0},
        'E1': {'0': pC1E0, '1': pC0E0},
    }

    pickle.dump(emission_probability, open("emission_probability.p", "wb"))

    # TODO work on this later
    # defines the random final time array
    # not really using this at the moment
    # should go back to this in the future
    finalTimeArrNormal = normData([probFinalTime(t, T, kwUncertainTime) for t in tValuesForward], probMinFT, probMaxFT)
    finalTimeArr = {t: v for t, v in zip(tValuesForward, finalTimeArrNormal)}
    pickle.dump(finalTimeArr, open('finalTimeArr.p', 'wb'))

    """""
    FORWARD PASS
    before calculating fitness values need to perform a forward pass in order to calculate the updated posterior values
    specifies a mapping from cue set combinations to posterior probabilities  
    
    """""
    if buildPosteriorTreeFlag:
        startTime = time.clock()
        print "start building tree"
        buildPosteriorTree(x0Values, x1Values, emission_probability, states, start_probability, transition_probability,
                           T, tree_path)
        elapsedTime = time.clock() - startTime
        print 'Elapsed time for computing the posterior tree: ' + str(elapsedTime)
    """""
    Calculate terminal fitness
    the next section section will call a function that uses the last terminal tree to calculate the fitness
    at the final time step
    """""
    print "calculate fitness for the last step"
    startTime = time.clock()

    if not os.path.exists("fitness/%s" % T):
        os.makedirs("fitness/%s" % T)

    terminalFitness = calcTerminalFitness(tree_path, y0Values, y1Values, ywValues, T, beta, argumentR, argumentP,
                                          psi_weightingParam, adultT, transition_probability, fitnessWeighting)

    pickle.dump(terminalFitness, open(os.path.join('fitness/%s' % T, "TF.p"), 'wb'))
    elapsedTime = time.clock() - startTime
    print 'Elapsed time for calculating terminal fitness: ' + str(elapsedTime)
    """""
    Stochastic dynamic programming via backwards induction 
    applies DP from T-1 to 1
    """""

    # developmental cycle; iterate from t = T-1 to t = 0 (backward induction)
    # step size parameter for loading the trees

    for t in tValues:
        # IMPORTANT: note, that the t is referring to the sum of the phenotypic state variables, i.e., t+1 cues have been
        # sampled
        print "currently computing the optimal policy for time step %s" % t

        t1 = t + 1

        if not os.path.exists("fitness/%s" % t):
            os.makedirs("fitness/%s" % t)

        allCombinations = (itertools.product(*[y0Values, y1Values, ywValues]))
        phenotypes = list(((y0, y1, yw) for y0, y1, yw in allCombinations if sum([y0, y1, yw]) == t))

        trees = pickle.load(open(os.path.join(tree_path, "tree%s.p" % (t1)), "rb"))
        # store the value in the treeLengthDict for plotting purposes
        treeLengthDict[t] = len(trees)

        if os.path.exists('fitness/%s/TF.p' % (t1)):
            currFt1 = pickle.load(open('fitness/%s/TF.p' % (t1), 'rb'))
            set_global_Ft1(currFt1)  # make the global fitness function available

        del currFt1

        # this is the bit that will do the actual fitness calculation

        """
        call the dynamic programming function here
        """
        currFt = oneFitnessSweep(trees, t, phenotypes, finalTimeArr, T, argumentR, argumentP, adultT,
                                 transition_probability, fitnessWeighting)

        """
        store the current fitness / policy function
        """
        currentFile = os.path.join('fitness/%s' % t, "TF.p")
        pickle.dump(currFt, open(currentFile, 'wb'))

        set_global_Ft1({})
        del currFt
    del Ft1


def fitnessWeightingFunc(adultT, fitnessWeightingArg):
    if fitnessWeightingArg == 'equal':
        fitnessWeighting = [1] * adultT
    else:
        print "unknown argument for fitness weighting"
    return fitnessWeighting


def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx


def calcStationaryDist(markov_chain):
    pE0E0, pE1E1 = markov_chain
    pE0E1 = float(D(1) - D(pE0E0))
    pE1E0 = float(D(1) - D(pE1E1))

    P = np.array([[pE0E0, pE0E1], [pE1E0, pE1E1]])

    w, v = np.linalg.eig(P.transpose())
    oneIdx = find_nearest(w, 1)
    # this corresponds to the overall prior
    # the starting distribution does however matter for the actual behavior over time
    pE0, pE1 = v[:, oneIdx] / float(sum(v[:, oneIdx]))

    return pE0


if __name__ == '__main__':
    # specify parameters

    """
    T is an integer specifying the number of discrete time steps of the model
    T is the only parameter read in via the terminal
    to run the model, type the following in the terminal: python VarCueRelModel.py T
    """

    T = sys.argv[1]

    optimalPolicyFlag = False  # would you like to compute the optimal policy
    buildPosteriorTreeFlag = False  # would you like compute posterior probabilities in HMM for all possible cue sets
    preparePlottingFlag = False  # would you like to prepare the policy for plotting?
    plotFlag = True  # would you like to plot the aggregated data?
    standardPlots = False
    advancedPlots = True
    performSimulation = False  # do you need to simulate data based on the "preapredPlotting data" for plotting?; i.e.,
    # simulate twins and/or mature phenotypes
    autoCorrFlag = False   # do you want to compute autcorrelationDictionaries for the current markov chain
    merge = False #can be used for merging asymmetric and symmetric probabilities, as well as priors
    exploreAssymetries = False
    # this is the directory where modeling results are stored

    mainPath = "/home/nicole/PhD/Project 2 changing environments/asymmetricFinal10_All"
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    # new form of input parameters for HMM

    states = ('E0', 'E1')

    # define the probabilities to remain in the respective environment

    """
    the following markov chains will be for matching to stationary dist
    """
    # 0.5
    # markovProbabilities = [(0.73, 0.73), (0.74, 0.74), (0.75, 0.75), (0.76, 0.76), (0.77, 0.77), (0.79, 0.79),
    #                        (0.8, 0.8), (0.81, 0.81), (0.82, 0.82), (0.84, 0.84), (0.85, 0.85), (0.86, 0.86),
    #                        (0.88, 0.88), (0.89, 0.89), (0.94, 0.94)]

    # # 0.7
    # markovProbabilities = [(0.62, 0.84), (0.63, 0.84), (0.65, 0.85), (0.67, 0.86), (0.68, 0.86), (0.69, 0.87),
    #                        (0.7, 0.87), (0.72, 0.88), (0.74, 0.89), (0.77, 0.9), (0.79, 0.91), (0.81, 0.92),
    #                        (0.84, 0.93), (0.86, 0.94), (0.93, 0.97)]

    # # 0.9
    # markovProbabilities = [(0.53, 0.95), (0.54, 0.95), (0.55, 0.95), (0.56, 0.95), (0.57, 0.95), (0.62, 0.96),
    #                        (0.63, 0.96), (0.64, 0.96), (0.65, 0.96), (0.72, 0.97), (0.73, 0.97), (0.74, 0.97),
    #                        (0.81, 0.98), (0.82, 0.98), (0.91, 0.99)]

    # markovProbabilities = [(0.95, 0.95), (0.9, 0.9), (0.85, 0.85), (0.8, 0.8), (0.75, 0.75), (0.7, 0.7), (0.65, 0.65),
    #                    (0.6, 0.6), (0.55, 0.55), (0.5, 0.5)]


    # markovProbabilities =[(0.95, 0.93), (0.9, 0.88), (0.85, 0.83), (0.8, 0.78), (0.75, 0.73), (0.7, 0.68), (0.65, 0.63), (0.6, 0.58),
    #                        (0.55, 0.53)]

    #markovProbabilities = [(0.95, 0.90), (0.9, 0.85), (0.85, 0.80), (0.8, 0.75), (0.75, 0.70), (0.7, 0.65),
    #                        (0.65, 0.60), (0.6, 0.55), (0.55, 0.50)]

    markovProbabilities = [(0.95, 0.85), (0.9, 0.80), (0.85, 0.75), (0.8, 0.70), (0.75, 0.65), (0.7, 0.60),(0.65, 0.55), (0.6, 0.50)]

    #markovProbabilities = [(0.95, 0.75), (0.9, 0.70), (0.85, 0.65), (0.8, 0.60), (0.75, 0.55), (0.7, 0.50)]

    # corresponds to the probability of receiving C0 when in E0
    cueValidityC0E0Arr = [0.55, 0.75, 0.95]

    """
    TODO continue the previous run with 40 and all combinations 
    """
    argumentRArr =['increasing']#'linear','diminishing','increasing'] #
    argumentPArr = ['linear','diminishing','increasing']  #

    adultTList =[1,5,20] #1,5,20
    fitnessWeightingArg = 'equal'  # TODO: make this more flexible, declining, increasing, invertedU

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

    # penalty weighting
    psi_weightingParam = 1

    # global policy dict
    set_global_variables(a, b, prob, funFT, probMax, probMin)

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
        accurateDict = calcAutocorrelationSim(markovProbabilities, 1, 500, 2000)
        pickle.dump(accurateDict, open(os.path.join(autoCorrPath, "accurate_dict.p"), "wb"))
        tDict = calcAutocorrelationSim(markovProbabilities, 1, 10, 2000)
        pickle.dump(tDict, open(os.path.join(autoCorrPath, "t_dict.p"), "wb"))

    # prepare results for different input parameters
    # first make sure that results are saved properly

    for adultT in adultTList:
        print adultT
        print "\n"
        if not os.path.exists(os.path.join(mainPath, "%s" % adultT)):
            os.makedirs(os.path.join(mainPath, "%s" % adultT))
        fitnessWeighting = fitnessWeightingFunc(adultT, fitnessWeightingArg)
        mainPathCurr = os.path.join(mainPath, "%s" % adultT)
        for argumentR in argumentRArr:
            for argumentP in argumentPArr:

                for markov_chain in markovProbabilities:
                    pE0E0, pE1E1 = markov_chain

                    pE0E1 = D(1) - D(pE0E0)
                    pE1E0 = 1 - D(pE1E1)
                    transition_probability = {
                        'E0': {'E0': float(pE0E0), 'E1': float(pE0E1)},
                        'E1': {'E0': float(pE1E0), 'E1': float(pE1E1)},
                    }

                    # calculate the stationary distribution
                    priorE0 = calcStationaryDist(markov_chain)


                    for cueValidityC0E0 in cueValidityC0E0Arr:

                        print "Run with pE0E0 " + str(pE0E0) + " and pE1E1 " + str(pE1E1) + " and cue validity " + str(
                            cueValidityC0E0)
                        print "Run with reward " + str(argumentR) + " and penalty " + str(argumentP)

                        # create a folder for that specific combination, also encode the reward and penalty argument
                        currPath = "runTest_%s%s_%s%s_%s" % (argumentR[0], argumentP[0], pE0E0, pE1E1, cueValidityC0E0)

                        if not os.path.exists(os.path.join(mainPathCurr, currPath)):
                            os.makedirs(os.path.join(mainPathCurr, currPath))
                        # set the working directory for this particular parameter combination
                        os.chdir(os.path.join(mainPathCurr, currPath))

                        if optimalPolicyFlag:
                            # calculates the optimal policy
                            run(priorE0, cueValidityC0E0, T, argumentR, argumentP, kwUncertainTime, beta,
                                psi_weightingParam, states, transition_probability, buildPosteriorTreeFlag, adultT,
                                fitnessWeighting)

                        if preparePlottingFlag:
                            print "prepare plotting"
                            T = int(T)
                            # probability of dying at any given state
                            finalTimeArr = pickle.load(open("finalTimeArr.p", "rb"))
                            pE1 = float(D(1) - D(priorE0))

                            emission_prob = pickle.load(open("emission_probability.p", "rb"))

                            # create plotting folder
                            if not os.path.exists('plotting'):
                                os.makedirs('plotting')

                            """
                            run code that prepares the optimal policy data for plotting
                            preparePlotting will prepare the results from the optimal policy for plotting and store them in the
                            folder for each parameter combination
                            """
                            preparePlotting(T, priorE0, pE1, kwUncertainTime,
                                            finalTimeArr,
                                            emission_prob)  # took out batch size from prepare plotting and changed cue reliabilities to emission probabilities

                            print 'Creating data for plots'

                            dataPath2 = 'plotting/aggregatedResults'
                            resultsPath = 'plotting/resultDataFrames'
                            os.chdir(os.path.join(mainPathCurr, currPath))

                            joinIndidividualResultFiles('raw', np.arange(1, T + 1, 1), resultsPath)  # raw results
                            joinIndidividualResultFiles('aggregated', np.arange(1, T + 1, 1), dataPath2)  # raw results
                            joinIndidividualResultFiles('plotting', np.arange(1, T + 1, 1), dataPath2)  # raw results


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

                        numAgents = 5000
                        baselineFitness = 0
                        lag = [3]  # number of discrete time steps that twins are separated
                        endOfExposure = False
                        adoptionType = "yoked"
                        plotVar = False  # do you want to plot variance in phenotypic distances?
                        startEnvList = [0,1]#,1]  #1 specify the starting environment

                        if standardPlots:
                            for startEnv in startEnvList:
                                policyPlotReduced(int(T) + 1, r, markovProbabilitiesCurr, cueValidityC0E0Arr,
                                                 np.arange(1, int(T) + 1, 1), dataPath2, True,
                                                 argumentR, argumentP, minProb,
                                                 mainPathCurr, twinResultsPath)
                                os.chdir(mainPathCurr)
                                plotArgs = ["FitnessDifference","MaturePhenotypes"]  # "MaturePhenotypes",
                                runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath,
                                         baselineFitness,
                                         mainPathCurr, argumentR, argumentP, lag, adoptionType, endOfExposure, plotArgs,
                                         plotVar,
                                         performSimulation, adultT, startEnv)

                                '''
                                The subsequent code block is specifically for variants of adoption studies
                                '''
                                plotArgs = ["Twinstudy"]  # ,"BeliefTwinstudy"]
                                for adoptionType in ["yoked"]:  # , "oppositePatch","deprivation"]:
                                    print "adoptionType: " + str(adoptionType)
                                    runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents,
                                             twinResultsPath,
                                             baselineFitness,
                                             mainPathCurr, argumentR,
                                             argumentP, lag, adoptionType, endOfExposure, plotArgs, plotVar,
                                             performSimulation,
                                             adultT, startEnv)

                                """
                                Code for plotting variance around the plasticity curve
                                Code for plotting the belieftwinStudy

                                Note: no need to perform simulations agains after this has been done for the twinStudy
                                """
                                runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath,
                                         baselineFitness,
                                         mainPathCurr, argumentR,
                                         argumentP, lag, "yoked", endOfExposure, ["Twinstudy"], True, False, adultT,
                                         startEnv)

                                runPlots(markovProbabilitiesCurr, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath,
                                         baselineFitness,
                                         mainPathCurr, argumentR,
                                         argumentP, lag, "yoked", endOfExposure, ['BeliefTwinstudy'], False, False, adultT,
                                         startEnv)

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

                for startEnv in startEnvList:
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

                        runAggregatePlotsMerge(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T), adultTList, startEnv,
                                          "autoCorrelationDictionaries",
                                          twinResultsAggregatedPath, mainPath, argumentR, argumentP, autoCorrArg,
                                          adoptionType, lag, endOfExposure, studyArg, coarseArg,
                                          normalize, mergeArr, mergeName)

                    elif exploreAssymetries:

                        mergeArr = ['0','02', '05', '10', '20']
                        mergeName = 'asymmetry' # or "prior"
                        for startEnv in startEnvList:

                            runAggregatePlotsAssymetries(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T), adultTList, startEnv,
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

                        runAggregatePlots(markovProbabilitiesTotal, cueValidityC0E0Arr, int(T), adultTList, startEnv,
                                          autoCorrPath,
                                          twinResultsAggregatedPath, mainPath, argumentR, argumentP, autoCorrArg,
                                          adoptionType, lag, endOfExposure, studyArg, coarseArg, levelsAutoCorrToPlot,
                                          normalize)
