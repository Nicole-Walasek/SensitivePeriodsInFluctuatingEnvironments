import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import cPickle as pickle
import ternary
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from MarkovChain import MarkovChain, HiddenMarkovChain
from sympy.utilities.iterables import multiset_permutations
from PlottingParallel import policyPlotReducedOverview2, policyPlotReducedMerge, policyPlotReducedMergeBW
import operator

"""""
This script will run experimental twin studies to test how sensitive different optimal policies are to cues
    in these experiments I simulate agents in specific environments based on the previously calculated optimal policies
    importantly I will try to use the finalRAW file rather than the dictionary files 
    
    procedure:
    - simulate twins who are identical up to time period t 
    - keep one twin ("original") in its natal patch
    - send the other twin ("doppelgaenger") to mirror patch 
    - doppelgaenger receives opposite (yoked) cues from the original twin
        the cues are opposite but not from the opposite patch
"""""


def setGlobalPolicy(policyPath):
    global policy
    policy = pd.read_csv(policyPath, index_col=0).reset_index(drop=True)


def func_star(allArgs):
    return simulateTwins(*allArgs)


def func_star2(allArgs):
    return simulateExperimentalTwins(*allArgs)


def updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker):
    optDecisions = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                      subDF['yw'] == phenotypeTracker[idx, 2])
                              ]['cStar'].item() for idx in
                    simValues]

    # additionally keep track of the posterior belief
    posBelief = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                   subDF['yw'] == phenotypeTracker[idx, 2])
                           ]['pE1'].item() for idx in
                 simValues]
    # post process optimal decisions
    optDecisionsNum = [
        int(a) if not '(' in str(a) else int(np.random.choice(str(a).replace("(", "").replace(")", "").split(","))) for
        a in
        optDecisions]
    # update phenotype tracker
    idx0 = [idx for idx, val in enumerate(optDecisionsNum) if val == 0]
    if idx0:
        phenotypeTracker[idx0, 0] += 1

    idx1 = [idx for idx, val in enumerate(optDecisionsNum) if val == 1]
    if idx1:
        phenotypeTracker[idx1, 1] += 1

    idx2 = [idx for idx, val in enumerate(optDecisionsNum) if val == 2]
    if idx2:
        phenotypeTracker[idx2, 2] += 1

    return phenotypeTracker, posBelief


def simulateExperimentalTwins(tAdopt, twinNum, env, lag, T, adoptionType, endOfExposure, transitionProb, emissionProb):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability
    :return: phenotypic distance between pairs of twins
    """

    T = T + lag - 1

    if env == 1:
        startEnv = "E1"  # take the very first cue reliability
        start_probability = {'E0': 0,
                             'E1': 1}
    else:
        startEnv = "E0"
        start_probability = {'E0': 1,
                             'E1': 0}

    states = ('E0', 'E1')

    """
    compute an exact behavior if feasible   
    """
    if isinstance(twinNum, list):
        allCues = np.stack(twinNum)
        twinNum = len(twinNum)

    else:
        HMM = HiddenMarkovChain(transitionProb, emissionProb)
        allCues = np.stack([HMM.generate_states(startEnv, T) for x in np.arange(twinNum)])  # shape agentsxtime

    cueProbabilities = []
    result = [fwd(observation, states, start_probability, transitionProb, emissionProb) for
              observation in allCues]
    cueProbabilities += [sum(curr) for curr in result]

    tValues = np.arange(1, tAdopt, 1)

    cues = [int(cue) for cue in allCues[:, 0]]
    cues = np.array(cues)

    # need to reverse the last update
    if adoptionType == "yoked":
        oppositeCues = 1 - cues

    elif adoptionType == "oppositePatch":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])  # TODO fix the opposite cues
        oppositeCues = np.array(oppositeCues)
    elif adoptionType == "deprivation":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
        oppositeCues = np.array(oppositeCues)

    else:
        print "wrong input argument to adoption type!"
        exit(1)

    cueTracker = np.zeros((twinNum, 2))
    cueTracker[:, 0] = 1 - cues
    cueTracker[:, 1] = cues
    phenotypeTracker = np.zeros((twinNum, 3))

    simValues = np.arange(0, twinNum, 1)

    for t in tValues:
        # now we have to recompute this for every timestep

        np.random.seed()
        # print "currently simulating time step: " + str(t)
        subDF = policy[policy['time'] == t].reset_index(drop=True)
        # next generate 10000 new cues
        # generate 10000 optimal decisions

        # probably need an identity tracker for the new policies
        phenotypeTracker, __ = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)
        # update identity tracker for new cues
        # update cue tracker
        if t < T:
            cues = [int(cue) for cue in allCues[:, t]]
            cues = np.array(cues)

            cueTracker[:, 0] += (1 - cues)
            cueTracker[:, 1] += cues

    originalTwin = np.copy(phenotypeTracker)
    doppelgaenger = np.copy(phenotypeTracker)

    restPeriod = np.arange(tAdopt, tAdopt + lag, 1)

    # setting up the matrix for the yoked opposite cues
    cueTrackerDoppel = np.copy(cueTracker)

    cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
    cueTrackerDoppel[:, 1] += -cues + oppositeCues

    for t2 in restPeriod:

        np.random.seed()
        subDF = policy[policy['time'] == t2].reset_index(drop=True)

        doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)
        # update the phenotypes of the twins
        originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker)

        if t2 < T:
            # probably need an identity tracker for the new policies
            cuesOriginal = [int(cue) for cue in allCues[:, t2]]
            cuesOriginal = np.array(cuesOriginal)

            if adoptionType == "yoked":
                oppositeCues = 1 - cuesOriginal
            elif adoptionType == "oppositePatch":
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                oppositeCues = np.array(oppositeCues)
            else:  # adoptionType = deprivation
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                oppositeCues = np.array(oppositeCues)

            # update cue tracker
            cueTracker[:, 0] += (1 - cuesOriginal)
            cueTracker[:, 1] += cuesOriginal

            cueTrackerDoppel[:, 0] += (1 - oppositeCues)
            cueTrackerDoppel[:, 1] += oppositeCues

    restPeriodReunited = np.arange(tAdopt + lag, T + 1, 1)

    # need to reverse the last update
    cueTrackerDoppel[:, 0] += -(1 - oppositeCues) + (1 - cuesOriginal)
    cueTrackerDoppel[:, 1] += -(oppositeCues) + cuesOriginal

    if not endOfExposure:  # this means we want to measure phenotypic distance at the end of onotgeny
        for t3 in restPeriodReunited:
            # they will receive the same cues again

            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t3].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker)
            # update identity tracker for new cues
            doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)

            if t3 < T:
                # probably need an identity tracker for the new policies
                cuesOriginal = [int(cue) for cue in allCues[:, t3]]
                cuesOriginal = np.array(cuesOriginal)

                # update cue tracker
                cueTracker[:, 0] += (1 - cuesOriginal)
                cueTracker[:, 1] += cuesOriginal

                cueTrackerDoppel[:, 0] += (1 - cuesOriginal)
                cueTrackerDoppel[:, 1] += cuesOriginal

    return originalTwin, doppelgaenger, cueProbabilities


# todo move fwd inot a separate script which will be called from this script and the main script
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


def generateCueSequencesAndProabilities(states, T, startProb, transProb, emProb):
    x0Values = np.arange(0, T + 1, 1)
    x1Values = np.arange(0, T + 1, 1)
    cueSequences = []
    cueProbabilities = []
    for x0 in x0Values:
        for x1 in x1Values:
            if x0 + x1 == T:
                current = multiset_permutations(['0'] * x0 + ['1'] * x1)
                result = [fwd(observation, states, startProb, transProb, emProb) for
                          observation in current]

                cueProbabilities += [sum(curr) for curr in result]
                cueSequences += list(multiset_permutations(['0'] * x0 + ['1'] * x1))

    return cueSequences, cueProbabilities


def simulateTwins(tAdopt, twinNum, env, adopt, T, adoptionType, transitionProb, emissionProb):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability array!
    :return: phenotypic distance between pairs of twins
    """
    if env == 1:
        startEnv = "E1"  # take the very first cue reliability
        start_probability = {'E0': 0,
                             'E1': 1}
    else:
        startEnv = "E0"
        start_probability = {'E0': 1,
                             'E1': 0}

    states = ('E0', 'E1')

    """
    compute an exact behavior for T <= 9 
    """
    if isinstance(twinNum, list):
        allCues = np.stack(twinNum)
        twinNum = len(twinNum)

    else:
        HMM = HiddenMarkovChain(transitionProb, emissionProb)
        allCues = np.stack([HMM.generate_states(startEnv, T) for x in np.arange(twinNum)])  # shape agentsxtime

    cueProbabilities = []
    result = [fwd(observation, states, start_probability, transitionProb, emissionProb) for
              observation in allCues]
    cueProbabilities += [sum(curr) for curr in result]

    if adopt:
        tValues = np.arange(1, tAdopt, 1)

        cues = [int(cue) for cue in allCues[:, 0]]
        cues = np.array(cues)

        # need to reverse the last update
        if adoptionType == "yoked":
            oppositeCues = 1 - cues

        elif adoptionType == "oppositePatch":
            """
            opposite patch cues in this model would be cues from an HMM with switches transition probabilities;
            effectively it will be the same for a model with symmetric transition probabilities;
            easy to do if necessary
            """
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)

        elif adoptionType == "deprivation":
            """
            This still works for a model with changing environments 
            """
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)
        else:
            print "wrong input argument to adoption type!"
            exit(1)

        cueTracker = np.zeros((twinNum, 2))
        cueTracker[:, 0] = 1 - cues
        cueTracker[:, 1] = cues
        phenotypeTracker = np.zeros((twinNum, 3))
        posBeliefTracker = [0] * twinNum

        simValues = np.arange(0, twinNum, 1)
        for t in tValues:
            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions
            phenotypeTracker, posBeliefTracker = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)

            # probably need an identity tracker for the new policies
            if t < T:
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)
                # update cue tracker
                cueTracker[:, 0] += (1 - cues)
                cueTracker[:, 1] += cues

        # post adoption period
        # continue here
        originalTwin = np.copy(phenotypeTracker)
        doppelgaenger = np.copy(phenotypeTracker)

        posBeliefTrackerOrg = np.zeros((twinNum, T + 1 - tAdopt + 1))
        posBeliefTrackerDG = np.zeros((twinNum, T + 1 - tAdopt + 1))

        # for the first time point where twins are separated the whole time we only add a placeholder for the prior
        # an array of zeros; therefore the postprocessinf needs to be doner atfer the arguments have been returned
        posBeliefTrackerOrg[:, 0] = posBeliefTracker
        posBeliefTrackerDG[:, 0] = posBeliefTracker
        del posBeliefTracker

        restPeriod = np.arange(tAdopt, T + 1, 1)

        # setting up the matrix for the yoked opposite cues
        cueTrackerDoppel = np.copy(cueTracker)

        cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
        cueTrackerDoppel[:, 1] += -cues + oppositeCues

        for t2 in restPeriod:  # this is where adoption starts

            np.random.seed()
            subDF = policy[policy['time'] == t2].reset_index(drop=True)
            # update the phenotypes of the twins
            originalTwin, posBeliefOrg = updatePhenotype(subDF, originalTwin, simValues, cueTracker)

            posBeliefTrackerOrg[:, t2 - tAdopt + 1] = posBeliefOrg

            doppelgaenger, posBeliefDG = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)

            posBeliefTrackerDG[:, t2 - tAdopt + 1] = posBeliefDG

            if t2 < T:
                # update cue tracker
                # probably need an identity tracker for the new policies
                cuesOriginal = [int(cue) for cue in allCues[:, t2]]
                cuesOriginal = np.array(cuesOriginal)

                if adoptionType == "yoked":
                    oppositeCues = 1 - cuesOriginal
                elif adoptionType == "oppositePatch":
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])  # TODO see above
                    oppositeCues = np.array(oppositeCues)
                else:  # adoptionType = deprivation
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                    oppositeCues = np.array(oppositeCues)

                cueTracker[:, 0] += (1 - cuesOriginal)
                cueTracker[:, 1] += cuesOriginal

                cueTrackerDoppel[:, 0] += (1 - oppositeCues)
                cueTrackerDoppel[:, 1] += oppositeCues

            # store the very first phenotype following adotption to limit the amount of data you need to store
            if t2 == tAdopt:
                originalTwinTemp = np.copy(originalTwin)
                doppelgaengerTemp = np.copy(doppelgaenger)

        return originalTwin, doppelgaenger, posBeliefTrackerOrg, posBeliefTrackerDG, originalTwinTemp, doppelgaengerTemp, cueProbabilities


    else:  # to just calculate mature phenotypes and rank order stability
        tValues = np.arange(1, T + 1, 1)
        cuesSTart = [int(cue) for cue in allCues[:, 0]]
        cuesSTart = np.array(cuesSTart)
        cueTracker = np.zeros((twinNum, 2))
        cueTracker[:, 0] = 1 - cuesSTart
        cueTracker[:, 1] = cuesSTart
        phenotypeTracker = np.zeros((twinNum, 3))
        phenotypeTrackerTemporal = np.zeros((twinNum, 3, T))
        posBeliefTrackerTemporal = np.zeros((twinNum, T))

        simValues = np.arange(0, twinNum, 1)
        for t in tValues:
            np.random.seed()
            subDF = policy[policy['time'] == t].reset_index(drop=True)

            # print identTracker
            phenotypeTracker, posBelief = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)
            phenotypeTrackerTemporal[:, :, t - 1] = np.copy(phenotypeTracker)
            posBeliefTrackerTemporal[:, t - 1] = np.copy(posBelief)

            if t < T:
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)

                # update cue tracker
                cueTracker[:, 0] += (1 - cues)
                cueTracker[:, 1] += cues

        # successfully computed mature phenotypes
        return phenotypeTracker, phenotypeTrackerTemporal, posBeliefTrackerTemporal, cueProbabilities


def runExperimentalTwinStudiesParallel(tAdopt, twinNum, env, pE0E0, pE1E1, pC1E1, lag, T, resultsPath, argumentR,
                                       argumentP,
                                       adoptionType, endOfExposure):
    policyPath = os.path.join(resultsPath,
                              'runTest_%s%s_%s%s_%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE0E0, pE1E1, pC1E1))
    setGlobalPolicy(policyPath)

    # load the cue reliability array
    emissionProb = pickle.load(open(os.path.join(resultsPath, 'runTest_%s%s_%s%s_%s/emission_probability.p' % (
        argumentR[0], argumentP[0], pE0E0, pE1E1, pC1E1)), "rb"))

    pE0E1 = 1 - pE0E0
    pE1E0 = 1 - pE1E1

    transitionProb = {
        'E0': {'E0': pE0E0, 'E1': pE0E1},
        'E1': {'E0': pE1E0, 'E1': pE1E1},
    }

    states = ('E0', 'E1')

    if env == 1:
        start_probability = {'E0': 0,
                             'E1': 1}
    else:
        start_probability = {'E0': 1,
                             'E1': 0}

    allCues, probabilities = generateCueSequencesAndProabilities(states, T, start_probability, transitionProb,
                                                                 emissionProb)
    if len(probabilities) <= twinNum:
        simulationChunk = chunks(allCues,
                                 12)  # thi provides sublists of length 12 each, not exactly what I wanted bu perhaps it works
    else:
        simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    pool = Pool(processes=12)

    results = pool.map(func_star2, itertools.izip(itertools.repeat(tAdopt),
                                                  simulationChunk, itertools.repeat(env),
                                                  itertools.repeat(lag),
                                                  itertools.repeat(T), itertools.repeat(adoptionType),
                                                  itertools.repeat(endOfExposure), itertools.repeat(transitionProb),
                                                  itertools.repeat(emissionProb)))
    pool.close()
    pool.join()

    results1a, results2a, results3a = zip(*results)
    if len(probabilities) > twinNum:
        return np.concatenate(results1a), np.concatenate(results2a), None
    else:
        return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), n):
        yield lst[i:i + n]


def runTwinStudiesParallel(tAdopt, twinNum, env, pE0E0, pE1E1, pC1E1, adopt, T, resultsPath, argumentR, argumentP,
                           adoptionType,
                           allENV):
    policyPath = os.path.join(resultsPath,
                              'runTest_%s%s_%s%s_%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE0E0, pE1E1, pC1E1))
    setGlobalPolicy(policyPath)

    emissionProb = pickle.load(open(os.path.join(resultsPath, 'runTest_%s%s_%s%s_%s/emission_probability.p' % (
        argumentR[0], argumentP[0], pE0E0, pE1E1, pC1E1)), "rb"))

    pE0E1 = 1 - pE0E0
    pE1E0 = 1 - pE1E1

    transitionProb = {
        'E0': {'E0': pE0E0, 'E1': pE0E1},
        'E1': {'E0': pE1E0, 'E1': pE1E1},
    }
    states = ('E0', 'E1')

    if env == 1:
        startEnv = "E1"  # take the very first cue reliability
        start_probability = {'E0': 0,
                             'E1': 1}
    else:
        startEnv = "E0"
        start_probability = {'E0': 1,
                             'E1': 0}

    allCues, probabilities = generateCueSequencesAndProabilities(states, T, start_probability, transitionProb,
                                                                 emissionProb)
    if len(probabilities) <= twinNum:
        simulationChunk = chunks(allCues,
                                 12)  # this provides sublists of length 12 each, not exactly what I wanted bu perhaps it works

    else:
        simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    # load the cue reliability array
    if not allENV:

        if adopt:
            pool = Pool(processes=12)

            results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                         simulationChunk, itertools.repeat(env),
                                                         itertools.repeat(adopt),
                                                         itertools.repeat(T), itertools.repeat(adoptionType),
                                                         itertools.repeat(transitionProb),
                                                         itertools.repeat(emissionProb)))
            pool.close()
            pool.join()
            # results1, results2 refer to the phenotypes of orginals and clones
            # results3, results4 refer to the belief matrices of original and clone; shape: numAgents x separationTime +1
            results1a, results2a, results3a, results4a, results5a, results6a, result7a = zip(*results)

            if len(probabilities) > twinNum:
                return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                    results4a), np.concatenate(results5a), np.concatenate(results6a), None
            else:
                return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                    results4a), np.concatenate(results5a), np.concatenate(results6a), np.concatenate(result7a)

        else:

            pool = Pool(processes=12)

            results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                         simulationChunk, itertools.repeat(env),
                                                         itertools.repeat(adopt),
                                                         itertools.repeat(T), itertools.repeat(adoptionType),
                                                         itertools.repeat(transitionProb),
                                                         itertools.repeat(emissionProb)))
            pool.close()
            pool.join()

            results1a, results2a, results3a, results4a = zip(*results)

            if len(probabilities) > twinNum:
                return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), None
            else:
                return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                    results4a)
            # results 2: the first dimension refers to agents, the second to
            # phenotypes and the third to time

    else:
        resultsAllENV = {}
        for cueRel in allENV:
            print cueRel

            pool = Pool(processes=12)

            results = pool.map(func_star, itertools.izip(itertools.repeat(tAdopt),
                                                         simulationChunk, itertools.repeat(env),
                                                         itertools.repeat(adopt),
                                                         itertools.repeat(T), itertools.repeat(adoptionType),
                                                         itertools.repeat(transitionProb),
                                                         itertools.repeat(emissionProb)))
            pool.close()
            pool.join()

            results1a, results2a, results3a, results4a = zip(*results)
            # What we return here is only the mature phenotypes; for a reaction norm, we probably would want
            # the overall level of plasticity for each developmental system in each different environment;
            # look at old versions of this and leave it in here for now
            allResults = np.concatenate(results1a), np.concatenate(results2a)

            resultsAllENV[cueRel] = allResults[0]

        return resultsAllENV


def calcEuclideanDistance(original, doppelgaenger):
    result = [np.sqrt(np.sum((x - y) ** 2)) for x, y in
              zip(original[:, 0:2], doppelgaenger[:, 0:2])]  # TODO: if interested in euclidean distance incoporate here
    return np.array(result)


def runExperimentalAdoptionExperiment(T, numAgents, env, pE0E0, pE1E1, cueReliability, resultsPath, argumentR,
                                      argumentP, lag,
                                      adoptionType, endOfExposure):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # proportiional distance: absolute distance divided by maximum possible distance
    # maximum possible distance: 20 * sqrt(2)
    tValues = np.arange(1, T + 1, 1)

    resultLen = determineLength(T)
    if resultLen > numAgents:
        resultLen = int(math.ceil(float(numAgents) / 12)) * 12  # Does this still work?

    results = np.zeros((T, resultLen))

    for t in tValues:
        print "currently working on time step: " + str(t)
        original, doppelgaenger, cueProbabilities = runExperimentalTwinStudiesParallel(t, numAgents, env, pE0E0, pE1E1,
                                                                                       cueReliability,
                                                                                       lag, T,
                                                                                       resultsPath, argumentR,
                                                                                       argumentP, adoptionType,
                                                                                       endOfExposure)
        results[t - 1, :] = calcEuclideanDistance(original, doppelgaenger)

    return results, cueProbabilities


def postProcessPosteriorBelief(posBeliefOrg, posBeliefDG, cueProbabilities):
    # first calculate difference across separation time; that is the average of the difference matrix
    absDifferencesOrg = np.absolute(np.diff(posBeliefOrg))
    meanDifferenceOrg = np.average(absDifferencesOrg, axis=0, weights=cueProbabilities)
    absDifferencesDG = np.absolute(np.diff(posBeliefDG))
    meanDifferenceDG = np.average(absDifferencesDG, axis=0, weights=cueProbabilities)
    posDifferences = np.abs(posBeliefOrg[:, 1:] - posBeliefDG[:, 1:])
    meanPosDifferences = np.average(posDifferences, axis=0, weights=cueProbabilities)
    return meanDifferenceOrg, meanDifferenceDG, meanPosDifferences


def determineLength(T):
    a = np.arange(0, T + 1, 1)
    b = np.arange(0, T + 1, 1)
    lengthCtr = 0
    for a0 in a:
        for b0 in b:
            if a0 + b0 == T:
                current = multiset_permutations(['0'] * a0 + ['1'] * b0)
                lengthCtr += len(list(current))

    return lengthCtr


def runAdoptionExperiment(T, numAgents, env, pE0E0, pE1E1, cueReliability, resultsPath, argumentR, argumentP,
                          adoptionType):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # proportional distance: absolute distance divided by maximum possible distance
    # maximum possible distance: 20 * sqrt(2)
    tValues = np.arange(1, T + 1, 1)

    resultsBeliefAggr = np.zeros((T, 3))

    posBeliefDiffStart = [0] * T
    posBeliefDiffEnd = [0] * T

    resultLen = determineLength(T)
    if resultLen > numAgents:
        resultLen = int(math.ceil(float(numAgents) / 12)) * 12  # Does this still work?

    results = np.zeros((T, resultLen))
    resultsTempPhenotypes = np.zeros(
        (T, resultLen))  # euclidean distance between original and twin right after exposure

    for t in tValues:
        print "currently working on time step: " + str(t)
        original, doppelgaenger, posBeliefOrg, posBeliefDG, originalTemp, doppelgaengerTemp, cueProbabilities = runTwinStudiesParallel(
            t,
            numAgents,
            env,
            pE0E0,
            pE1E1,
            cueReliability,
            True,
            T,
            resultsPath,
            argumentR,
            argumentP,
            adoptionType,
            [])

        if t == 1:
            prior = calcStationaryDist((pE0E0, pE1E1))
            simNum = posBeliefOrg.shape[0]
            posBeliefOrg[:, 0] = [1 - prior] * simNum
            posBeliefDG[:, 0] = [1 - prior] * simNum

        results[t - 1, :] = calcEuclideanDistance(original, doppelgaenger)
        resultsTempPhenotypes[t - 1, :] = calcEuclideanDistance(originalTemp, doppelgaengerTemp)
        meanDifferenceOrg, meanDifferenceDG, meanPosDifferences = postProcessPosteriorBelief(posBeliefOrg, posBeliefDG,
                                                                                             cueProbabilities)

        if t == 1:
            posBeliefDeltaOrg = meanDifferenceOrg
            posBeliefDeltaOrg = posBeliefDeltaOrg.reshape(T, 1)
            posBeliefDeltaDG = meanDifferenceDG
            posBeliefDeltaDG = posBeliefDeltaDG.reshape(T, 1)
        posBeliefDiffEnd[t - 1] = meanPosDifferences[-1]  # store the last difference
        posBeliefDiffStart[t - 1] = meanPosDifferences[0]  # store the first difference

        # it might still be interesting to have a plot with one line per ontogeny indicating
        # belief change of the orginal in one plot, the doppelgaenger, and the posterior belief change

        # is the absolute average difference across time and agents in posterior belief interesting?
        # I think it might be: it is a different proxy for plasticity in belief
        # how different is twins' belief in environment 1 due to exposure to cues?; focus on this for now, but keep
        # thinking about this
        resultsBeliefAggr[t - 1, :] = [np.mean(meanDifferenceOrg), np.mean(meanDifferenceDG),
                                       np.mean(meanPosDifferences)]

    # need to add the other two columns
    posBeliefDiffEnd = np.array(posBeliefDiffEnd).reshape(T, 1)
    posBeliefDiffStart = np.array(posBeliefDiffStart).reshape(T, 1)

    resultsBeliefAggr = np.hstack(
        (resultsBeliefAggr, posBeliefDeltaOrg, posBeliefDeltaDG, posBeliefDiffEnd, posBeliefDiffStart))

    return results, resultsBeliefAggr, resultsTempPhenotypes, cueProbabilities


def normRows(vec):
    if vec.min() != vec.max():
        curRowNorm = (vec - vec.min()) / float((vec.max() - vec.min()))
        return curRowNorm
    else:
        return vec


def meanAbsDistance(data, currMean, cueProbabilities):
    dataDiff = data - currMean
    return np.average(abs(dataDiff), weights=cueProbabilities)


def postProcessResultsMat(results, T, endOfExposure, lag, cueProbabilities):
    resultsVec = []
    resultsVecVar = []
    resultsVecRelative = []
    resultsVecRelativeVar = []

    if not endOfExposure:  # if phenotypic distance has been measured at the end of ontogeny
        resultsNorm = results / float(T * np.sqrt(2))
        for idx in range(results.shape[0]):
            curRowNorm = resultsNorm[idx, :]
            curRow = results[idx, :]
            curRowRelative = curRow / float((T - idx) * np.sqrt(2))

            resultsVec.append(np.average(curRowNorm, weights=cueProbabilities))
            resultsVecRelative.append(np.average(curRowRelative, weights=cueProbabilities))
            varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1], cueProbabilities)
            varAbs = meanAbsDistance(curRowNorm, resultsVec[-1], cueProbabilities)
            resultsVecVar.append(varAbs)
            resultsVecRelativeVar.append(varRel)
    else:
        for idx in range(results.shape[0]):
            curRow = results[idx, :]
            curRowNorm = curRow / float((lag + idx) * np.sqrt(2))
            curRowRelative = curRow / float(lag * np.sqrt(2))
            resultsVec.append(np.average(curRowNorm, weights=cueProbabilities))
            resultsVecRelative.append(np.average(curRowRelative, weights=cueProbabilities))
            varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1], cueProbabilities)
            varAbs = meanAbsDistance(curRowNorm, resultsVec[-1], cueProbabilities)
            resultsVecVar.append(varAbs)
            resultsVecRelativeVar.append(varRel)
    return resultsVec, resultsVecRelative, resultsVecVar, resultsVecRelativeVar


def rescaleNumbers(newMin, newMax, numbersArray):
    OldMin = np.min(numbersArray)
    OldMax = np.max(numbersArray)
    result = [(((OldValue - OldMin) * (newMax - newMin)) / float(OldMax - OldMin)) + newMin for OldValue in
              numbersArray]
    return result


def area_calc(probs, r):
    # result = [(p)**2 * np.pi*r for p in probs]
    result = [np.sqrt(float(p)) * r for p in probs]
    return result


def plotTriangularPlots(tValues, markovProbabilities, cueValidityArr, maturePhenotypes, T, twinResultsPath, env):
    # first step is to permute indices
    permuteIdx = [0, 2, 1]

    fig, axes = plt.subplots(len(cueValidityArr), len(markovProbabilities), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            ax.set(aspect='equal')  # if you remove this, plots won't be square
            plt.sca(ax)

            """
            Here goes the actual plotting code 
            """
            maturePhenotypesCurr, cueProbabilities = maturePhenotypes[(markov_chain, cueVal)]
            numAgents = maturePhenotypesCurr.shape[0]
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
            # now need to work on the scaling of points

            unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
            # area = area_calc(uniqueCounts / float(numAgents), 150)
            if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                uniqueFrac = []
                for matPhen in unique:
                    probIdx = np.where(
                        (maturePhenotypesCurr[:, 0] == matPhen[0]) & (maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                    uniqueFrac.append(sum(cueProbabilities[probIdx]))

                area2 = np.array(uniqueFrac) * float(250)


            else:
                area2 = (uniqueCounts / float(numAgents)) * 250
            # this one would be scalling according to area
            tax.scatter(unique[:, permuteIdx], s=area2, color='black')
            tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            if jx == len(markovProbabilities) / 2 and ix == len(cueValidityArr) / 2:
                fontsize = 20
                tax.right_corner_label("P0", fontsize=fontsize, offset=-0.15)
                tax.top_corner_label("wait time", fontsize=fontsize)
                tax.left_corner_label("P1", fontsize=fontsize, offset=-0.15)
                tax._redraw_labels()
            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')

    plt.savefig(os.path.join(twinResultsPath, 'ternary_%s.png' % env), dpi=1000)
    plt.close()


def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    y0, y1, yw = state
    b0_D, b1_D = float(b0_D), float(b1_D)

    if argumentR == 'linear':
        phiVar = b0_D * y0 + b1_D * y1

    elif argumentR == 'diminishing':
        alphaRD = (T) / float(1 - float(np.exp(-beta * (T))))
        phiVar = b0_D * alphaRD * (1 - np.exp(-beta * y0)) + b1_D * alphaRD * (1 - np.exp(-beta * y1))

    elif argumentR == 'increasing':
        alphaRI = (T) / float(float(np.exp(beta * (T))) - 1)
        phiVar = b0_D * alphaRI * (np.exp(beta * y0) - 1) + b1_D * alphaRI * (np.exp(beta * y1) - 1)
    else:
        print 'Wrong input argument to additive fitness reward function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    if argumentP == 'linear':
        psiVar = -(b0_D * y1 + b1_D * y0)

    elif argumentP == 'diminishing':
        alphaPD = (T) / float(1 - float(np.exp(-beta * (T))))
        psiVar = -(b0_D * alphaPD * (1 - np.exp(-beta * y1)) + b1_D * alphaPD * (1 - np.exp(-beta * y0)))

    elif argumentP == 'increasing':
        alphaPI = (T) / float(float(np.exp(beta * (T))) - 1)
        psiVar = -(b0_D * alphaPI * (np.exp(beta * y1) - 1) + b1_D * alphaPI * (np.exp(beta * y0) - 1))
    else:
        print 'Wrong input argument to additive fitness penalty function'
        print 'Argument must be linear, increasing or diminishing'
        exit(1)

    tf = 0 + phiVar + psi_weighting * psiVar

    return tf


def calcFitness(state, argumentR, argumentP, adultT, markovChain, pE1, T, beta, psi_weighting):
    tfList = []  # this will hold all fitness values across the adult lifespan

    pE0E0, pE1E1 = markovChain
    P = np.array([[pE0E0, 1 - pE0E0], [1 - pE1E1, pE1E1]])
    b0_D = 1 - pE1
    b1_D = pE1
    for t in np.arange(1, adultT + 1, 1):
        currTf = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)
        tfList.append(currTf)
        b0_D, b1_D = np.dot([(1-pE1), pE1], np.linalg.matrix_power(P, t))

    tfList = np.array(tfList)
    return np.sum(tfList)  # TODO possibly incoporporate the fitness weights gain here


def fitnessDifference(markovProbabilities, cueValidityArr, policyPath, T, resultsPath, baselineFitness,
                      argumentR,
                      argumentP, adultT, numAgents):
    # fitness functions
    # keep in mind fitnessMax is equivalent to T

    beta = 0.2
    # dictionary for storing the results
    resultsDict = {}
    for markov_chain in markovProbabilities:
        pE0E0, pE1E1 = markov_chain
        for cueReliability in cueValidityArr:

            print "Currently calculating expected fitness differences with pE0E0: " + str(
                pE0E0) + " and cue reliability: " + str(cueReliability)

            # fitness following the optimal policy
            # simulate mature phenotypes for each environment
            maturePhenotypesEnv0, _, _, cueProbabilitiesEnv0 = \
                runTwinStudiesParallel(0, numAgents, 0, pE0E0, pE1E1, cueReliability, False, T, policyPath,
                                       argumentR, argumentP, None, [])

            maturePhenotypesEnv1, _, _, cueProbabilitiesEnv1 = \
                runTwinStudiesParallel(0, numAgents, 1, pE0E0, pE1E1, cueReliability, False, T, policyPath,
                                       argumentR, argumentP, None, [])

            prior = 1 - calcStationaryDist(markov_chain)

            OEnv1 = np.average(
                np.array([calcFitness(y, argumentR, argumentP, adultT, markov_chain, 1, T, beta, 1) for y in
                          maturePhenotypesEnv1]), weights=cueProbabilitiesEnv1)
            OEnv0 = np.average(
                np.array([calcFitness(y, argumentR, argumentP, adultT, markov_chain, 0, T, beta, 1) for y in
                          maturePhenotypesEnv0]), weights=cueProbabilitiesEnv0)

            OFitness = ((prior * OEnv1 + (1 - prior) * OEnv0) - baselineFitness) / float(T*adultT)

            # next specialist Fitness
            if prior < 0.5:
                phenotypeS = np.array([T, 0, 0])
                SEnv1 = calcFitness(phenotypeS, argumentR, argumentP, adultT, markov_chain, 1, T, beta, 1)
                SEnv0 = calcFitness(phenotypeS, argumentR, argumentP, adultT, markov_chain, 0, T, beta, 1)


            else:
                if isinstance(cueProbabilitiesEnv1, list) or isinstance(cueProbabilitiesEnv1, np.ndarray):
                    resultLen = len(cueProbabilitiesEnv1)
                else:
                    resultLen = numAgents

                specialistPhenotypes = np.zeros((resultLen, 3))
                specialistPhenotypes[:, 0] = np.append(np.array([T] * int(resultLen / 2)),
                                                       np.array([0] * (resultLen - int(resultLen / 2))))

                specialistPhenotypes[:, 1] = np.append(np.array([0] * int(resultLen / 2)),
                                                       np.array([T] * (resultLen- int(resultLen / 2))))

                SEnv1 = np.mean(np.array(
                    [calcFitness(y, argumentR, argumentP, adultT, markov_chain, 1, T, beta, 1) for y in
                     specialistPhenotypes]))
                SEnv0 = np.mean(np.array(
                    [calcFitness(y, argumentR, argumentP, adultT, markov_chain, 0, T, beta, 1) for y in
                     specialistPhenotypes]))

            SFitness = ((prior * SEnv1 + (1 - prior) * SEnv0) - baselineFitness) / float(T*adultT)
            phenotypeG = np.array([T / float(2), T / float(2), 0])

            GEnv1 = calcFitness(phenotypeG, argumentR, argumentP, adultT, markov_chain, 1, T, beta, 1)
            GEnv0 = calcFitness(phenotypeG, argumentR, argumentP, adultT, markov_chain, 0, T, beta, 1)
            GFitness = ((prior * GEnv1 + (1 - prior) * GEnv0) - baselineFitness) / float(T*adultT)

            resultsDict[((pE0E0, pE1E1), cueReliability)] = np.array([SFitness, OFitness, GFitness])

    pickle.dump(resultsDict, open(os.path.join(resultsPath, "fitnessDifferences.p"), "wb"))


def plotFitnessDifference(markovProbabilities, cueValidityArr, twinResultsPath):
    # first open the dictionary containing the results

    differencesDict = pickle.load(open(os.path.join(twinResultsPath, "fitnessDifferences.p"), "rb"))
    # define the xAxis
    x = np.arange(3)
    xLabels = ["S", "O", "G"]
    fig, axes = plt.subplots(len(cueValidityArr), len(markovProbabilities), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            ax.set(aspect="equal")
            plt.sca(ax)
            # open the relevant fitness difference array
            fitnessDifferences = differencesDict[(markov_chain, cueVal)]

            barList = plt.bar(x, fitnessDifferences)

            barList[0].set_color("lightgray")
            barList[1].set_color("grey")
            barList[2].set_color("black")

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=20, labelpad=10)
                plt.xticks(x, xLabels, fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('Fitness difference', fontsize=20, labelpad=10)
                plt.yticks([-1, 0, 1], fontsize=15)

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1

        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'fitnessDifferences.png'), dpi=1000)
    plt.close()


def plotFitnessDifferenceOverview(cueValidityArr, T, adultTArr, autoCorrDict,
                                  twinResultsAggregatedPath, dataPath, argumentR, argumentP,levelsAutoCorrToPlot):
    # first open the dictionary containing the results
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
    # prepare the x-axis values
    fig, axes = plt.subplots(len(cueValidityArr), len(adultTArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for adultT in adultTArr:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(adultTArr) + jx]
            ax.set(aspect=4)
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = "fitnessDifferences.p"
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                fitnessDifferences = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                fitnessDifferences.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: fitnessDifferences[(key, cueVal)] for key, val in autoCorrDict_sorted}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]
                fitnessDifferences = []
                for idx, autoCorr in enumerate(autoCorrValSubset):
                    fitnessDifferences += list(cueValDict[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % autoCorrValSubset[0], "G|", "|S", "O\n\n%s" % autoCorrValSubset[1], "G|",
                           "|S",
                           "O\n\n%s" % autoCorrValSubset[2], "G|"]




                barList = plt.bar(x, fitnessDifferences)

                lightGreys = np.arange(0,len(x)-1,3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")

            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    fitnessDifferenceMean = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                    cueValDictSubset[levl] = fitnessDifferenceMean

                fitnessDifferences = []
                for idx, autoCorr in enumerate(loopArrayLevl):
                    fitnessDifferences += list(cueValDictSubset[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % loopArrayLevl[0][0].upper(), "G|", "|S", "O\n\n%s" % loopArrayLevl[1][0].upper(), "G|",
                           "|S",
                           "O\n\n%s" % loopArrayLevl[2][0].upper(), "G|"]

                barList = plt.bar(x, fitnessDifferences)

                lightGreys = np.arange(0, len(x) - 1, 3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")
            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """

            plt.plot(x, [1] * len(x), linestyle='dashed', linewidth=1, color='grey')
            plt.plot(x, [-1] * len(x), linestyle='dashed', linewidth=1, color='grey')

            #yvals = np.arange(-0.9,1,0.1)
            #plt.plot([2.5]*len(yvals),yvals,linestyle='dashed', linewidth=1, color='grey')
            #plt.plot([5.5] * len(yvals), yvals, linestyle='dashed', linewidth=1, color='grey')


            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)

            if ix == 0:
                plt.title("%s" %adultT, fontsize = 20, pad = 15)

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=20, labelpad=10)
                plt.tick_params(pad = 10)
                plt.xticks(x, xLabels, fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.yticks([-1, 0, 1], fontsize=15)
                if ix == len(cueValidityArr)-1:
                    plt.ylabel('fitness difference', fontsize=20, labelpad=15)
            else:
                ax.tick_params(axis='y', which='both',length=0)

            if jx == len(adultTArr) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")


            ix += 1
        jx += 1

        plt.suptitle('adult life span', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')

        fig.text(0.17, 0.05, 'autocorrelation', fontsize=20, horizontalalignment='left', verticalalignment='bottom',
                 transform=ax.transAxes, rotation='horizontal')
    plt.savefig(os.path.join(twinResultsAggregatedPath, 'fitnessDifferencesOverview%s.png'%levelsAutoCorrToPlot), dpi=1000)
    plt.close()



def plotFitnessDifferenceOverviewMerge(cueValidityArr, T, adultTArr, autoCorrDict,
                                  twinResultsAggregatedPath, mainDataPath, argumentR, argumentP,levelsAutoCorrToPlot, nameArg):

    if len(nameArg) == 3:
        nameArg = nameArg[0:2]

    rowVec = []
    for currX in nameArg:
        for currY in adultTArr:
            rowVec.append((currX, currY))

    # prepare the x-axis values
    fig, axes = plt.subplots(len(cueValidityArr), len(rowVec), sharex=True, sharey=True)
    fig.set_size_inches(32, 16)
    ax_list = fig.axes


    jx = 0
    for symmArg, adultT in rowVec:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect=4)
            plt.sca(ax)


            autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(symmArg))


            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = "fitnessDifferences.p"
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                fitnessDifferences = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                fitnessDifferences.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: fitnessDifferences[(key, cueVal)] for key, val in autoCorrDict_sorted}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]
                fitnessDifferences = []
                for idx, autoCorr in enumerate(autoCorrValSubset):
                    fitnessDifferences += list(cueValDict[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % round(autoCorrValSubset[0],1), "G|", "|S", "O\n\n%s" % round(autoCorrValSubset[1],1), "G|",
                           "|S",
                           "O\n\n%s" % round(autoCorrValSubset[2],1), "G|"]

                #splot for symmetyric and asymmetric
                barList = plt.bar(x, fitnessDifferences)


                lightGreys = np.arange(0,len(x)-1,3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")

            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    fitnessDifferenceMean = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                    cueValDictSubset[levl] = fitnessDifferenceMean

                fitnessDifferences = []
                for idx, autoCorr in enumerate(loopArrayLevl):
                    fitnessDifferences += list(cueValDictSubset[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % loopArrayLevl[0][0].upper(), "G|", "|S", "O\n\n%s" % loopArrayLevl[1][0].upper(), "G|",
                           "|S",
                           "O\n\n%s" % loopArrayLevl[2][0].upper(), "G|"]

                barList = plt.bar(x, fitnessDifferences)

                lightGreys = np.arange(0, len(x) - 1, 3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")
            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """
            xLines = np.arange(-0.5,8.6,0.1)

            plt.plot(xLines, [1] * len(xLines), linestyle='dashed', linewidth=1, color='grey')
            plt.plot(xLines, [-1] * len(xLines), linestyle='dashed', linewidth=1, color='grey')

            yvals = np.arange(-0.9,1,0.1)
            plt.plot([2.5]*len(yvals),yvals,linestyle='dashed', linewidth=1, color='grey')
            plt.plot([5.5] * len(yvals), yvals, linestyle='dashed', linewidth=1, color='grey')


            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)




            if (jx+1) %len(adultTArr) == 0 and not jx == (len(rowVec))-1:
                paramVLine = max(x) +1.5
                ax.vlines([paramVLine], -0.05, 1.05, transform=ax.get_xaxis_transform(),color='black', lw=2,clip_on=False)

            if ix == 0:
                plt.title("%s     " %str(adultT), fontsize = 25, pad = 15, loc = 'center')

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=25, labelpad=10)
                plt.tick_params(pad = 10)


                plt.xticks(x, xLabels, fontsize=20)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                plt.yticks([-1, 0, 1], fontsize=20)
                if ix == len(cueValidityArr)-1:
                    plt.ylabel('fitness difference', fontsize=25, labelpad=20)
            else:
                ax.tick_params(axis='y', which='both',length=0)

            if jx == len(rowVec) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=25)
                ax.yaxis.set_label_position("right")


            ix += 1
        jx += 1


    fig.text(0.14, 0.01, 'autocorrelation', fontsize=25, horizontalalignment='left', verticalalignment='bottom',
             transform=ax.transAxes, rotation='horizontal')

    top = 0.8
    fig.text(0.94, 0.45, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.875

    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.1, top=top)

    fig.text(0.514, 0.95, 'transition probabilities', fontsize=25, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(nameArg)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    nameArg2 = ["symmetric","asymmetric"]

    for figCoord, adultT in zip(figVals, nameArg2):
        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.064
            else:
                figCoordF = figCoord - 0.05
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.085
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.514
            else:
                figCoordF = figCoord - 0.055
        fig.text(figCoordF, autoCorrCoord, '%s\n\n\nadult life span' % adultT, fontsize=25, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    plt.savefig(os.path.join(twinResultsAggregatedPath, 'fitnessDifferencesOverviewMerge%s.png'%levelsAutoCorrToPlot),bbox_inches='tight', dpi=1000)
    plt.close()


def calcNegativeRankSwitches(rankDf, T, arg):
    tValues = np.arange(0, T, 1)
    results = np.zeros((T, T))
    # need possible number of ranks at each time step
    for t in tValues:
        rankDfDiff = rankDf.loc[:, t:].sub(rankDf.loc[:, t], axis=0)
        rankDfDiff2 = rankDfDiff.copy(deep=True)
        if arg == 'unstable':
            rankDfDiff[rankDfDiff2 == 0] = 0
            rankDfDiff[rankDfDiff2 != 0] = 1
        else:
            rankDfDiff[rankDfDiff2 != 0] = 0
            rankDfDiff[rankDfDiff2 == 0] = 1
        results[t, t:] = rankDfDiff.sum(axis=0) / float(rankDf.shape[0])

    return results


def plotRankOrderStability(markovProbabilities, cueValidityArr, twinResultsPath, T, types, env):
    for distFun in types:
        plotRankOrderStability2(markovProbabilities, cueValidityArr, twinResultsPath, T, distFun, env)


def createLABELS(T):
    labels = [" "] * T
    labels[0] = str(1)
    labels[T - 1] = str(T)
    labels[int(T / 2) - 1] = str(T / 2)
    return labels


def plotRankOrderStability2(markovProbabilities, cueValidityArr, twinResultsPath, T, distFun, env):
    """
    :param priorE0Arr:
    :param cueValidityArr:
    :param twinResultsPath:
    :param T:
    :param distFun:
    :return:
    """

    '''
    We cannot use a correlation coefficient to determine rank-order stability because there might be cases in which 
    there is no variability in ranks 
    '''

    # first open the dictionary containing the results
    # for prior, cue reliability combination it contains a matrix with the ranks across time steps
    ranks = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRanks_%s.p" % (env)), "rb"))

    # what do we want to plot?
    # could have a plot with the correlation coefficient between consecutive timesteps
    # or a whole correlation matrix, heatplot? start with this
    # want to represent the proportion of ties as well

    fig, axes = plt.subplots(len(cueValidityArr), len(markovProbabilities), sharex=True, sharey=True)
    plt.subplots_adjust(top=0.92, bottom=0.12)
    specialAx = fig.add_axes([.16, .040, .7, .01])
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for markov_chain in markovProbabilities:
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1

            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')

            simRange += list(sim.flatten())

    boundary1 = min(simRange)
    boundary2 = max(simRange)

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
            # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
                axis=0)] + 0.1  # returns columns that are all zeros

            # calculating the similarity matrix
            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
                cmap = 'YlGnBu'
                yLabel = 'Cosine similarity'
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
                cmap = 'Greys'  # 'YlGnBu'
                yLabel = 'time step'

            # only negative rank switches

            # create a mask for the upper triangle
            mask = np.tri(sim.shape[0], k=0)
            if jx == len(markovProbabilities) - 1 and ix == 0:
                cbar = True
                cbar_ax = specialAx
                cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
                            'label': "proportion of rank switches"}  # 'label':"Proportion of negative rank switches",
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)

                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=15)
                cbar.ax.xaxis.label.set_size(20)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)


            else:
                cbar = False
                cbar_ax = None
                cbar_kws = None
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
                ax.tick_params(labelsize=15)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax2.set_ylim(0, 1)
            ax2.get_xaxis().tick_bottom()
            ax2.get_yaxis().tick_right()

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                ax.set_xlabel('time step', fontsize=20, labelpad=15)
            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)
                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.tick_params(labelsize=15)
            else:
                ax2.set_yticks([])

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStability2%s_%s.png' % (distFun, env)), dpi=1000)
    plt.close()

    # second plot is for rank stability
    fig, axes = plt.subplots(len(cueValidityArr), len(markovProbabilities), sharex=True, sharey=True)
    specialAx = fig.add_axes([.16, .055, .7, .01])
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for markov_chain in markovProbabilities:
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1

            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, "stable")

            simRange += list(sim.flatten())

    boundary1 = min(simRange)
    boundary2 = max(simRange)

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
            # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
                axis=0)] + 0.1  # returns columns that are all zeros

            # calculating the similarity matrix
            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
                cmap = 'YlGnBu'
                yLabel = 'Cosine similarity'
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'stable')
                cmap = 'Greys'  # 'YlGnBu'
                yLabel = 'Time step'

            # only negative rank switches

            # create a mask for the upper triangle
            mask = np.tri(sim.shape[0], k=0)
            if jx == len(markovProbabilities) - 1 and ix == 0:
                cbar = True
                cbar_ax = specialAx
                cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
                            'label': "Proportion of stable ranks"}  # 'label':"Proportion of negative rank switches",
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)

                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.xaxis.label.set_size(20)
            else:
                cbar = False
                cbar_ax = None
                cbar_kws = None
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
                ax.tick_params(labelsize=14)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                ax.set_xlabel('Time step', fontsize=20, labelpad=10)
            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=10)
            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos1%s_%s.png' % (distFun, env)), dpi=400)
    plt.close()

    # 3rd plot
    fig, axes = plt.subplots(len(markovProbabilities), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for markov_chain in markovProbabilities:
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1

            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')

            simRange += list(sim.flatten())

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            rankMatrix = ranks[(markov_chain, cueVal)]

            rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
            # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
                axis=0)] + 0.1  # returns columns that are all zeros

            # calculating the similarity matrix
            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable')

            if jx == len(markovProbabilities) - 1 and ix == 0:
                ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)


            else:
                ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)

            ax.set_ylim(0, 1)
            plt.yticks([])
            plt.xticks([])

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)
            #
            # if jx == 0:
            #     plt.title(str(cueVal), fontsize=30)
            #
            # if ix == 0 and jx == 0:
            #     ax.set_xlabel('Time', fontsize=30, labelpad=10)
            #     ax.spines['left'].set_visible(True)
            #     ax.yaxis.set_label_position("left")
            #     ax.set_ylabel('Proportion of rank switches', fontsize=30, labelpad=10)

            if ix == len(cueValidityArr) - 1:
                ax.set_xlabel('Time step', fontsize=20, labelpad=10)
            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('Proportion of rank switches', fontsize=20, labelpad=10)

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos2%s_%s.png' % (distFun, env)), dpi=400)
    plt.close()


def plotBeliefDistances(tValues, markovProbabilities, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp, env):
    fig, axes = plt.subplots(len(markovProbabilities), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(markov_chain, cueVal)]

            posBeliefDiffNoAverage = beliefDict[(markov_chain, cueVal)][:,
                                     5]  # measured at the end of ontogeny after the last cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            plt.plot(tValues, relativeDistance, color='black', linestyle='solid', linewidth=2, markersize=8,
                     marker='o',
                     markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([])

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sPlasticityAndBeliefEndOntogeny.png' % (argument, adoptionType, lag, safeStr, env)),
        dpi=1000)
    plt.close()

    fig, axes = plt.subplots(len(markovProbabilities), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]

            plt.sca(ax)

            posBeliefDiffNoAverage = beliefDict[(markov_chain, cueVal)][:,
                                     6]  # measured after each cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            relativeDistanceTemp = relativeDistanceDictTemp[(markov_chain, cueVal)]
            plt.plot(tValues, relativeDistanceTemp, color='black', linestyle='solid', linewidth=2, markersize=8,
                     marker='o', markerfacecolor='black')

            print "The current pE0E0 is %s and the cue reliability is %s" % (markov_chain[0], cueVal)
            print "The correlation between information and phenotype divergence is: " + str(
                stats.pearsonr(relativeDistanceTemp, posBeliefDiffNoAverage)[0])

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([])

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sPlasticityAndBeliefAfterCue.png' % (argument, adoptionType, lag, safeStr, env)),
        dpi=1000)
    plt.close()


def plotDistances(tValues, markovProbabilities, cueValidityArr, absoluteDistanceDict, relativeDistanceDict,
                  twinResultsPath,
                  argument, adoptionType, lag, endOfExposure, VarArg, absoluteDistanceDictVar, relativeDistanceDictVar,
                  env):
    fig, axes = plt.subplots(len(cueValidityArr), len(markovProbabilities), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for markov_chain in markovProbabilities:
            ax = ax_list[ix * len(markovProbabilities) + jx]
            ax.set(aspect=10)
            plt.sca(ax)
            absoluteDistance = absoluteDistanceDict[(markov_chain, cueVal)]
            relativeDistance = relativeDistanceDict[(markov_chain, cueVal)]

            if VarArg:
                # absoluteDistanceVar = absoluteDistanceDictVar[(markov_chain, cueVal)]
                # plt.plot(tValues, absoluteDistance, color='grey', linestyle='solid', linewidth=2, markersize=8,
                #          marker='D',
                #          markerfacecolor='grey')
                # plt.errorbar(tValues, absoluteDistance, yerr=absoluteDistanceVar, fmt="none", ecolor='grey')

                relativeDistanceVar = relativeDistanceDictVar[(markov_chain, cueVal)]

                plt.plot(tValues, relativeDistance, color='black', linestyle='--', linewidth=2, markersize=8,
                         marker='o', markerfacecolor='black')
                plt.errorbar(tValues, relativeDistance, yerr=relativeDistanceVar, fmt="none", ecolor='black')

            else:
                plt.plot(tValues, absoluteDistance, color='grey', linestyle='solid', linewidth=2, markersize=8,
                         marker='D',
                         markerfacecolor='grey')
                plt.plot(tValues, relativeDistance, color='black', linestyle='solid', linewidth=2, markersize=8,
                         marker='o', markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title("%s, %s" % (markov_chain[0], markov_chain[1]), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([])

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('phenotypic distance', fontsize=20, labelpad=10)

            if jx == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            jx += 1
        ix += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sPlasticity%s.png' % (argument, adoptionType, lag, safeStr, env, VarArg)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotIncludingOntogeny3_33(cueValidityArr, T, adultTArr, env, autoCorrDictTotal,
                                            twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType,
                                            lag, endOfExposure,
                                            studyArg, normalize, priorArr, levelsAutoCorrToPlotArg, nameArg):
    # select the plasticity argument
    arg = "relative"
    # switch adult T and and the correlation in this one
    tValues = np.arange(1, T + 1, 1)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for prior in priorArr:
        if 'E' in prior:
            env = prior[-2]
        # here select the right autocorr dict and dataPath
        autoCorrDict_sorted = sorted(autoCorrDictTotal[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

        dataPath = os.path.join(mainDataPath, str(prior))
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"
                    exit()

                heatPlot1.update(
                    {(adultT, cueVal): distanceDict[(key, cueVal)] for key in
                     autoCorrKeys})

        if arg == 'relative':
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Rel = min(dataList)  # theoretical min: 0
            boundary2Rel = max(dataList)  # theoretical max: 1
        else:
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Abs = min(dataList)  # theoretical min: 0
            boundary2Abs = max(dataList)  # theoretical max: 1

    boundary1 = 0  # actual range of data
    boundary2 = 1

    fig, axes = plt.subplots(len(cueValidityArr) * len(adultTArr), len(priorArr), sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .10, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.13, .10, .35, .01])
        specialAxAbs = fig.add_axes([.56, .10, .35, .01])

    ix = 0
    for trackIDX in range(len(cueValidityArr) * len(adultTArr)):

        # pick the correct cue reliability
        cueValIDX = trackIDX / len(adultTArr)
        cueVal = cueValidityArr[cueValIDX]
        jx = 0

        for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
            if 'E' in prior:
                env = prior[-2]
            autoCorrDict = autoCorrDictTotal[prior]
            autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(prior))

            if levelsAutoCorrToPlotArg:
                levelsAutoCorrToPlot = sorted(autoCorrDict.values())
                subPlotsNum = len(levelsAutoCorrToPlot)
            else:
                levelsAutoCorrToPlot = None
                subPlotsNum = 3

            ax = ax_list[ix * len(priorArr) + jx]
            plt.sca(ax)

            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # gives phenotypic distance for adultT x autocorr combination
                heatPlot1.update(
                    {(adultT, val): distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}

            if levelsAutoCorrToPlot:

                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for adultT in adultTArr:
                    plasticityDict1.update({(val, adultT): heatPlot1[adultT, val] for val in autoCorrValSubset})
            else:

                cueValDictSubset = []
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]

                loopArrayLevl = ['low', 'moderate', 'high']

                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]
                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    cueValDictSubset.append(levl)
                    for adultT in adultTArr:
                        plasticityDict1.update(
                            ({(levl, adultT): np.mean([heatPlot1[adultT, val] for val in autoCorrValSubset], axis=0)}))

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=tValues).sort_index(
                axis=0, ascending=False)  # up until here everything is the same

            """
            next: need to subset to the correct data frame for the current subplot, we want to have the
            autocorrelations grouped per adultT
            """

            numPlots = len(ax_list)
            saveIdx = np.arange(0, numPlots, int(len(priorArr) * len(adultTArr)))
            currPlot = ix * len(priorArr) + jx

            currentSubset = \
                [list(np.arange(j, j + (len(priorArr) * len(adultTArr)))).index(currPlot) for i, j in enumerate(saveIdx)
                 if
                 currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)))][0]

            correctSubset = currentSubset / len(priorArr)
            """
            first todo extract pandas column values
            """
            plasticityDF1 = plasticityDF1.sort_index(axis=0, ascending=False)
            autoCorrRow, adultTRow = zip(*plasticityDF1.index.values)

            if levelsAutoCorrToPlot:
                reverseArg = True
            else:
                reverseArg = False
                autoCorrValSubset = cueValDictSubset

            uniqueAdultT = sorted(list(set(adultTRow)), reverse=True)
            rowIDX = [(x, uniqueAdultT[correctSubset]) for x in autoCorrValSubset]

            plasticityDF1Subset = plasticityDF1.loc[rowIDX, :].sort_index(axis=0, ascending=reverseArg)

            del plasticityDF1

            """
            optional: normalize results for better visibility of differences
            """
            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"

            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            """
            Customize when to display ticklabels
            """
            currentCorrLabels, currentTimeLabels = zip(*plasticityDF1Subset.index.values)

            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:  # in case it is uneven
                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        # specify the yticks and title
                        yticklabels = ['L', 'H']
                    else:
                        yticklabels = []
                else:  # if is even
                    if jx == 0 and ((currentSubset + 1) == ((len(adultTArr)) / 2)):
                        yticklabels = ['L', 'H']

                    else:
                        yticklabels = []
            else:
                yticklabels = []

            if (ix == len(cueValidityArr) * len(adultTArr) - 1) and jx == 0:

                xticklabels = tValues
            else:
                xticklabels = []
            cmap = 'Greys'

            if normalize:
                plt.suptitle(nameArg, fontsize=20)
            if ix == 0:

                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)
                cb = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cb.ax.set_xticklabels([round(boundary1, 2), round(boundary2, 2)])
                cb.ax.tick_params(labelsize=15)
                cb.set_label(currLabel, labelpad=10, size=20)

                if normalize:
                    ax.set_title("%s" % (prior), size=20, pad=30)
            else:
                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)

            plt.subplots_adjust(wspace=0.05, hspace=0.08)

            ax.set_yticklabels(yticklabels, fontsize=15)
            ax.set_yticks([0 + 0.4, subPlotsNum - 1 - 0.4])
            ax.set_xticks(np.arange(0,T,1))
            ax.set_xticklabels(xticklabels, fontsize=15)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cb.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            if (ix + 1) % len(adultTArr) == 0 and not (ix == (len(cueValidityArr) * len(adultTArr) - 1)):
                ax.plot([-0.01, 1.01], [0, 0], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if (ix + 1) % len(adultTArr) == 1 and not (ix == 0):
                ax.plot([-0.01, 1.01], [1, 1], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if ix == len(cueValidityArr) * len(adultTArr) - 1 and jx == 0:
                plt.xlabel("ontogeny", fontsize=20, labelpad=20)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == len(priorArr) - 1:

                numPlots = len(ax_list)
                saveIdx = np.arange(len(priorArr) - 1, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:  # uneven adult T array
                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                else:  # even
                    if (currentSubset + 1) == ((len(adultTArr)) / 2):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.set_label_coords(1.03, 0)
            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:

                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
                else:
                    if (currentSubset + 1) == ((len(
                            adultTArr)) / 2):  # (currPlot)% len(adultTArr) == 0 and ((currPlot/len(adultTArr))%2 ==1):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                        # ax.yaxis.set_label_coords(-0.03, 0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
                        # ax.yaxis.set_label_coords(-0.03, 0)
            jx += 1
        ix += 1

    fig.subplots_adjust(bottom=0.2)
    fig.text(0.03, 0.54, 'adult life span', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.97, 0.54, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityContourPlotInclOntogenyDifferentOrder33.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()



def plasticityAcrossAsymmetries(cueValidityArr, T, adultTArr, env, autoCorrDictTotal,
                                            twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType,
                                            lag, endOfExposure,
                                            studyArg, normalize, priorArr, levelsAutoCorrToPlotArg, nameArg):
    # select the plasticity argument
    arg = "relative"
    # switch adult T and and the correlation in this one
    tValues = np.arange(1, T + 1, 1)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for prior in priorArr:
        # here select the right autocorr dict and dataPath
        autoCorrDict_sorted = sorted(autoCorrDictTotal[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

        dataPath = os.path.join(mainDataPath, str(prior))
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"
                    exit()

                heatPlot1.update(
                    {(adultT, cueVal): distanceDict[(key, cueVal)] for key in
                     autoCorrKeys})

        if arg == 'relative':
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Rel = min(dataList)  # theoretical min: 0
            boundary2Rel = max(dataList)  # theoretical max: 1
        else:
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Abs = min(dataList)  # theoretical min: 0
            boundary2Abs = max(dataList)  # theoretical max: 1

    boundary1 = 0  # actual range of data
    boundary2 = 1

    fig, axes = plt.subplots(len(cueValidityArr) * len(adultTArr), len(priorArr), sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .10, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.13, .10, .35, .01])
        specialAxAbs = fig.add_axes([.56, .10, .35, .01])

    ix = 0
    for trackIDX in range(len(cueValidityArr) * len(adultTArr)):

        # pick the correct cue reliability
        cueValIDX = trackIDX / len(adultTArr)
        cueVal = cueValidityArr[cueValIDX]
        jx = 0

        for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
            autoCorrDict = autoCorrDictTotal[prior]
            autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(prior))

            if levelsAutoCorrToPlotArg:
                levelsAutoCorrToPlot = sorted(autoCorrDict.values())
                subPlotsNum = len(levelsAutoCorrToPlot)
            else:
                levelsAutoCorrToPlot = None
                subPlotsNum = 3

            ax = ax_list[ix * len(priorArr) + jx]
            plt.sca(ax)

            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # gives phenotypic distance for adultT x autocorr combination
                heatPlot1.update(
                    {(adultT, val): distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}

            if levelsAutoCorrToPlot:

                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for adultT in adultTArr:
                    plasticityDict1.update({(val, adultT): heatPlot1[adultT, val] for val in autoCorrValSubset})
            else:

                cueValDictSubset = []
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]

                loopArrayLevl = ['low', 'moderate', 'high']

                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]
                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    cueValDictSubset.append(levl)
                    for adultT in adultTArr:
                        plasticityDict1.update(
                            ({(levl, adultT): np.mean([heatPlot1[adultT, val] for val in autoCorrValSubset], axis=0)}))

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=tValues).sort_index(
                axis=0, ascending=False)  # up until here everything is the same

            """
            next: need to subset to the correct data frame for the current subplot, we want to have the
            autocorrelations grouped per adultT
            """

            numPlots = len(ax_list)
            saveIdx = np.arange(0, numPlots, int(len(priorArr) * len(adultTArr)))
            currPlot = ix * len(priorArr) + jx

            currentSubset = \
                [list(np.arange(j, j + (len(priorArr) * len(adultTArr)))).index(currPlot) for i, j in enumerate(saveIdx)
                 if
                 currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)))][0]

            correctSubset = currentSubset / len(priorArr)
            """
            first todo extract pandas column values
            """
            plasticityDF1 = plasticityDF1.sort_index(axis=0, ascending=False)
            autoCorrRow, adultTRow = zip(*plasticityDF1.index.values)

            if levelsAutoCorrToPlot:
                reverseArg = True
            else:
                reverseArg = False
                autoCorrValSubset = cueValDictSubset

            uniqueAdultT = sorted(list(set(adultTRow)), reverse=True)
            rowIDX = [(x, uniqueAdultT[correctSubset]) for x in autoCorrValSubset]

            plasticityDF1Subset = plasticityDF1.loc[rowIDX, :].sort_index(axis=0, ascending=reverseArg)

            del plasticityDF1

            """
            optional: normalize results for better visibility of differences
            """
            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"

            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            """
            Customize when to display ticklabels
            """
            currentCorrLabels, currentTimeLabels = zip(*plasticityDF1Subset.index.values)

            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:  # in case it is uneven
                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        # specify the yticks and title
                        yticklabels = ['L', 'H']
                    else:
                        yticklabels = []
                else:  # if is even
                    if jx == 0 and ((currentSubset + 1) == ((len(adultTArr)) / 2)):
                        yticklabels = ['L', 'H']

                    else:
                        yticklabels = []
            else:
                yticklabels = []

            if (ix == len(cueValidityArr) * len(adultTArr) - 1) and jx == 0:

                xticklabels = tValues
            else:
                xticklabels = []
            cmap = 'Greys'

            if ix == 0:

                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)
                cb = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cb.ax.set_xticklabels([round(boundary1, 2), round(boundary2, 2)])
                cb.ax.tick_params(labelsize=15)
                cb.set_label(currLabel, labelpad=10, size=20)

                if normalize:
                    ax.set_title("%s: %s" % (nameArg, prior), size=20, pad=30)
            else:
                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)

            plt.subplots_adjust(wspace=0.05, hspace=0.08)

            ax.set_yticklabels(yticklabels, fontsize=15)
            ax.set_yticks([0 + 0.4, subPlotsNum - 1 - 0.4])
            ax.set_xticks(np.arange(0,T,1))
            ax.set_xticklabels(xticklabels, fontsize=15)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cb.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            if (ix + 1) % len(adultTArr) == 0 and not (ix == (len(cueValidityArr) * len(adultTArr) - 1)):
                ax.plot([-0.01, 1.01], [0, 0], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if (ix + 1) % len(adultTArr) == 1 and not (ix == 0):
                ax.plot([-0.01, 1.01], [1, 1], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if ix == len(cueValidityArr) * len(adultTArr) - 1 and jx == 0:
                plt.xlabel("ontogeny", fontsize=20, labelpad=20)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == len(priorArr) - 1:

                numPlots = len(ax_list)
                saveIdx = np.arange(len(priorArr) - 1, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:  # uneven adult T array
                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                else:  # even
                    if (currentSubset + 1) == ((len(adultTArr)) / 2):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.set_label_coords(1.03, 0)
            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * len(adultTArr)))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * len(adultTArr)), len(priorArr))][0]

                if len(adultTArr) % 2 == 1:

                    if currentSubset == ((len(adultTArr) - 1) / 2):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
                else:
                    if (currentSubset + 1) == ((len(
                            adultTArr)) / 2):  # (currPlot)% len(adultTArr) == 0 and ((currPlot/len(adultTArr))%2 ==1):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                        # ax.yaxis.set_label_coords(-0.03, 0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
                        # ax.yaxis.set_label_coords(-0.03, 0)
            jx += 1
        ix += 1

    fig.subplots_adjust(bottom=0.2)
    fig.text(0.03, 0.54, 'adult life span', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.97, 0.54, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityAcrossAsymmetries.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotIncludingOntogeny3(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                         twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType,
                                         lag, endOfExposure,
                                         studyArg, normalize, levelsAutoCorrToPlot):
    # switch adult T and and the correlation in this one

    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
    tValues = np.arange(1, T + 1, 1)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for arg in ['relative', 'absolute']:
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"
                    exit()

                heatPlot1.update(
                    {(adultT, cueVal): distanceDict[(key, cueVal)] for key in
                     autoCorrKeys})

        if arg == 'relative':
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Rel = min(dataList)  # theoretical min: 0
            boundary2Rel = max(dataList)  # theoretical max: 1
        else:
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Abs = min(dataList)  # theoretical min: 0
            boundary2Abs = max(dataList)  # theoretical max: 1

    boundary1 = 0  # actual range of data
    boundary2 = 1

    """
    first step: create the data for heatplot 
    1. autocorrelationxadultT dict mapping onto sum of plasticity 
    """
    if levelsAutoCorrToPlot:
        subPlotsNum = len(levelsAutoCorrToPlot)
    else:
        subPlotsNum = 3

    fig, axes = plt.subplots(len(cueValidityArr) * len(adultTArr), 2, sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .10, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.13, .10, .35, .01])
        specialAxAbs = fig.add_axes([.56, .10, .35, .01])

    ix = 0
    for trackIDX in range(len(cueValidityArr) * len(adultTArr)):

        # pick the correct cue reliability
        cueValIDX = trackIDX / len(adultTArr)
        cueVal = cueValidityArr[cueValIDX]
        jx = 0

        for arg in ['relative', 'absolute']:  # one plot for relative, one plot for absolute phenotypic distance

            ax = ax_list[ix * 2 + jx]
            plt.sca(ax)


            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # gives phenotypic distance for adultT x autocorr combination
                heatPlot1.update(
                    {(adultT, val): distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for adultT in adultTArr:
                    plasticityDict1.update({(val, adultT): heatPlot1[adultT, val] for val in autoCorrValSubset})


            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    for adultT in adultTArr:
                        plasticityDict1.update(
                            ({(levl, adultT): np.mean([heatPlot1[adultT, val] for val in autoCorrValSubset], axis=0)}))

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=tValues).sort_index(
                axis=0, ascending=False)  # up until here everything is the same

            """
            next: need to subset to the correct data frame for the current subplot, we want to have the
            autocorrelations grouped per adultT
            """

            numPlots = len(ax_list)
            saveIdx = np.arange(0, numPlots, int(2 * len(adultTArr)))
            currPlot = ix * 2 + jx

            currentSubset = \
                [list(np.arange(j, j + (2 * len(adultTArr)))).index(currPlot) for i, j in enumerate(saveIdx) if
                 currPlot in np.arange(j, j + (2 * len(adultTArr)))][0]

            correctSubset = currentSubset / 2
            """
            first todo extract pandas column values
            """
            plasticityDF1 = plasticityDF1.sort_index(axis=0, ascending=False)

            autoCorrRow, adultTRow = zip(*plasticityDF1.index.values)
            if levelsAutoCorrToPlot:
                reverseArg = True
            else:
                reverseArg = False

            uniqueAdultT = sorted(list(set(adultTRow)), reverse=reverseArg)
            rowIDX = [(x, uniqueAdultT[correctSubset]) for x in autoCorrValSubset]
            plasticityDF1Subset = plasticityDF1.loc[rowIDX, :].sort_index(axis=0, ascending=True)
            del plasticityDF1

            """
            optional: normalize results for better visibility of differences
            """
            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"

            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            """
            Customize when to display ticklabels
            """
            currentCorrLabels, currentTimeLabels = zip(*plasticityDF1Subset.index.values)

            if len(adultTArr) % 2 == 1:  # in case it is uneven
                if jx == 0 and (currPlot + 1) % len(adultTArr) == 0:
                    # specify the yticks and title
                    yticklabels = ['L', 'H']
                else:
                    yticklabels = []
            else:  # if is even
                if jx == 0 and ((currPlot) % len(adultTArr) == 0 and ((currPlot / len(adultTArr)) % 2 == 1)):
                    yticklabels = ['L', 'H']
                else:
                    yticklabels = []

            if (ix == len(cueValidityArr) * len(adultTArr) - 1) and jx == 0:

                xticklabels = tValues
            else:
                xticklabels = []
            cmap = 'Greys'

            if ix == 0:

                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)
                cb = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cb.ax.set_xticklabels([round(boundary1, 2), round(boundary2, 2)])
                cb.ax.tick_params(labelsize=15)
                cb.set_label(currLabel, labelpad=10, size=20)

                if normalize:
                    ax.set_title("%s phenotypic distance" % arg, size=20, pad=30)
            else:
                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            ax.set_yticklabels(yticklabels, fontsize=15)
            ax.set_yticks([0 + 0.4, subPlotsNum - 1 - 0.4])
            ax.set_xticks(np.arange(0,T,1))
            ax.set_xticklabels(xticklabels, fontsize=15)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cb.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            if (ix + 1) % len(adultTArr) == 0 and not (ix == (len(cueValidityArr) * len(adultTArr) - 1)):
                ax.plot([-0.01, 1.01], [0, 0], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if (ix + 1) % len(adultTArr) == 1 and not (ix == 0):
                ax.plot([-0.01, 1.01], [1, 1], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if ix == len(cueValidityArr) * len(adultTArr) - 1 and jx == 0:
                plt.xlabel("ontogeny", fontsize=20, labelpad=20)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == 1:
                if len(adultTArr) % 2 == 1:  # uneven adult T array
                    if currPlot % len(adultTArr) == 0:
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                else:  # even
                    if (currPlot - 1) % len(adultTArr) == 0 and (((currPlot - 1) / len(adultTArr)) % 2 == 1):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.set_label_coords(1.03, 1.05)
            else:

                if len(adultTArr) % 2 == 1:
                    if (currPlot + 1) % len(adultTArr) == 0:  # (len(cueValidityArr) * len(adultTArr) - 1 ):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
                else:
                    if (currPlot) % len(adultTArr) == 0 and ((currPlot / len(adultTArr)) % 2 == 1):
                        plt.ylabel("%s \n \n $\it{r}$" % currentTimeLabels[0], fontsize=20, labelpad=0)
                    else:
                        plt.ylabel("%s \n \n " % currentTimeLabels[0], fontsize=20, labelpad=15)
            jx += 1
        ix += 1

    fig.subplots_adjust(bottom=0.2)
    fig.text(0.05, 0.54, 'adult life span', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.95, 0.54, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityContourPlotInclOntogenyDifferentOrder.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotIncludingOntogeny2_33(cueValidityArr, T, adultTArr, env, autoCorrDictTotal,
                                            twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType,
                                            lag, endOfExposure,
                                            studyArg, normalize, priorArr, levelsAutoCorrToPlotArg, nameArg):
    arg = "relative"
    tValues = np.arange(1, T + 1, 1)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for prior in priorArr:
        if 'E' in prior:
            env = prior[-2]
        autoCorrDict_sorted = sorted(autoCorrDictTotal[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        dataPath = os.path.join(mainDataPath, str(prior))
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"
                    exit()

                heatPlot1.update(
                    {(adultT, cueVal): distanceDict[(key, cueVal)] for key in
                     autoCorrKeys})

        if arg == 'relative':
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Rel = min(dataList)  # theoretical min: 0
            boundary2Rel = max(dataList)  # theoretical max: 1
        else:
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Abs = min(dataList)  # theoretical min: 0
            boundary2Abs = max(dataList)  # theoretical max: 1

    boundary1 = 0  # actual range of data
    boundary2 = 1

    """
    first step: create the data for heatplot 
    1. autocorrelationxadultT dict mapping onto sum of plasticity 
    """
    if levelsAutoCorrToPlotArg:

        lenLevelsAutoCorrToPlot = len(autoCorrDictTotal[prior].items())
        if lenLevelsAutoCorrToPlot > 5:
            subPlotsNum = 3
        else:
            subPlotsNum = lenLevelsAutoCorrToPlot
    else:
        subPlotsNum = 3

    fig, axes = plt.subplots(len(cueValidityArr) * subPlotsNum, len(priorArr), sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .10, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.13, .10, .35, .01])
        specialAxAbs = fig.add_axes([.56, .10, .35, .01])

    ix = 0
    for trackIDX in range(len(cueValidityArr) * subPlotsNum):
        # pick the correct cue reliability
        cueValIDX = trackIDX / subPlotsNum
        cueVal = cueValidityArr[cueValIDX]
        jx = 0

        for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
            if 'E' in prior:
                env = prior[-2]
            autoCorrDict = autoCorrDictTotal[prior]
            autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(prior))

            if levelsAutoCorrToPlotArg:

                if lenLevelsAutoCorrToPlot > 5:
                    subPlotsNum = 3
                    levelsAutoCorrToPlot = None  # in this case it will compute an aggregate

                else:
                    levelsAutoCorrToPlot = sorted(autoCorrDict.values())
                    subPlotsNum = len(levelsAutoCorrToPlot)
            else:
                levelsAutoCorrToPlot = None
                subPlotsNum = 3
            ax = ax_list[ix * len(priorArr) + jx]
            plt.sca(ax)

            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # gives phenotypic distance for adultT x autocorr combination
                heatPlot1.update(
                    {(adultT, val): distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for adultT in adultTArr:
                    plasticityDict1.update({(val, adultT): heatPlot1[adultT, val] for val in autoCorrValSubset})


            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]

                    for adultT in adultTArr:
                        plasticityDict1.update(
                            ({(levl, adultT): np.mean([heatPlot1[adultT, val] for val in autoCorrValSubset], axis=0)}))

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=tValues).sort_index(
                axis=0, ascending=False)

            """
            next: need to subset to the correct data frame for the current subplot
            """

            numPlots = len(ax_list)
            saveIdx = np.arange(0, numPlots, int(len(priorArr) * subPlotsNum))
            currPlot = ix * len(priorArr) + jx

            currentSubset = \
            [list(np.arange(j, j + (len(priorArr) * subPlotsNum))).index(currPlot) for i, j in enumerate(saveIdx) if
             currPlot in np.arange(j, j + (len(priorArr) * subPlotsNum))][0]

            correctSubset = currentSubset / len(priorArr)
            """
            first todo extract pandas column values
            """
            plasticityDF1 = plasticityDF1.sort_index(axis=0, ascending=False)

            autoCorrRow, adultTRow = zip(*plasticityDF1.index.values)
            if levelsAutoCorrToPlot:
                reverseArg = True
                uniqueAutoCorr = sorted(list(set(autoCorrRow)), reverse=reverseArg)
            else:
                reverseArg = False
                uniqueAutoCorr = ['high', 'moderate', 'low']

            rowIDX = [(uniqueAutoCorr[correctSubset], x) for x in adultTArr]
            plasticityDF1Subset = plasticityDF1.loc[rowIDX, :].sort_index(axis=0, ascending=True)
            del plasticityDF1

            """
            optional: normalize results for better visibility of differences
            """
            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"

            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            """
            Customize when to display ticklabels
            """
            currentCorrLabels, currentTimeLabels = zip(*plasticityDF1Subset.index.values)

            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * subPlotsNum))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))][0]
                if subPlotsNum % 2 == 1:  # in case it is uneven
                    if currentSubset == (subPlotsNum - 1) / 2:
                        # specify the yticks and title
                        yticklabels = ['S', 'L']
                    else:
                        yticklabels = []
                else:  # if is even
                    if ((currentSubset + 1) == (subPlotsNum) / 2):
                        yticklabels = ['S', 'L']
                    else:
                        yticklabels = []
            else:
                yticklabels = []

            if (ix == len(cueValidityArr) * subPlotsNum - 1) and jx == 0:

                xticklabels = tValues
            else:
                xticklabels = []
            cmap = 'Greys'

            if normalize:
                plt.suptitle(nameArg, fontsize=20)
            if ix == 0:

                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)
                cb = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cb.ax.set_xticklabels([round(boundary1, 2), round(boundary2, 2)])
                cb.ax.tick_params(labelsize=15)
                cb.set_label(currLabel, labelpad=10, size=20)

                if normalize:
                    ax.set_title("%s" % (prior), size=20, pad=30)
            else:
                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)

            plt.subplots_adjust(wspace=0.05, hspace=0.08)

            ax.set_yticklabels(yticklabels, fontsize=15)
            ax.set_yticks([0 + 0.4, len(adultTArr) - 1 - 0.4])
            ax.set_xticks(np.arange(0,T,1))
            ax.set_xticklabels(xticklabels, fontsize=15)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cb.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            if (ix + 1) % subPlotsNum == 0 and not (ix == (len(cueValidityArr) * subPlotsNum - 1)):
                ax.plot([-0.01, 1.01], [0, 0], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if (ix + 1) % subPlotsNum == 1 and not (ix == 0):
                ax.plot([-0.01, 1.01], [1, 1], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if ix == len(cueValidityArr) * subPlotsNum - 1 and jx == 0:
                plt.xlabel("ontogeny", fontsize=20, labelpad=20)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == len(priorArr) - 1:
                numPlots = len(ax_list)
                saveIdx = np.arange(len(priorArr) - 1, numPlots,
                                    int(len(priorArr) * subPlotsNum))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))][0]

                if subPlotsNum % 2 == 1:
                    if currentSubset == ((subPlotsNum - 1) / 2):  # uneven adult T array
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                else:  # even
                    if (currentSubset + 1) == ((subPlotsNum) / 2):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.set_label_coords(1.03, 0)
            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(priorArr) * subPlotsNum))  # np.arange(0, numPlots, len(priorArr))

                currPlot = ix * len(priorArr) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(priorArr) * subPlotsNum), len(priorArr))][0]

                if subPlotsNum % 2 == 1:

                    if currentSubset == ((subPlotsNum - 1) / 2):
                        plt.ylabel("%s \n \n adulthood" % currentCorrLabels[0], fontsize=20, labelpad=5)
                    else:
                        plt.ylabel("%s \n \n " % currentCorrLabels[0], fontsize=20, labelpad=20)
                else:
                    if (currentSubset + 1) == ((subPlotsNum) / 2):
                        plt.ylabel("%s \n \n adulthood" % currentCorrLabels[0], fontsize=20, labelpad=5)
                    else:
                        plt.ylabel("%s \n \n " % currentCorrLabels[0], fontsize=20, labelpad=20)
            jx += 1
        ix += 1

    fig.subplots_adjust(bottom=0.2)
    fig.text(0.03, 0.55, 'autocorrelation', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.97, 0.55, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityContourPlotInclOntogeny33.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotIncludingOntogeny2(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                         twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType,
                                         lag, endOfExposure,
                                         studyArg, normalize, levelsAutoCorrToPlot):
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
    tValues = np.arange(1, T + 1, 1)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for arg in ['relative', 'absolute']:
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"

                heatPlot1.update(
                    {(adultT, cueVal): distanceDict[(key, cueVal)] for key in
                     autoCorrKeys})

        if arg == 'relative':
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Rel = min(dataList)  # theoretical min: 0
            boundary2Rel = max(dataList)  # theoretical max: 1
        else:
            dataList = list(itertools.chain.from_iterable(heatPlot1.values()))
            boundary1Abs = min(dataList)  # theoretical min: 0
            boundary2Abs = max(dataList)  # theoretical max: 1

    boundary1 = 0  # actual range of data
    boundary2 = 1

    """
    first step: create the data for heatplot 
    1. autocorrelationxadultT dict mapping onto sum of plasticity 
    """
    if levelsAutoCorrToPlot:

        if len(levelsAutoCorrToPlot) > 5:
            subPlotsNum = 3
            levelsAutoCorrToPlot = None  # in this case it will compute an aggregate

        else:
            subPlotsNum = len(levelsAutoCorrToPlot)
    else:
        subPlotsNum = 3

    fig, axes = plt.subplots(len(cueValidityArr) * subPlotsNum, 2, sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .10, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.13, .10, .35, .01])
        specialAxAbs = fig.add_axes([.56, .10, .35, .01])

    ix = 0
    for trackIDX in range(len(cueValidityArr) * subPlotsNum):

        # pick the correct cue reliability
        cueValIDX = trackIDX / subPlotsNum
        cueVal = cueValidityArr[cueValIDX]
        jx = 0

        for arg in ['relative', 'absolute']:  # one plot for relative, one plot for absolute phenotypic distance

            ax = ax_list[ix * 2 + jx]
            plt.sca(ax)


            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # gives phenotypic distance for adultT x autocorr combination
                heatPlot1.update(
                    {(adultT, val): distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for adultT in adultTArr:
                    plasticityDict1.update({(val, adultT): heatPlot1[adultT, val] for val in autoCorrValSubset})


            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]

                    for adultT in adultTArr:
                        plasticityDict1.update(
                            ({(levl, adultT): np.mean([heatPlot1[adultT, val] for val in autoCorrValSubset], axis=0)}))

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=tValues).sort_index(
                axis=0, ascending=False)

            """
            next: need to subset to the correct data frame for the current subplot
            """

            numPlots = len(ax_list)
            saveIdx = np.arange(0, numPlots, int(2 * subPlotsNum))
            currPlot = ix * 2 + jx

            currentSubset = [list(np.arange(j, j + (2 * subPlotsNum))).index(currPlot) for i, j in enumerate(saveIdx) if
                             currPlot in np.arange(j, j + (2 * subPlotsNum))][0]

            correctSubset = currentSubset / 2
            """
            first todo extract pandas column values
            """
            plasticityDF1 = plasticityDF1.sort_index(axis=0, ascending=False)

            autoCorrRow, adultTRow = zip(*plasticityDF1.index.values)
            if levelsAutoCorrToPlot:
                reverseArg = True
                uniqueAutoCorr = sorted(list(set(autoCorrRow)), reverse=reverseArg)
            else:
                reverseArg = False
                uniqueAutoCorr = ['high', 'moderate', 'low']

            rowIDX = [(uniqueAutoCorr[correctSubset], x) for x in adultTArr]
            plasticityDF1Subset = plasticityDF1.loc[rowIDX, :].sort_index(axis=0, ascending=True)
            del plasticityDF1

            """
            optional: normalize results for better visibility of differences
            """
            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"

            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            """
            Customize when to display ticklabels
            """
            currentCorrLabels, currentTimeLabels = zip(*plasticityDF1Subset.index.values)

            if subPlotsNum % 2 == 1:  # in case it is uneven
                if jx == 0 and (currPlot + 1) % subPlotsNum == 0:
                    # specify the yticks and title
                    yticklabels = ['S', 'L']
                else:
                    yticklabels = []
            else:  # if is even
                if jx == 0 and ((currPlot) % subPlotsNum == 0 and ((currPlot / subPlotsNum) % 2 == 1)):
                    yticklabels = ['S', 'L']
                else:
                    yticklabels = []

            if (ix == len(cueValidityArr) * subPlotsNum - 1) and jx == 0:

                xticklabels = tValues
            else:
                xticklabels = []
            cmap = 'Greys'

            if ix == 0:

                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)
                cb = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cb.ax.set_xticklabels([round(boundary1, 2), round(boundary2, 2)])
                cb.ax.tick_params(labelsize=15)
                cb.set_label(currLabel, labelpad=10, size=20)

                if normalize:
                    ax.set_title("%s phenotypic distance" % arg, size=20, pad=30)
            else:
                g = plt.contourf(plasticityDF1Subset, list(np.arange(boundary1, boundary2 + 0.05, 0.05)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)
                # plt.contour(plasticityDF1Subset, list(np.arange(0, 1.05, 0.05)), colors = 'black',linewidths = 1, vmin=boundary1 - 0.01,
                #             vmax=boundary2)

            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            ax.set_yticklabels(yticklabels, fontsize=15)
            ax.set_yticks([0 + 0.4, len(adultTArr) - 1 - 0.4])
            ax.set_xticks(np.arange(0,T,1))
            ax.set_xticklabels(xticklabels, fontsize=15)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cb.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            if (ix + 1) % subPlotsNum == 0 and not (ix == (len(cueValidityArr) * subPlotsNum - 1)):
                ax.plot([-0.01, 1.01], [0, 0], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if (ix + 1) % subPlotsNum == 1 and not (ix == 0):
                ax.plot([-0.01, 1.01], [1, 1], color='black', lw=2, ls='solid', transform=ax.transAxes, clip_on=False)

            if ix == len(cueValidityArr) * subPlotsNum - 1 and jx == 0:
                plt.xlabel("ontogeny", fontsize=20, labelpad=20)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == 1:
                if subPlotsNum % 2 == 1:  # uneven adult T array
                    if currPlot % subPlotsNum == 0:
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                else:  # even
                    if (currPlot - 1) % subPlotsNum == 0 and (((currPlot - 1) / subPlotsNum) % 2 == 1):
                        plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.set_label_coords(1.03, 1.05)
            else:

                if subPlotsNum % 2 == 1:
                    if (currPlot + 1) % subPlotsNum == 0:  # (len(cueValidityArr) * len(adultTArr) - 1 ):
                        plt.ylabel("%s \n \n adulthood" % currentCorrLabels[0], fontsize=20, labelpad=5)
                    else:
                        plt.ylabel("%s \n \n " % currentCorrLabels[0], fontsize=20, labelpad=20)
                else:
                    if (currPlot) % subPlotsNum == 0 and ((currPlot / subPlotsNum) % 2 == 1):
                        plt.ylabel("%s \n \n adulthood" % currentCorrLabels[0], fontsize=20, labelpad=5)
                    else:
                        plt.ylabel("%s \n \n " % currentCorrLabels[0], fontsize=20, labelpad=20)

            jx += 1
        ix += 1

    fig.subplots_adjust(bottom=0.2)
    fig.text(0.03, 0.55, 'autocorrelation', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.95, 0.55, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityContourPlotInclOntogeny.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotAggregatedAcrossOntogeny_33(cueValidityArr, T, adultTArr, env, autoCorrDictTotal,
                                                  twinResultsAggregatedPath, mainDataPath, argumentR, argumentP,
                                                  adoptionType,
                                                  lag, endOfExposure,
                                                  studyArg, normalize, priorArr, nameArg):
    arg = "relative"

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for prior in priorArr:
        if 'E' in prior:
            env = prior[-2]
        autoCorrDict_sorted = sorted(autoCorrDictTotal[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        dataPath = os.path.join(mainDataPath, str(prior))
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"
                    exit()

                heatPlot1.update(
                    {(adultT, cueVal, key, arg): sum(distanceDict[(key, cueVal)]) / float(T) for key, val in
                     autoCorrDict_sorted})
        if arg == 'relative':
            boundary1Rel = min(heatPlot1.values())  # theoretical min: 0
            boundary2Rel = max(heatPlot1.values())  # theoretical max: 1
        else:
            boundary1Abs = min(heatPlot1.values())  # theoretical min: 0
            boundary2Abs = max(heatPlot1.values())  # theoretical max: 1

    boundary1 = 0  # min([boundary1Rel, boundary1Abs])-0.1 #actual range of data
    boundary2 = 0.6  # max([boundary2Rel, boundary2Abs])+0.1

    """
    first step: create the data for heatplot 
    1. autocorrelationxadultT dict mapping onto sum of plasticity 
    """
    fig, axes = plt.subplots(len(cueValidityArr), len(priorArr), sharex=False, sharey=True)
    fig.set_size_inches(18, 18)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .22, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.1, .22, .4, .01])
        specialAxAbs = fig.add_axes([.52, .22, .4, .01])

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
            if 'E' in prior:
                env = prior[-2]

            autoCorrDict_sorted = sorted(autoCorrDictTotal[prior].items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(prior))
            ax = ax_list[ix * len(priorArr) + jx]
            plt.sca(ax)

            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                heatPlot1.update(
                    {(adultT, val): sum(distanceDict[(key, cueVal)]) / float(T) for key, val in autoCorrDict_sorted})


            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}
            for adultT in adultTArr:
                plasticityDict1[adultT] = [heatPlot1[(adultT, autoCorr)] for autoCorr in autoCorrVal]

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=autoCorrVal).sort_index(
                axis=0, ascending=True)

            """
            optional: normalize results for better visibility of differences
            """

            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"
            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            if ix == len(cueValidityArr) - 1:
                xticklabels = autoCorrVal
            else:
                xticklabels = []

            if normalize:
                plt.suptitle(nameArg, fontsize=20)
            if ix == 0:

                cmap = 'Greys'

                g = plt.contourf(plasticityDF1, list(np.arange(boundary1, boundary2 + 0.02, 0.02)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)

                ax.tick_params(labelsize=15)

                cbar = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(currLabel, labelpad=10, size=20)
                if normalize:
                    ax.set_title("%s" % (prior), size=20, pad=30)

            else:
                g = plt.contourf(plasticityDF1, list(np.arange(boundary1, boundary2 + 0.02, 0.02)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)

                ax.tick_params(labelsize=15, pad=10)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                plt.xlabel("autocorrelation", fontsize=20, labelpad=15)

                xTicks = np.arange(0, len(autoCorrVal) - 1, 1) + 0.1
                xTicks = list(xTicks) + [len(autoCorrVal) - 1 - 0.1]
                xTicksFinal = [xTicks[0], xTicks[-1]]
                ax.set_xticks(xTicksFinal)
                ax.set_xticklabels(['L', "H"], fontsize=15)
            else:
                ax.set_xticks([], [])

            if ix == 0:
                ax.set_yticklabels(adultTArr, fontsize=15)
                yTicks = np.arange(0, len(adultTArr) - 1, 1) + 0.1
                yTicks = list(yTicks) + [len(adultTArr) - 1 - 0.1]
                ax.set_yticks(yTicks)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == len(priorArr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cbar.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            jx += 1
        ix += 1

    if len(adultTArr) == 5:
        fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.1)
    else:
        fig.subplots_adjust(bottom=0.3, wspace=0.1, hspace=0.1)
    fig.text(0.08, 0.395, 'adult life span', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.98, 0.59, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityHeatPlotAgrrOntogeny33.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plasticityHeatPlotAggregatedAcrossOntogeny(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                               twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType,
                                               lag, endOfExposure,
                                               studyArg, normalize):
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

    """
    find the max for plotting
    """
    boundary1Rel = 0
    boundary2Rel = 0
    boundary1Abs = 0
    boundary2Abs = 0

    for arg in ['relative', 'absolute']:
        heatPlot1 = {}
        for cueVal in cueValidityArr:
            # one plot for relative, one plot for absolute phenotypic distance
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print "No data availabale"

                heatPlot1.update(
                    {(adultT, cueVal, key, arg): sum(distanceDict[(key, cueVal)]) / float(T) for key, val in
                     autoCorrDict_sorted})
        if arg == 'relative':
            boundary1Rel = min(heatPlot1.values())  # theoretical min: 0
            boundary2Rel = max(heatPlot1.values())  # theoretical max: 1
        else:
            boundary1Abs = min(heatPlot1.values())  # theoretical min: 0
            boundary2Abs = max(heatPlot1.values())  # theoretical max: 1

    boundary1 = 0  # min([boundary1Rel, boundary1Abs])-0.1 #actual range of data
    boundary2 = 0.7  # max([boundary2Rel, boundary2Abs])+0.1

    """
    first step: create the data for heatplot 
    1. autocorrelationxadultT dict mapping onto sum of plasticity 
    """
    fig, axes = plt.subplots(len(cueValidityArr), 2, sharex=True, sharey=True)
    fig.set_size_inches(18, 18)
    ax_list = fig.axes

    if normalize:
        specialAx = fig.add_axes([.16, .22, .7, .01])  # .040
    else:
        specialAxRel = fig.add_axes([.1, .22, .4, .01])
        specialAxAbs = fig.add_axes([.52, .22, .4, .01])

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for arg in ['relative', 'absolute']:  # one plot for relative, one plot for absolute phenotypic distance

            ax = ax_list[ix * 2 + jx]
            plt.sca(ax)
            # if len(adultTArr) == 5:
            #     ax.set(aspect='equal')
            # else:
            #     ax.set(aspect = 1.5)

            heatPlot1 = {}
            for adultT in adultTArr:  # one plot for adult T
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                heatPlot1.update(
                    {(adultT, val): sum(distanceDict[(key, cueVal)]) / float(T) for key, val in autoCorrDict_sorted})

            """
            create a 2D matrix for the heatplot
            """
            plasticityDict1 = {}
            for adultT in adultTArr:
                plasticityDict1[adultT] = [heatPlot1[(adultT, autoCorr)] for autoCorr in autoCorrVal]

            plasticityDF1 = pd.DataFrame.from_dict(plasticityDict1, orient="index", columns=autoCorrVal).sort_index(
                axis=0, ascending=True)

            """
            optional: normalize results for better visibility of differences
            """

            if normalize:
                orientation = 'horizontal'
                fraction = 0.15,
                pad = 0.15
                ticks = [boundary1, boundary2]
                currLabel = "phenotypic distance"
            else:
                if arg == "relative":
                    boundary1 = boundary1Rel
                    boundary2 = boundary2Rel
                    specialAx = specialAxRel
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "relative phenotypic distance"
                else:
                    boundary1 = boundary1Abs
                    boundary2 = boundary2Abs
                    specialAx = specialAxAbs
                    orientation = 'horizontal'
                    fraction = 0.15,
                    pad = 0.15
                    ticks = [boundary1, boundary2]
                    currLabel = "absolute phenotypic distance"

            if ix == len(cueValidityArr) - 1:
                xticklabels = autoCorrVal
            else:
                xticklabels = []

            if ix == 0:

                cmap = 'Greys'

                g = plt.contourf(plasticityDF1, list(np.arange(boundary1, boundary2 + 0.01, 0.01)), cmap=cmap,
                                 vmin=boundary1 - 0.01, vmax=boundary2)

                ax.tick_params(labelsize=15)

                cbar = plt.colorbar(g, orientation=orientation, fraction=fraction, pad=pad, ticks=ticks, cax=specialAx)

                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(currLabel, labelpad=10, size=20)
                if normalize:
                    ax.set_title("%s phenotypic distance" % arg, size=20, pad=30)

            else:
                g = plt.contourf(plasticityDF1, list(np.arange(boundary1, boundary2 + 0.01, 0.01)), cmap=cmap,
                                 alpha=1, vmin=boundary1 - 0.01, vmax=boundary2)

                ax.tick_params(labelsize=15, pad=10)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                plt.xlabel("autocorrelation", fontsize=20, labelpad=20)

            if ix == 0:
                ax.set_yticklabels(adultTArr, fontsize=15)
                yTicks = np.arange(0, len(adultTArr) - 1, 1) + 0.1
                yTicks = list(yTicks) + [len(adultTArr) - 1 - 0.1]
                ax.set_yticks(yTicks)

            ax.tick_params(axis='both', which='both', length=0, pad=10)

            if jx == 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
                ax.set_xticklabels(autoCorrVal, fontsize=15)
                xTicks = np.arange(0, len(autoCorrVal) - 1, 1) + 0.1
                xTicks = list(xTicks) + [len(autoCorrVal) - 1 - 0.1]
                ax.set_xticks(xTicks)

            # removing the frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            cbar.outline.set_visible(False)
            plt.yticks(rotation='vertical')

            jx += 1
        ix += 1

    if len(adultTArr) == 5:
        fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.1)
    else:
        fig.subplots_adjust(bottom=0.3, wspace=0.1, hspace=0.1)
    fig.text(0.08, 0.395, 'adult life span', fontsize=20, ha='center', va='center', rotation='vertical')
    fig.text(0.98, 0.59, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')

    # fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes, rotation='vertical')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityHeatPlotAgrrOntogeny.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False)),
        dpi=1000)

    plt.close()


def plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                   twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType, lag,
                                   endOfExposure,
                                   studyArg, priorArr, nameArg):

    arg = "relative"  # choose whether you want this plot for relative or absolute distance

    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    levelsAutoCorrToPlot = [0.2,0.5,0.8]  #TODO change this back to None

    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)
    fig, axes = plt.subplots(len(cueValidityArr), len(priorArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
        if prior == '05':
            levelsAutoCorrToPlot = [0.2, 0.5, 0.77] # otherwise it picks 0.85 instead of 0.75
        else:
            levelsAutoCorrToPlot = [0.2, 0.5, 0.8]


        if 'E' in prior:
            env = prior[-2]

        # here select the right autocorr dict and dataPath
        autoCorrDict_sorted = sorted(autoCorrDict[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        dataPath = os.path.join(mainDataPath, str(prior))



        ix = 0
        for cueVal in cueValidityArr:
            ax = ax_list[ix * len(priorArr) + jx]

            plt.sca(ax)
            for i, adultT in enumerate(adultTArr):  # one line per adult T
                # linestyle depends on current adult T value
                linestyle = linestyle_tuple[i][1]
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)


                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print 'no data available'
                    exit()

                # for he current cueVal load the distancedictionaries
                cueValDict = {val: distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted}

                if levelsAutoCorrToPlot:
                    # the next line find the indices of the closest autocorrelation values that match the user input
                    idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                           levelsAutoCorrToPlot]
                    autoCorrValSubset = np.array(autoCorrVal)[idx]
                    [plt.plot(tValues, cueValDict[autoCorr], linestyle=linestyle, linewidth=2, markersize=5,
                              marker='o', color=str(1 - autoCorr - 0.1), markerfacecolor=str(1 - autoCorr - 0.1),
                              label="%s & %s" % (autoCorr, adultT)) for autoCorr in autoCorrValSubset]

                else:
                    """
                    in case that the user did not specify values to pick, compute an average
                    - first need to calculate cutoff points
                    """
                    extremeIDX = np.floor(len(autoCorrVal) / float(3))
                    midIDX = np.ceil(len(autoCorrVal) / float(3))
                    loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                    loopArrayLevl = ['low', 'moderate', 'high']

                    cueValDictSubset = {}
                    for idx in range(len(loopArrayIDX)):
                        levl = loopArrayLevl[idx]

                        if idx == 0:
                            startIdx = int(idx)
                        else:
                            startIdx = int(endIdx)
                        endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                        autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                        plastcityVal = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                        cueValDictSubset[levl] = plastcityVal

                    colorArr = [0.1, 0.5, 0.9]
                    [plt.plot(tValues, cueValDictSubset[autoCorr], linestyle=linestyle, linewidth=2, markersize=5,
                              marker='o', color=str(1 - colorArr[idx] - 0.1),
                              markerfacecolor=str(1 - colorArr[idx] - 0.1),
                              label="%s & %s" % (autoCorr[0].upper(), adultT)) for idx, autoCorr in
                     enumerate(loopArrayLevl)]

            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """
            plt.plot(tValues, [0] * T, linestyle='dashed', linewidth=1, color='grey')
            plt.plot(tValues, [1] * T, linestyle='dashed', linewidth=1, color='grey')

            if ix == len(cueValidityArr) - 1 and jx == 0:
                anchPar = len(priorArr) / float(2)
                legend = ax.legend(loc='upper center', bbox_to_anchor=(anchPar, -0.3),
                                   title='autocorrelation & adult lifespan',
                                   ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=20)
                plt.setp(legend.get_title(), fontsize='20')

            plt.suptitle(nameArg, fontsize = 20)
            if ix == 0:
                plt.title("%s" % (prior), fontsize=20)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
            plt.xticks([])

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                plt.xlabel('ontogeny', fontsize=20, labelpad=20)

            if jx == len(priorArr) - 1:
                ax.yaxis.set_label_position("right")
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)

            ix += 1
        jx += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.2, top=0.90)
    fig.text(0.98, 0.54, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.08, 0.31, 'phenotypic distance', fontsize=20, ha='center', va='center', rotation='vertical')
    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityOverviewTotal33%s.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False, levelsAutoCorrToPlot)),
        dpi=1000)

    plt.close()


def ternaryPlotOverviewJitter(cueValidityArr, T, adultTArr, env, autoCorrDict,
                        twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                        studyArg, levelsAutoCorrToPlot):
    permuteIdx = [0, 2, 1]
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)
    fig, axes = plt.subplots(len(cueValidityArr), len(adultTArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for adultT in adultTArr:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(adultTArr) + jx]
            #ax.set(aspect=1)
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = "maturePhenotypes_%s.p" % (env)
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                maturePhenotypesDict = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                maturePhenotypesDict.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: maturePhenotypesDict[(key, cueVal)] for key, val in autoCorrDict_sorted}
            # cueValDict contains a tuple now; first part are the mature phenotypes and second the cue prob

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                for idx, autoCorr in enumerate(autoCorrValSubset):
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorr]
                    numAgents = maturePhenotypesCurr.shape[0]

                    # now need to work on the scaling of points

                    unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                    # area = area_calc(uniqueCounts / float(numAgents), 150)
                    if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                        uniqueFrac = []
                        for matPhen in unique:
                            probIdx = np.where(
                                (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                        maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                        maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                            uniqueFrac.append(sum(cueProbabilities[probIdx]))

                        area2 = np.array(uniqueFrac) * float(400)


                    else:
                        area2 = (uniqueCounts / float(numAgents)) * 400
                    # this one would be scalling according to area
                    colorShades = [0.2, 0.5, 1]
                    colorTuple = [0, 0, 0, 1]


                    if idx == 0:
                        toPlot = np.copy(np.array(unique[:, permuteIdx]))
                        toPlot[:,2] += 0.1 #TODO continue here ; true copy?
                        toPlot[:, 0] -= 0.1


                        colorTuple[3] = colorShades[idx]
                        tax.scatter(toPlot, s=area2, marker=MarkerStyle(marker='o', fillstyle='left'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    elif idx == 1:
                        toPlot = np.copy(np.array(unique[:, permuteIdx]))
                        toPlot[:,0] += 0.1
                        toPlot[:, 2] -= 0.1

                        colorTuple[3] = colorShades[idx]
                        tax.scatter(toPlot, s=area2, marker=MarkerStyle(marker='o', fillstyle='right'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    else:
                        toPlot = np.copy(np.array(unique[:, permuteIdx]))
                        #toPlot[:,1] += 0.1

                        #toPlot[:, 0] -= 0.05
                        #toPlot[:, 2] -= 0.05

                        colorTuple[3] = colorShades[idx]
                        tax.scatter(toPlot, s=area2, linewidths=0.6, facecolors='none',
                                   edgecolors=colorTuple,
                                   label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    if levl == 'moderate':
                        if len(autoCorrValSubset)==3:
                            test = []
                            test.append(autoCorrValSubset[0])
                            test.append(autoCorrValSubset[2])
                            autoCorrValSubset = test

                    # merge data for these autoCorr values!
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorrValSubset[0]]
                    for autoCorr in autoCorrValSubset[1:]:
                        """
                        continue here; need to merge all the results
                        """
                        maturePhenotypesCurrNext, cueProbabilitiesNext = cueValDict[autoCorr]
                        maturePhenotypesCurr = np.concatenate((maturePhenotypesCurr, maturePhenotypesCurrNext))
                        cueProbabilities = np.concatenate((cueProbabilities, cueProbabilitiesNext))
                    cueValDictSubset[levl] = (maturePhenotypesCurr, cueProbabilities)

                colorArr = [0.1, 0.5, 0.9]
                for idx, autoCorr in enumerate(loopArrayLevl):
                    maturePhenotypesCurr, cueProbabilities = cueValDictSubset[autoCorr]
                    numAgents = maturePhenotypesCurr.shape[0]
                    tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                    # now need to work on the scaling of points

                    unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                    # area = area_calc(uniqueCounts / float(numAgents), 150)
                    if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                        uniqueFrac = []
                        for matPhen in unique:
                            probIdx = np.where(
                                (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                        maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                        maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                            uniqueFrac.append(sum(cueProbabilities[probIdx]))

                        area2 = np.array(uniqueFrac) * float(130)
                    else:
                        area2 = (uniqueCounts / float(numAgents)) * 130
                    # this one would be scalling according to area
                    # colVal =1-colorArr[idx]
                    colorShades = [0.2, 0.5, 1]
                    colorTuple = [0, 0, 0, 1]
                    if idx == 0:
                        toPlot = np.copy(np.array(unique[:, permuteIdx]))
                        toPlot[:,2] += 0.1 #TODO continue here ; true copy?
                        toPlot[:, 0] -= 0.1

                        colorTuple[3] = colorShades[idx]
                        tax.scatter(toPlot, s=area2, marker=MarkerStyle(marker='o', fillstyle='left'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    elif idx == 1:
                        toPlot = np.copy(np.array(unique[:, permuteIdx]))
                        toPlot[:,0] += 0.1
                        toPlot[:, 2] -= 0.1

                        colorTuple[3] = colorShades[idx]
                        tax.scatter(toPlot, s=area2, marker=MarkerStyle(marker='o', fillstyle='right'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    else:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, linewidths=0.6, facecolors='none',
                                    edgecolors=colorTuple,
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            if ix == len(cueValidityArr) - 1 and jx == len(adultTArr) / 2:
                legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), title='autocorrelation',
                                   ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=20)

                for i in range(len(legend.legendHandles)):
                    legend.legendHandles[i]._sizes = [100]
                plt.setp(legend.get_title(), fontsize='20')
            if ix == 0:
                plt.title("%s" % adultT, fontsize=20)
            else:
                ax.get_xaxis().set_visible(False)

            if jx == len(adultTArr) - 1:
                plt.ylabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            if jx == len(adultTArr) / 2 and ix == len(cueValidityArr) / 2:
                fontsize = 20
                tax.right_corner_label("P0", fontsize=fontsize, offset=-0.15)
                tax.top_corner_label("wait time", fontsize=fontsize)
                tax.left_corner_label("P1", fontsize=fontsize, offset=-0.15)
                tax._redraw_labels()

            ix += 1
        jx += 1
    plt.suptitle('adult life span', fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.2, top=0.9)
    fig.text(0.98, 0.58, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sTernaryOverviewTotalJitter%s.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False, levelsAutoCorrToPlot)),
        dpi=1000)

    plt.close()


def calcPointToPlot(focalPoint, idx):
    sidePoints = (10-focalPoint)/float(3)
    focalResult = 10 - (2*sidePoints)
    result = [sidePoints]*3
    result[idx] = focalResult
    return result


def ternaryMercedesMerge(cueValidityArr, T, adultTArr, env, autoCorrDict,
                        twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                        studyArg, levelsAutoCorrToPlot, nameArg, title):

    permuteIdx = [0, 2, 1]

    rowVec = []
    for currX in nameArg:
        for currY in adultTArr:
            rowVec.append((currX, currY))

    columnVec = []
    for currX in levelsAutoCorrToPlot:
        for currY in cueValidityArr:
            columnVec.append((currX,currY))
    # prepare the x-axis values
    fig, axes = plt.subplots(len(columnVec), len(rowVec), sharex=False, sharey=False)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes


    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)

    jx = 0
    for symmArg, adultT in rowVec:  # one column per adultT
        ix = 0
        for autoCorrCurr,cueVal in columnVec:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect="equal")
            plt.sca(ax)

            autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(symmArg))

            # get the data for this adult T and a specific cue reliability value
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            if "E" in symmArg:
                env = symmArg[-2]
            else:
                env = 0
            fileName = "maturePhenotypes_%s.p" % (env)
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                maturePhenotypesDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    maturePhenotypesDict.update(pickle.load(open(filePath1, 'rb')))
            else:
                maturePhenotypesDict = {}
                print "no data"
                exit()
            # for the current cueVal load the distancedictionaries
            cueValDict = {val: maturePhenotypesDict[(key, cueVal)] for key, val in autoCorrDict_sorted}
            # cueValDict contains a tuple now; first part are the mature phenotypes and second the cue prob

            if not "low" in levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - autoCorrCurr))]
                autoCorr = np.array(autoCorrVal)[idx][0]
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorr]
                numAgents = maturePhenotypesCurr.shape[0]

                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)

                # area = area_calc(uniqueCounts / float(numAgents), 150)
                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                    maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))

                    area2 = np.array(uniqueFrac) * float(95)

                    averagePhenotype = np.average(unique,axis = 0, weights= uniqueFrac)[permuteIdx]

                else:
                    averagePhenotype = np.average(unique, axis=0, weights=uniqueCounts)[permuteIdx]
                    area2 = (uniqueCounts / float(numAgents)) * 95


                toPlot = np.copy(np.array(unique[:, permuteIdx]))

                tax.scatter(toPlot, s=area2, linewidths=0.6, facecolors='#B8B8B8',
                           edgecolors='#B8B8B8')
                tax.boundary(axes_colors={'l': '#D3D3D3', 'r': "#D3D3D3", 'b': "#D3D3D3"}, linewidth=0.6, zorder=-1)


                """
                plot the average phenotype
                """


                midPoint = [10 / float(3), 10 / float(3), 10 / float(3)]

                P0 = calcPointToPlot(averagePhenotype[0], 0)
                P1 = calcPointToPlot(averagePhenotype[2], 2)
                Pw = calcPointToPlot(averagePhenotype[1],1)


                tax.line(P1,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")
                tax.line(P0,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")
                tax.line(Pw,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")


            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]


                    # merge data for these autoCorr values!
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorrValSubset[0]]
                    for autoCorr in autoCorrValSubset[1:]:
                        """
                        continue here; need to merge all the results
                        """
                        maturePhenotypesCurrNext, cueProbabilitiesNext = cueValDict[autoCorr]
                        maturePhenotypesCurr = np.concatenate((maturePhenotypesCurr, maturePhenotypesCurrNext))
                        cueProbabilities = np.concatenate((cueProbabilities, cueProbabilitiesNext))
                    cueValDictSubset[levl] = (maturePhenotypesCurr, cueProbabilities)

                autoCorr = autoCorrCurr
                maturePhenotypesCurr, cueProbabilities = cueValDictSubset[autoCorr]
                numAgents = maturePhenotypesCurr.shape[0]
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                # area = area_calc(uniqueCounts / float(numAgents), 150)
                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                    maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))
                    averagePhenotype = np.average(unique, axis=0, weights=uniqueFrac)[permuteIdx]
                    area2 = np.array(uniqueFrac) * float(50)
                else:
                    averagePhenotype = np.average(unique, axis=0, weights=uniqueCounts)[permuteIdx]
                    area2 = (uniqueCounts / float(numAgents)) * 50

                tax.scatter(unique[:, permuteIdx], s=area2, linewidths=0.6, facecolors='#B8B8B8',
                            edgecolors='#B8B8B8')
                tax.boundary(axes_colors={'l': '#D3D3D3', 'r': "#D3D3D3", 'b': "#D3D3D3"}, linewidth=0.6, zorder=-1)

                """
                plot the average phenotype
                """

                midPoint = [10 / float(3), 10 / float(3), 10 / float(3)]

                P0 = calcPointToPlot(averagePhenotype[0], 0)
                P1 = calcPointToPlot(averagePhenotype[2], 2)
                Pw = calcPointToPlot(averagePhenotype[1], 1)

                tax.line(P1,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")
                tax.line(P0,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")
                tax.line(Pw,midPoint, linewidth=3, color='#787878',zorder = -1, linestyle="solid")


            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            """
            add separator lines 
            """

            if (jx+1) %len(adultTArr) == 0 and not jx == (len(rowVec))-1:
                paramVLine = T +1.5
                ax.vlines([paramVLine], 0, 1, transform=ax.get_xaxis_transform(),color='grey', lw=1,clip_on=False)


            if (ix + 1) % len(levelsAutoCorrToPlot) == 1 and not (ix == 0):
                ax.plot([-0, 1], [1.05, 1.05], color='grey', lw=1, ls='solid', transform=ax.transAxes, clip_on=False)


            if ix == 0:
                plt.title("%s" % adultT, fontsize=25)
            else:
                ax.get_xaxis().set_visible(False)

            """
            setting the cue reliability labels
            """
            if jx == len(rowVec) - 1 and (autoCorrCurr == levelsAutoCorrToPlot[1]):
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=25)
                ax.yaxis.set_label_position("right")

            """
            setting the correct autocorrlabels
            """
            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(rowVec) * len(levelsAutoCorrToPlot)))

                currPlot = ix * len(rowVec) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(rowVec) * len(levelsAutoCorrToPlot)), len(rowVec))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(rowVec) * len(levelsAutoCorrToPlot)), len(rowVec))][0]

                if len(levelsAutoCorrToPlot) % 2 == 1:  # uneven adult T array
                    if currentSubset == ((len(levelsAutoCorrToPlot) - 1) / 2):
                        plt.ylabel(str(autoCorrCurr), labelpad=30, rotation='vertical', fontsize=25)
                        ax.yaxis.set_label_position("left")
                else:  # even
                    if (currentSubset + 1) == ((len(levelsAutoCorrToPlot)) / 2):
                        plt.ylabel(str(autoCorrCurr), labelpad=30, rotation='vertical', fontsize=25)
                        ax.yaxis.set_label_position("left")
                        ax.yaxis.set_label_coords(1.03, 0)


            if jx == len(rowVec) / 2 and ix == len(columnVec) / 2:
                fontsize = 13
                tax.right_corner_label("P0", fontsize=fontsize, position = (1, -0.075, 0))
                tax.top_corner_label("W", fontsize=fontsize, offset = 0.15)
                tax.left_corner_label("P1", fontsize=fontsize,position = (0.075, -0.075, 0))
                tax._redraw_labels()

            ix += 1
        jx += 1


    fig.text(0.05, 0.45, 'autocorrelation', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.97, 0.45, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.875

    plt.subplots_adjust(left = 0.1,wspace=0.1, hspace=0.1, bottom=0.1, top=0.8)


    fig.text(0.5, 0.93, title, fontsize=25, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(nameArg)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    nameArg2 = ['symmetric (E0)', 'asymmetric (E0)','asymmetric (E1)']

    for figCoord, adultT in zip(figVals, nameArg2):

        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.064
            else:
                figCoordF = figCoord - 0.05
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.065
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.5
            else:
                figCoordF = figCoord - 0.06
        fig.text(figCoordF, autoCorrCoord, '%s\n\n\nadult life span' % adultT, fontsize=25, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%sTernaryMercedes%s.png' % (
                         studyArg, adoptionType, lag, safeStr, False, levelsAutoCorrToPlot)),bbox_inches='tight',
        dpi=1000)

    plt.close()


def ternaryPlotOverviewJitterMerge(cueValidityArr, T, adultTArr, env, autoCorrDict,
                        twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                        studyArg, levelsAutoCorrToPlot, nameArg):

    permuteIdx = [0, 2, 1]

    rowVec = []
    for currX in nameArg:
        for currY in adultTArr:
            rowVec.append((currX, currY))

    columnVec = []
    for currX in levelsAutoCorrToPlot:
        for currY in cueValidityArr:
            columnVec.append((currX,currY))
    # prepare the x-axis values
    fig, axes = plt.subplots(len(columnVec), len(rowVec), sharex=True, sharey=True)
    fig.set_size_inches(20, 20)
    ax_list = fig.axes


    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)

    jx = 0
    for symmArg, adultT in rowVec:  # one column per adultT
        ix = 0
        for autoCorrCurr,cueVal in columnVec:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect="equal")
            plt.sca(ax)

            autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(symmArg))

            # get the data for this adult T and a specific cue reliability value
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            if "E" in symmArg:
                env = symmArg[-2]
            else:
                env = 0
            fileName = "maturePhenotypes_%s.p" % (env)
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                maturePhenotypesDict = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                maturePhenotypesDict.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: maturePhenotypesDict[(key, cueVal)] for key, val in autoCorrDict_sorted}
            # cueValDict contains a tuple now; first part are the mature phenotypes and second the cue prob

            if not "low" in levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - autoCorrCurr))]
                autoCorr = np.array(autoCorrVal)[idx][0]
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorr]
                numAgents = maturePhenotypesCurr.shape[0]

                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                # area = area_calc(uniqueCounts / float(numAgents), 150)
                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                    maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))

                    area2 = np.array(uniqueFrac) * float(105)


                else:
                    area2 = (uniqueCounts / float(numAgents)) * 105


                toPlot = np.copy(np.array(unique[:, permuteIdx]))

                tax.scatter(toPlot, s=area2, linewidths=0.6, facecolors='black',
                           edgecolors='black')
                tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]


                    # merge data for these autoCorr values!
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorrValSubset[0]]
                    for autoCorr in autoCorrValSubset[1:]:
                        """
                        continue here; need to merge all the results
                        """
                        maturePhenotypesCurrNext, cueProbabilitiesNext = cueValDict[autoCorr]
                        maturePhenotypesCurr = np.concatenate((maturePhenotypesCurr, maturePhenotypesCurrNext))
                        cueProbabilities = np.concatenate((cueProbabilities, cueProbabilitiesNext))
                    cueValDictSubset[levl] = (maturePhenotypesCurr, cueProbabilities)

                autoCorr = autoCorrCurr
                maturePhenotypesCurr, cueProbabilities = cueValDictSubset[autoCorr]
                numAgents = maturePhenotypesCurr.shape[0]
                tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                # area = area_calc(uniqueCounts / float(numAgents), 150)
                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                    maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))

                    area2 = np.array(uniqueFrac) * float(50)
                else:
                    area2 = (uniqueCounts / float(numAgents)) * 50

                tax.scatter(unique[:, permuteIdx], s=area2, linewidths=0.6, facecolors='black',
                            edgecolors='black')
                tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            # if ix == len(cueValidityArr) - 1 and jx == len(rowVec) / 2:
            #     legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), title='autocorrelation',
            #                        ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=15)
            #
            #     for i in range(len(legend.legendHandles)):
            #         legend.legendHandles[i]._sizes = [100]
            #     plt.setp(legend.get_title(), fontsize='20')


            """
            add separator lines 
            """

            if (jx+1) %len(adultTArr) == 0 and not jx == (len(rowVec))-1:
                paramVLine = T +1.5
                ax.vlines([paramVLine], 0, 1, transform=ax.get_xaxis_transform(),color='grey', lw=1,clip_on=False)


            if (ix + 1) % len(levelsAutoCorrToPlot) == 1 and not (ix == 0):
                ax.plot([-0, 1], [1.05, 1.05], color='grey', lw=1, ls='solid', transform=ax.transAxes, clip_on=False)


            if ix == 0:
                plt.title("%s" % adultT, fontsize=20)
            else:
                ax.get_xaxis().set_visible(False)

            """
            setting the cue reliability labels
            """
            if jx == len(rowVec) - 1 and (autoCorrCurr == levelsAutoCorrToPlot[1]):
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            """
            setting the correct autocorrlabels
            """
            if jx == 0:
                numPlots = len(ax_list)
                saveIdx = np.arange(0, numPlots,
                                    int(len(rowVec) * len(levelsAutoCorrToPlot)))

                currPlot = ix * len(rowVec) + jx

                currentSubset = \
                    [list(np.arange(j, j + (len(rowVec) * len(levelsAutoCorrToPlot)), len(rowVec))).index(currPlot) for
                     i, j in
                     enumerate(saveIdx) if
                     currPlot in np.arange(j, j + (len(rowVec) * len(levelsAutoCorrToPlot)), len(rowVec))][0]

                if len(levelsAutoCorrToPlot) % 2 == 1:  # uneven adult T array
                    if currentSubset == ((len(levelsAutoCorrToPlot) - 1) / 2):
                        plt.ylabel(str(autoCorrCurr), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("left")
                else:  # even
                    if (currentSubset + 1) == ((len(levelsAutoCorrToPlot)) / 2):
                        plt.ylabel(str(autoCorrCurr), labelpad=30, rotation='vertical', fontsize=20)
                        ax.yaxis.set_label_position("left")
                        ax.yaxis.set_label_coords(1.03, 0)


            if jx == len(rowVec) / 2 and ix == len(columnVec) / 2:
                fontsize = 13
                tax.right_corner_label("P0", fontsize=fontsize, position = (0.725, 0.15, 0))
                tax.top_corner_label("W", fontsize=fontsize, offset = -0.15)
                tax.left_corner_label("P1", fontsize=fontsize,position = (0.125, 0.15, 0))
                tax._redraw_labels()

            ix += 1
        jx += 1


    fig.text(0.05, 0.45, 'autocorrelation', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.97, 0.45, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.85

    plt.subplots_adjust(left = 0.1,wspace=0.1, hspace=0.1, bottom=0.1, top=0.8)


    fig.text(0.5, 0.93, 'switch probabilities', fontsize=20, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(nameArg)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    nameArg2 = ['symmetric (E0)', 'asymmetric (E0)','asymmetric (E1)']

    for figCoord, adultT in zip(figVals, nameArg2):

        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.064
            else:
                figCoordF = figCoord - 0.05
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.065
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.5
            else:
                figCoordF = figCoord - 0.06
        fig.text(figCoordF, autoCorrCoord, '%s\n\n\nadult life span' % adultT, fontsize=20, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%sTernaryOverviewTotalJitter%s.png' % (
                         studyArg, adoptionType, lag, safeStr, False, levelsAutoCorrToPlot)),
        dpi=1000)

    plt.close()



def ternaryPlotOverview(cueValidityArr, T, adultTArr, env, autoCorrDict,
                        twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                        studyArg, levelsAutoCorrToPlot):
    permuteIdx = [0, 2, 1]
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)
    fig, axes = plt.subplots(len(cueValidityArr), len(adultTArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for adultT in adultTArr:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(adultTArr) + jx]
            #ax.set(aspect=1)
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = "maturePhenotypes_%s.p" % (env)
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                maturePhenotypesDict = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                maturePhenotypesDict.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: maturePhenotypesDict[(key, cueVal)] for key, val in autoCorrDict_sorted}
            # cueValDict contains a tuple now; first part are the mature phenotypes and second the cue prob

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]

                for idx, autoCorr in enumerate(autoCorrValSubset):
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorr]
                    numAgents = maturePhenotypesCurr.shape[0]
                    tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                    # now need to work on the scaling of points

                    unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                    # area = area_calc(uniqueCounts / float(numAgents), 150)
                    if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                        uniqueFrac = []
                        for matPhen in unique:
                            probIdx = np.where(
                                (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                        maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                        maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                            uniqueFrac.append(sum(cueProbabilities[probIdx]))

                        area2 = np.array(uniqueFrac) * float(600)


                    else:
                        area2 = (uniqueCounts / float(numAgents)) * 600
                    # this one would be scalling according to area
                    colorShades = [0.2, 0.5, 1]
                    colorTuple = [0, 0, 0, 1]
                    if idx == 0:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, marker=MarkerStyle(marker='o', fillstyle='left'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    elif idx == 1:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, marker=MarkerStyle(marker='o', fillstyle='right'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    else:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, linewidths=0.6, facecolors='none',
                                    edgecolors=colorTuple,
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    if levl == 'moderate':
                        if len(autoCorrValSubset)==3:
                            test = []
                            test.append(autoCorrValSubset[0])
                            test.append(autoCorrValSubset[2])
                            autoCorrValSubset = test
                    # merge data for these autoCorr values!
                    maturePhenotypesCurr, cueProbabilities = cueValDict[autoCorrValSubset[0]]
                    for autoCorr in autoCorrValSubset[1:]:
                        """
                        continue here; need to merge all the results
                        """
                        maturePhenotypesCurrNext, cueProbabilitiesNext = cueValDict[autoCorr]
                        maturePhenotypesCurr = np.concatenate((maturePhenotypesCurr, maturePhenotypesCurrNext))
                        cueProbabilities = np.concatenate((cueProbabilities, cueProbabilitiesNext))
                    cueValDictSubset[levl] = (maturePhenotypesCurr, cueProbabilities)

                colorArr = [0.1, 0.5, 0.9]
                for idx, autoCorr in enumerate(loopArrayLevl):
                    maturePhenotypesCurr, cueProbabilities = cueValDictSubset[autoCorr]
                    numAgents = maturePhenotypesCurr.shape[0]
                    tax = ternary.TernaryAxesSubplot(ax=ax, scale=T)
                    # now need to work on the scaling of points

                    unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)
                    # area = area_calc(uniqueCounts / float(numAgents), 150)
                    if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                        uniqueFrac = []
                        for matPhen in unique:
                            probIdx = np.where(
                                (maturePhenotypesCurr[:, 0] == matPhen[0]) & (
                                        maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                        maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                            uniqueFrac.append(sum(cueProbabilities[probIdx]))

                        area2 = np.array(uniqueFrac) * float(200)
                    else:
                        area2 = (uniqueCounts / float(numAgents)) * 200
                    # this one would be scalling according to area
                    # colVal =1-colorArr[idx]
                    colorShades = [0.2, 0.5, 1]
                    colorTuple = [0, 0, 0, 1]
                    if idx == 0:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, marker=MarkerStyle(marker='o', fillstyle='left'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    elif idx == 1:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, marker=MarkerStyle(marker='o', fillstyle='right'),
                                    facecolors=colorTuple, edgecolors='none',
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)
                    else:
                        colorTuple[3] = colorShades[idx]
                        tax.scatter(unique[:, permuteIdx], s=area2, linewidths=0.6, facecolors='none',
                                    edgecolors=colorTuple,
                                    label="%s" % (autoCorr))
                        tax.boundary(axes_colors={'l': 'grey', 'r': "grey", 'b': "grey"}, linewidth=0.8, zorder=-1)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Remove default Matplotlib Axe
            tax.clear_matplotlib_ticks()

            if ix == len(cueValidityArr) - 1 and jx == len(adultTArr) / 2:
                legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), title='autocorrelation',
                                   ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=20)

                for i in range(len(legend.legendHandles)):
                    legend.legendHandles[i]._sizes = [100]
                plt.setp(legend.get_title(), fontsize='20')
            if ix == 0:
                plt.title("%s" % adultT, fontsize=20)
            else:
                ax.get_xaxis().set_visible(False)

            if jx == len(adultTArr) - 1:
                plt.ylabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            if jx == len(adultTArr) / 2 and ix == len(cueValidityArr) / 2:
                fontsize = 20
                tax.right_corner_label("P0", fontsize=fontsize, offset=-0.15)
                tax.top_corner_label("wait time", fontsize=fontsize)
                tax.left_corner_label("P1", fontsize=fontsize, offset=-0.15)
                tax._redraw_labels()

            ix += 1
        jx += 1
    plt.suptitle('adult life span', fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.2, top=0.9)
    fig.text(0.98, 0.58, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sTernaryOverviewTotal%s.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False, levelsAutoCorrToPlot)),
        dpi=3000)

    plt.close()


def plotPlasticityCurvesOverview(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                 twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag,
                                 endOfExposure,
                                 studyArg, levelsAutoCorrToPlot):
    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)


    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)
    fig, axes = plt.subplots(len(cueValidityArr), 2, sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for arg in ['relative', 'absolute']:  # one plot for relative, one plot for absolute phenotypic distance
        ix = 0
        for cueVal in cueValidityArr:
            ax = ax_list[ix * 2 + jx]
            plt.sca(ax)
            for i, adultT in enumerate(adultTArr):  # one line per adult T
                # linestyle depends on current adult T value
                linestyle = linestyle_tuple[i][1]
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                if os.path.exists(filePath1):
                    distanceDict.update(pickle.load(open(filePath1, 'rb')))

                # for he current cueVal load the distancedictionaries
                cueValDict = {val: distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted}

                if levelsAutoCorrToPlot:
                    # the next line find the indices of the closest autocorrelation values that match the user input
                    idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                           levelsAutoCorrToPlot]
                    autoCorrValSubset = np.array(autoCorrVal)[idx]
                    [plt.plot(tValues, cueValDict[autoCorr], linestyle=linestyle, linewidth=2, markersize=8,
                              marker='o', color=str(1 - autoCorr - 0.1), markerfacecolor=str(1 - autoCorr - 0.1),
                              label="%s & %s" % (autoCorr, adultT)) for autoCorr in autoCorrValSubset]

                else:
                    """
                    in case that the user did not specify values to pick, compute an average
                    - first need to calculate cutoff points
                    """
                    extremeIDX = np.floor(len(autoCorrVal) / float(3))
                    midIDX = np.ceil(len(autoCorrVal) / float(3))
                    loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                    loopArrayLevl = ['low', 'moderate', 'high']

                    cueValDictSubset = {}
                    for idx in range(len(loopArrayIDX)):
                        levl = loopArrayLevl[idx]

                        if idx == 0:
                            startIdx = int(idx)
                        else:
                            startIdx = int(endIdx)
                        endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                        autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                        plastcityVal = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                        cueValDictSubset[levl] = plastcityVal

                    colorArr = [0.1, 0.5, 0.9]
                    [plt.plot(tValues, cueValDictSubset[autoCorr], linestyle=linestyle, linewidth=2, markersize=8,
                              marker='o', color=str(1 - colorArr[idx] - 0.1),
                              markerfacecolor=str(1 - colorArr[idx] - 0.1),
                              label="%s & %s" % (autoCorr[0].upper(), adultT)) for idx, autoCorr in
                     enumerate(loopArrayLevl)]

            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """
            plt.plot(tValues, [0] * T, linestyle='dashed', linewidth=1, color='grey')
            plt.plot(tValues, [1] * T, linestyle='dashed', linewidth=1, color='grey')

            if ix == len(cueValidityArr) - 1 and jx == 0:
                legend = ax.legend(loc='upper center', bbox_to_anchor=(1, -0.2),
                                   title='autocorrelation & adult lifespan',
                                   ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=20)
                plt.setp(legend.get_title(), fontsize='20')
            if ix == 0:
                plt.title("%s distance" % arg, fontsize=20)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
            plt.xticks([])

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                plt.xlabel('ontogeny', fontsize=20, labelpad=20)

            if jx == 1:
                ax.yaxis.set_label_position("right")
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)

            ix += 1
        jx += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.2, top=0.95)
    fig.text(0.98, 0.58, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.08, 0.32, 'phenotypic distance', fontsize=20, ha='center', va='center', rotation='vertical')
    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityOverviewTotal%s.png' % (
                         studyArg, adoptionType, lag, safeStr, env, False, levelsAutoCorrToPlot)),
        dpi=1000)

    plt.close()


def plotPlasticityCurves(cueValidityArr, T, adultTArr, env, autoCorrDict,
                         twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                         studyArg, coarseArg, levelsAutoCorrToPlot):
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)

    for arg in ['relative', 'absolute']:  # one plot for relative, one plot for absolute phenotypic distance
        for adultT in adultTArr:  # one plot for adult T
            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                arg, studyArg, adoptionType, lag, endOfExposure, env)
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):
                distanceDict = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                distanceDict.update(pickle.load(open(filePath1, 'rb')))

            fig, axes = plt.subplots(len(cueValidityArr), 1, sharex=True, sharey=True)
            fig.set_size_inches(16, 16)
            ax_list = fig.axes

            ix = 0
            for cueVal in cueValidityArr:  # one row per cue validity
                # for he current cueVal load the distancedictionaries
                ax = ax_list[ix]

                plt.sca(ax)
                cueValDict = {val: distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted}

                if coarseArg:

                    if levelsAutoCorrToPlot:
                        # the next line find the indices of the closest autocorrelation values that match the user input
                        idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                               levelsAutoCorrToPlot]
                        autoCorrValSubset = np.array(autoCorrVal)[idx]
                        [plt.plot(tValues, cueValDict[autoCorr], linestyle='solid', linewidth=2, markersize=8,
                                  marker='o', color=str(1 - autoCorr - 0.1), markerfacecolor=str(1 - autoCorr - 0.1),
                                  label=str(autoCorr)) for autoCorr in autoCorrValSubset]

                    else:
                        """
                        in case that the user did not specify values to pick, compute an average
                        - first need to calculate cutoff points
                        """
                        extremeIDX = np.floor(len(autoCorrVal) / float(3))
                        midIDX = np.ceil(len(autoCorrVal) / float(3))
                        loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                        loopArrayLevl = ['low', 'moderate', 'high']

                        cueValDictSubset = {}
                        for idx in range(len(loopArrayIDX)):
                            cutoffPoint = loopArrayIDX[idx]
                            levl = loopArrayLevl[idx]

                            if idx == 0:
                                startIdx = int(idx)
                            else:
                                startIdx = int(endIdx)
                            endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                            autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                            plastcityVal = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                            cueValDictSubset[levl] = plastcityVal

                        colorArr = [0.1, 0.5, 0.9]
                        [plt.plot(tValues, cueValDictSubset[autoCorr], linestyle='solid', linewidth=2, markersize=8,
                                  marker='o', color=str(1 - colorArr[idx] - 0.1),
                                  markerfacecolor=str(1 - colorArr[idx] - 0.1),
                                  label=str(autoCorr)) for idx, autoCorr in enumerate(loopArrayLevl)]


                else:
                    [plt.plot(tValues, cueValDict[autoCorr], linestyle='solid', linewidth=2, markersize=8,
                              marker='o', color=str(1 - autoCorr - 0.1), markerfacecolor=str(1 - autoCorr - 0.1),
                              label=str(autoCorr)) for autoCorr in autoCorrVal]

                """
                plot two parallels to the x-axis to highlight the 0 and 1 mark
                """
                plt.plot(tValues, [0] * T, linestyle='dashed', linewidth=1, color='grey')
                plt.plot(tValues, [1] * T, linestyle='dashed', linewidth=1, color='grey')

                if ix == 1:
                    legend = ax.legend(loc='center left', bbox_to_anchor=(-0.65, 0.5), title='autocorrelation',
                                       ncol=1, fancybox=True, shadow=False, fontsize=20)
                    plt.setp(legend.get_title(), fontsize='20')

                # stuff to amke the plot look pretty
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                plt.ylim(-0.05, 1.05)
                plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
                plt.xticks([])

                if ix == len(cueValidityArr) - 1:
                    ax.spines['left'].set_visible(True)
                    ax.spines['bottom'].set_visible(True)

                    plt.xlabel('ontogeny', fontsize=20, labelpad=20)

                ax.yaxis.set_label_position("right")
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)

                ix += 1

            fig.text(0.81, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                     transform=ax.transAxes, rotation='vertical')
            fig.text(0.25, 0.22, 'phenotypic distance', fontsize=20, ha='center', va='center', rotation='vertical')
            if endOfExposure:
                safeStr = "EndOfExposure"
            else:
                safeStr = "EndOfOntogeny"
            plt.savefig(
                os.path.join(twinResultsAggregatedPath,
                             '%s_%s_%s_%s_%s%sPlasticityOverview%s%s_%sDistAdultT%s.png' % (
                                 studyArg, adoptionType, lag, safeStr, env, False, coarseArg, levelsAutoCorrToPlot, arg,
                                 adultT)),
                dpi=1000)

            plt.close()


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


def performSimulationAnalysis(argument, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType, endOfExposure,
                              adultT, env):
    # first step create the directory for the results
    if not os.path.exists(twinResultsPath):
        os.makedirs(twinResultsPath)

    if argument == "ExperimentalTwinstudy":
        """
        this will implement a form of the twin study that an be considered more artifical
        it will be comparable to experimental manipulations done in lab environments
        it will manipulate the onset and amount of a
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}

        for markov_chain in markovProbabilities:
            pE0E0, pE1E1 = markov_chain
            for cueReliability in cueValidityC0E0Arr:
                print "currently working with pE0E0: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)

                startTime = time.time()

                """
                need to extract the correct prior from the markov chain
                """

                resultsMat, cueProbabilities = runExperimentalAdoptionExperiment(T, numAgents, env, pE0E0, pE1E1,
                                                                                 cueReliability,
                                                                                 resultsPath,
                                                                                 argumentR, argumentP, lag,
                                                                                 adoptionType, endOfExposure)
                elapsedTime = time.time() - startTime
                print "Elapsed time: " + str(elapsedTime)

                # normalize resultsmat
                pickle.dump(resultsMat,
                            open(os.path.join(twinResultsPath,
                                              "resultsMat_%s%s_%s_%s.p" % (pE0E0, pE1E1, env, cueReliability)),
                                 "wb"))
                absoluteDistance, relativeDistance, _, _ = postProcessResultsMat(resultsMat, T + lag - 1, endOfExposure,
                                                                                 lag, cueProbabilities)
                absoluteDistanceDict[((pE0E0, pE1E1), cueReliability)] = absoluteDistance
                relativeDistanceDict[((pE0E0, pE1E1), cueReliability)] = relativeDistance

        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        # plasticityAreaGradient(priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath)

    elif argument == "Twinstudy":
        """
        This will calculate the results from the twin studies
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}
        absoluteDistanceDictVar = {}
        relativeDistanceDictVar = {}

        absoluteDistanceDictTemp = {}
        relativeDistanceDictTemp = {}

        beliefDict = {}

        for markov_chain in markovProbabilities:
            pE0E0, pE1E1 = markov_chain
            for cueReliability in cueValidityC0E0Arr:
                print "currently working with pE0E0: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)

                # calculate the stationary distribution
                resultsMat, resultsMatBeliefs, resultsMatTempPhenotypes, cueProabailities = runAdoptionExperiment(T,
                                                                                                                  numAgents,
                                                                                                                  env,
                                                                                                                  pE0E0,
                                                                                                                  pE1E1,
                                                                                                                  cueReliability,
                                                                                                                  resultsPath,
                                                                                                                  argumentR,
                                                                                                                  argumentP,
                                                                                                                  adoptionType)

                # normalize resultsmat
                pickle.dump(resultsMat,
                            open(os.path.join(twinResultsPath,
                                              "resultsMat_%s%s_%s_%s.p" % (pE0E0, pE1E1, env, cueReliability)),
                                 "wb"))
                pickle.dump(cueReliability,open(os.path.join(twinResultsPath,
                                              "cueProbability_%s%s_%s_%s.p" % (pE0E0, pE1E1, env, cueReliability)),
                                 "wb") )

                absoluteDistance, relativeDistance, absoluteDistanceVariance, relativeDistanceVariance = postProcessResultsMat(
                    resultsMat, T, endOfExposure, lag, cueProabailities)

                absoluteDistanceDict[((pE0E0, pE1E1), cueReliability)] = absoluteDistance
                relativeDistanceDict[((pE0E0, pE1E1), cueReliability)] = relativeDistance

                absoluteDistanceDictVar[((pE0E0, pE1E1), cueReliability)] = absoluteDistanceVariance
                relativeDistanceDictVar[((pE0E0, pE1E1), cueReliability)] = relativeDistanceVariance

                beliefDict[((pE0E0, pE1E1), cueReliability)] = resultsMatBeliefs

                # do the same for the temporary phenotypes
                pickle.dump(resultsMatTempPhenotypes,
                            open(os.path.join(twinResultsPath,
                                              "resultsMatTempPhenotypes_%s%s_%s_%s.p" % (
                                                  pE0E0, pE1E1, env, cueReliability)),
                                 "wb"))
                absoluteDistanceTemp, relativeDistanceTemp, _, _ = postProcessResultsMat(resultsMatTempPhenotypes, T,
                                                                                         True, 1, cueProabailities)
                absoluteDistanceDictTemp[((pE0E0, pE1E1), cueReliability)] = absoluteDistanceTemp
                relativeDistanceDictTemp[((pE0E0, pE1E1), cueReliability)] = relativeDistanceTemp

        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        pickle.dump(absoluteDistanceDictVar,
                    open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDictVar,
                    open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        pickle.dump(absoluteDistanceDictTemp,
                    open(os.path.join(twinResultsPath, 'absoluteDistanceDictTemp%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDictTemp,
                    open(os.path.join(twinResultsPath, 'relativeDistanceDictTemp%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        pickle.dump(beliefDict, open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        # plasticityAreaGradient(priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath)

    elif argument == "MaturePhenotypes":
        maturePhenotypes = {}
        maturePhenotypesTemp = {}
        for markov_chain in markovProbabilities:
            pE0E0, pE1E1 = markov_chain
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on pE0E0: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)
                maturePheno, maturePhenotypesTemporal, _, cueProbabilities = \
                    runTwinStudiesParallel(0, numAgents, env, pE0E0, pE1E1,
                                           cueReliability, False, T,
                                           resultsPath, argumentR, argumentP,
                                           None, [])

                maturePhenotypes[((pE0E0, pE1E1), cueReliability)] = maturePheno, cueProbabilities
                maturePhenotypesTemp[((pE0E0, pE1E1), cueReliability)] = maturePhenotypesTemp, cueProbabilities

        pickle.dump(maturePhenotypes, open(os.path.join(twinResultsPath, "maturePhenotypes_%s.p" % (env)), "wb"))
        pickle.dump(maturePhenotypes, open(os.path.join(twinResultsPath, "maturePhenotypesTemp_%s.p" % (env)), "wb"))

    elif argument == "RankOrderStability":
        """
        for now this argument does not work with the changing environments because we have not agreed what the trait 
        of interest is when the environment fluctuates; best average fit with the environment across adulthood? 
        
        but is this really a single trait? I would leave it be for now 
        """

        rankOrderStabilityRaw = {}
        rankOrderStabilityRanks = {}
        rankOrderStabilityRanksNorm = {}

        # rankOrderStabilityRaw = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRaw.p"), "rb"))

        for markov_chain in markovProbabilities:
            pE0E0, pE1E1 = markov_chain
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on pE0E0: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)
                rankOrderStabilityRaw[((pE0E0, pE1E1), cueReliability)] = \
                    runTwinStudiesParallel(0, numAgents, env, pE0E0, pE1E1,
                                           cueReliability, False, T,
                                           resultsPath, argumentR,
                                           argumentP,
                                           None, [])[1]

                current = rankOrderStabilityRaw[((pE0E0, pE1E1), cueReliability)]
                currentMat = np.zeros((current.shape[0], T))
                currentMatNorm = np.zeros((current.shape[0], T))

                tValues = np.arange(1, T + 1, 1)
                for t in tValues:
                    possibleRanks = sorted(list(set(current[:, 1, t - 1])), reverse=True)
                    currentMat[:, t - 1] = [possibleRanks.index(a) + 1 for a in current[:, 1,
                                                                                t - 1]]  # the plus one makes sure that we don't have zero ranks, which are computationally inconvenient

                rankOrderStabilityRanks[((pE0E0, pE1E1), cueReliability)] = currentMat
                rankOrderStabilityRanksNorm[((pE0E0, pE1E1), cueReliability)] = currentMatNorm

        pickle.dump(rankOrderStabilityRaw,
                    open(os.path.join(twinResultsPath, "rankOrderStabilityRaw_%s.p" % (env)), "wb"))
        pickle.dump(rankOrderStabilityRanks,
                    open(os.path.join(twinResultsPath, "rankOrderStabilityRanks_%s.p" % (env)), "wb"))


    elif argument == "MaturePhenotypesTwoPatches":

        maturePhenotypes = {}
        for markov_chain in markovProbabilities:
            pE0E0, pE1E1 = markov_chain
            for cueReliability in cueValidityC0E0Arr:
                print "Currently working on pE0E0: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)
                prior = calcStationaryDist(markov_chain)
                agentsEnv1 = int((1 - prior) * numAgents)
                agentsEnv0 = numAgents - agentsEnv1
                resultsEnv1 = runTwinStudiesParallel(0, agentsEnv1, 1, pE0E0, pE1E1,
                                                     cueReliability, False, T,
                                                     resultsPath, argumentR, argumentP, None, [])[0]
                resultsEnv0 = runTwinStudiesParallel(0, agentsEnv0, 0, pE0E0, pE1E1,
                                                     cueReliability, False, T,
                                                     resultsPath, argumentR, argumentP, None, [])[0]

                maturePhenotypes[((pE0E0, pE1E1), cueReliability)] = np.concatenate(
                    [resultsEnv1, resultsEnv0])  # no weighting?
        pickle.dump(maturePhenotypes,
                    open(os.path.join(twinResultsPath, "maturePhenotypesTwoPatches_%s.p" % (env)), "wb"))



    elif argument == 'FitnessDifference':
        fitnessDifference(markovProbabilities, cueValidityC0E0Arr, resultsPath, T, twinResultsPath, baselineFitness,
                          argumentR,
                          argumentP, adultT, numAgents)


    else:
        print "Wrong input argument to plotting arguments!"
        exit()


def plotSimulationStudy(argument, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType,
                        endOfExposure, varArg, env):
    tValues = np.arange(1, T + 1, 1)

    if argument == "BeliefTwinstudy":
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        # for the temporary phenotypes
        relativeDistanceDictTemp = pickle.load(
            open(os.path.join(twinResultsPath, 'relativeDistanceDictTemp%s%s%s%s_%s.p' % (
                "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        beliefDict = pickle.load(open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s_%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        plotBeliefDistances(tValues, markovProbabilities, cueValidityC0E0Arr, relativeDistanceDict, twinResultsPath,
                            argument, adoptionType, lag, endOfExposure, beliefDict,
                            relativeDistanceDictTemp, env)

    elif argument == "Twinstudy":

        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        # load the variance
        if varArg:
            absoluteDistanceDictVar = pickle.load(
                open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s_%s.p' % (
                    argument, adoptionType, lag, endOfExposure, env)), 'rb'))
            relativeDistanceDictVar = pickle.load(
                open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s_%s.p' % (
                    argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        else:
            absoluteDistanceDictVar = None
            relativeDistanceDictVar = None
        plotDistances(tValues, markovProbabilities, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, varArg, absoluteDistanceDictVar,
                      relativeDistanceDictVar, env)

    elif argument == "ExperimentalTwinstudy":
        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        plotDistances(tValues, markovProbabilities, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, False, None, None, env)

    elif argument == "MaturePhenotypes":
        print "here about to load"
        maturePhenotypesResults = pickle.load(
            open(os.path.join(twinResultsPath, "maturePhenotypes_%s.p" % (env)), "rb"))

        # maturePhenotypesTempResults = pickle.load(
        #    open(os.path.join(twinResultsPath, "maturePhenotypesTemp_%s.p" % (env)), "rb"))
        plotTriangularPlots(tValues, markovProbabilities, cueValidityC0E0Arr, maturePhenotypesResults, T,
                            twinResultsPath, env)

    elif argument == "MaturePhenotypesTwoPatches":
        maturePhenotypes = pickle.load(
            open(os.path.join(twinResultsPath, "maturePhenotypesTwoPatches_%s.p" % (env)), "rb"))
        plotTriangularPlots(tValues, markovProbabilities, cueValidityC0E0Arr, maturePhenotypes, T, twinResultsPath)

    elif argument == "FitnessDifference":
        plotFitnessDifference(markovProbabilities, cueValidityC0E0Arr, twinResultsPath)

    elif argument == "RankOrderStability":
        plotRankOrderStability(markovProbabilities, cueValidityC0E0Arr, twinResultsPath, T, ['negativeSwitches'], env)

    else:
        print "Wrong input argument to plotting arguments!"


def runPlots(markovProbabilities, cueValidityC0E0Arr, TParam, numAgents, twinResultsPath, baselineFitness, resultsPath,
             argumentR, argumentP, lagArray, adoptionType, endOfExposure, plotArgs, plotVar, performSimulation, adultT,
             startingEnv):
    for arg in plotArgs:
        print arg

        if arg == 'ExperimentalTwinstudy':
            for lag in lagArray:
                TLag = TParam - lag + 1
                T = TLag
                if performSimulation:
                    performSimulationAnalysis(arg, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath,
                                              numAgents,
                                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType,
                                              endOfExposure, adultT, startingEnv)
                plotSimulationStudy(arg, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType,
                                    endOfExposure, plotVar, startingEnv)
        elif arg == 'BeliefTwinstudy':
            T = TParam
            plotSimulationStudy(arg, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, None, adoptionType,
                                endOfExposure, plotVar, startingEnv)

        else:
            T = TParam
            if performSimulation:
                performSimulationAnalysis(arg, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                                          resultsPath, baselineFitness, argumentR, argumentP, None, adoptionType,
                                          False, adultT, startingEnv)
            plotSimulationStudy(arg, markovProbabilities, cueValidityC0E0Arr, T, twinResultsPath, None, adoptionType,
                                False, plotVar, startingEnv)


def runAggregatePlots(markovProbabilities, cueValidityArr, T, adultTArr, startEnv, autoCorrPath,
                      twinResultsAggregatedPath, dataPath, argumentR, argumentP, autoCorrArg, adoptionType, lag,
                      endOfExposure, studyArg, coarseArg, levelsAutoCorrToPlot, normalize):
    if autoCorrArg == 'absolute':
        autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 'exact_dict.p'), "rb"))
        autoCorrDictAccurate = pickle.load(open(os.path.join(autoCorrPath, 'accurate_dict.p'), "rb"))

        for key, value in autoCorrDict.items():
            if not value:
                autoCorrDict[key] = autoCorrDictAccurate[key]
    elif autoCorrArg == 'experienced':
        autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 't_dict.p'), "rb"))
    else:
        print "wrong input argument for the autocorrelation dictionary; must be absolute or experienced"
        exit()

    # fix the following two into better shape
    # adapt the new layout
    adultTArrSubset = [1,5,20] #1,5, 20,
    r = 0.5#0.5  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
    minProb = 0  # minimal probability of reaching a state for those that are displayed
    lines = True

    adulthood = False
    policyPlotReducedOverview2(r, minProb, lines, cueValidityArr, T, adultTArrSubset,[0.2, 0.5, 0.8], autoCorrDict,
                             twinResultsAggregatedPath, dataPath, argumentR, argumentP, adulthood,False)

    """the plots below show the same plots as in the main paper but are separate for symmetric and asymmetric transition 
    probabilities and instead show the separated for absolute and relative plasticity"""


    plotFitnessDifferenceOverview(cueValidityArr, T, adultTArrSubset, autoCorrDict,
                                 twinResultsAggregatedPath, dataPath, argumentR, argumentP,[0.2, 0.5, 0.8])




def runAggregatePlotsAssymetries(markovProbabilities, cueValidityArr, T, adultTArr, startEnv, autoCorrPathCurr,
                           twinResultsAggregatedPath, dataPath, argumentR, argumentP, autoCorrArg, adoptionType, lag,
                           endOfExposure, studyArg, coarseArg, normalize, mergeArr, nameArg):

    autoCorrDictTotal = {}
    for mergeArg in mergeArr:
        dataPathPrior = os.path.join(dataPath, str(mergeArg))
        autoCorrPath = os.path.join(dataPathPrior, autoCorrPathCurr)

        if autoCorrArg == 'absolute':
            autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 'exact_dict.p'), "rb"))
            autoCorrDictAccurate = pickle.load(open(os.path.join(autoCorrPath, 'accurate_dict.p'), "rb"))

            for key, value in autoCorrDict.items():
                if not value:
                    autoCorrDict[key] = autoCorrDictAccurate[key]
        elif autoCorrArg == 'experienced':
            autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 't_dict.p'), "rb"))
        else:
            print "wrong input argument for the autocorrelation dictionary; must be absolute or experienced"
            exit()
        autoCorrDictTotal[mergeArg] = autoCorrDict



    adultTArrSubset = [1, 5, 20]
    plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArrSubset, startEnv, autoCorrDictTotal,
                                   twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag,
                                   endOfExposure,
                                   studyArg, mergeArr, nameArg)




def runAggregatePlotsMerge(markovProbabilities, cueValidityArr, T, adultTArr, startEnv, autoCorrPathCurr,
                           twinResultsAggregatedPath, dataPath, argumentR, argumentP, autoCorrArg, adoptionType, lag,
                           endOfExposure, studyArg, coarseArg, normalize, mergeArr, nameArg):
    autoCorrDictTotal = {}
    for mergeArg in mergeArr:
        dataPathPrior = os.path.join(dataPath, str(mergeArg))
        autoCorrPath = os.path.join(dataPathPrior, autoCorrPathCurr)

        if autoCorrArg == 'absolute':
            autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 'exact_dict.p'), "rb"))
            autoCorrDictAccurate = pickle.load(open(os.path.join(autoCorrPath, 'accurate_dict.p'), "rb"))

            for key, value in autoCorrDict.items():
                if not value:
                    autoCorrDict[key] = autoCorrDictAccurate[key]
        elif autoCorrArg == 'experienced':
            autoCorrDict = pickle.load(open(os.path.join(autoCorrPath, 't_dict.p'), "rb"))
        else:
            print "wrong input argument for the autocorrelation dictionary; must be absolute or experienced"
            exit()
        autoCorrDictTotal[mergeArg] = autoCorrDict

    # fix the following two into better shape
    # adapt the new layout
    adultTArrSubset = [1, 5, 20] #1, 5, 20
    levelsAutoCorrToPlotArg = None

    r = 0.5  # 0.5  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
    minProb = 0  # minimal probability of reaching a state for those that are displayed
    lines = True

    adulthood = False

    policyPlotReducedMergeBW(0.55, minProb, lines, cueValidityArr, T, 20, [0.2, 0.5, 0.8], autoCorrDictTotal,
                           twinResultsAggregatedPath, dataPath, argumentR, argumentP, True, False,
                           ['symmetric', 'asymmetric (E0)'])





    policyPlotReducedMerge(r, minProb, lines, cueValidityArr, T, 5, [0.2, 0.5, 0.8], autoCorrDictTotal,
                           twinResultsAggregatedPath, dataPath, argumentR, argumentP, adulthood, False,
                           ['symmetric', 'asymmetric (E0)'])


    ternaryMercedesMerge(cueValidityArr, T, adultTArrSubset, startEnv, autoCorrDictTotal,
                        twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
                        studyArg,[0.2, 0.5, 0.8] , ['symmetric', 'asymmetric (E0)','asymmetric (E1)'],nameArg)


    plotFitnessDifferenceOverviewMerge(cueValidityArr, T, adultTArrSubset, autoCorrDictTotal,
                                 twinResultsAggregatedPath, dataPath, argumentR, argumentP,[0.2, 0.5, 0.8], mergeArr)

    plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArrSubset, startEnv, autoCorrDictTotal,
                                   twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag,
                                   endOfExposure,studyArg, mergeArr, nameArg)


