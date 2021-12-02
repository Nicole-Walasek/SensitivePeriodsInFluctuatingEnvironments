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
from cmocean import cm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sympy.utilities.iterables import multiset_permutations
from decimal import Decimal as D
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


def updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker):
    optDecisions = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                      subDF['yw'] == phenotypeTracker[idx, 2])
                              & (subDF['Identifier'] == identTracker[idx])]['cStar'].item() for idx in
                    simValues]
    # additionally keep track of the posterior belief
    posBelief = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0'] == phenotypeTracker[idx, 0]) & (subDF['y1'] == phenotypeTracker[idx, 1]) & (
                                   subDF['yw'] == phenotypeTracker[idx, 2])
                           & (subDF['Identifier'] == identTracker[idx])]['pE1'].item() for idx in
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


def updateIdentiyTracker(identTracker, cues):
    newIdentTracker = [2 * i if c == 0 else (2 * i + 1) for i, c in zip(identTracker, cues)]

    return newIdentTracker


def simulateExperimentalTwins(tAdopt, twinNum, env, cueReliability, lag, T, adoptionType, endOfExposure):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability
    :return: phenotypic distance between pairs of twins
    """
    T = T +lag -1
    tValues = np.arange(1, tAdopt, 1)
    if env == 1:
        pC1Start = cueReliability[1]  # take the very first cue reliability
    else:
        pC1Start = 1 - cueReliability[1]
    pC0Start = 1 - pC1Start

    cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

    # need to reverse the last update
    if adoptionType == "yoked":
        oppositeCues = 1 - cues

    elif adoptionType == "oppositePatch":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
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

    identTracker = cues
    if len(tValues) != 0:
        identTrackerDoppel = cues
    else:
        identTrackerDoppel = oppositeCues
    simValues = np.arange(0, twinNum, 1)

    for t in tValues:
        # now we have to recompute this for every timestep
        if env == 1:
            pC1Start = cueReliability[t]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[t]

        pC0Start = 1 - pC1Start

        np.random.seed()
        # print "currently simulating time step: " + str(t)
        subDF = policy[policy['time'] == t].reset_index(drop=True)
        # next generate 10000 new cues
        # generate 10000 optimal decisions

        # probably need an identity tracker for the new policies
        cues = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])

        cues = np.array(cues)
        phenotypeTracker, __ = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker)
        # update identity tracker for new cues
        identTracker = updateIdentiyTracker(identTracker, cues)
        if t != tValues[-1]:
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cues)
        else:
            # last step is where we get yoked opposite cues
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
        # update cue tracker
        cueTracker[:, 0] += (1 - cues)
        cueTracker[:, 1] += cues

    # post adoption period
    # continue here
    originalTwin = np.copy(phenotypeTracker)
    doppelgaenger = np.copy(phenotypeTracker)

    restPeriod = np.arange(tAdopt, tAdopt + lag, 1)

    # setting up the matrix for the yoked opposite cues
    cueTrackerDoppel = np.copy(cueTracker)

    cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
    cueTrackerDoppel[:, 1] += -cues + oppositeCues

    for t2 in restPeriod:
        if env == 1:
            pC1Start = cueReliability[t2]  # take the very first cue reliability
        else:
            pC1Start = 1 - cueReliability[t2]

        pC0Start = 1 - pC1Start

        np.random.seed()
        subDF = policy[policy['time'] == t2].reset_index(drop=True)
        # probably need an identity tracker for the new policies
        cuesOriginal = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
        cuesOriginal = np.array(cuesOriginal)

        if adoptionType == "yoked":
            oppositeCues = 1 - cuesOriginal
        elif adoptionType == "oppositePatch":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
            oppositeCues = np.array(oppositeCues)
        else:  # adoptionType = deprivation
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)
        # update the phenotypes of the twins
        originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)
        identTracker = updateIdentiyTracker(identTracker, cuesOriginal)

        doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel, identTrackerDoppel)

        if t2 != restPeriod[-1]:
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)
        else:
            # last step is where we get yoked opposite cues
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cuesOriginal)

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

            if env == 1:
                pC1Start = cueReliability[t3]  # take the very first cue reliability
            else:
                pC1Start = 1 - cueReliability[t3]

            pC0Start = 1 - pC1Start

            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t3].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            # probably need an identity tracker for the new policies
            cuesOriginal = np.random.choice([0, 1], size=twinNum, p=[pC0Start, pC1Start])
            cuesOriginal = np.array(cuesOriginal)

            originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)
            # update identity tracker for new cues
            identTracker = updateIdentiyTracker(identTracker, cuesOriginal)
            doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel, identTrackerDoppel)
            identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cuesOriginal)

            # update cue tracker
            cueTracker[:, 0] += (1 - cuesOriginal)
            cueTracker[:, 1] += cuesOriginal

            cueTrackerDoppel[:, 0] += (1 - cuesOriginal)
            cueTrackerDoppel[:, 1] += cuesOriginal

    return originalTwin, doppelgaenger

class HiddenMarkovChain(object):
    def __init__(self, transition_prob, emission_prob):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition
            probabilities in Markov Chain.
            Should be of the form:
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.states = list(transition_prob.keys())
        self.hiddenStates = list(emission_prob[self.states[0]].keys())

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states,
            p=[self.transition_prob[current_state][next_state]
               for next_state in self.states]
        )

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        hidden_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            next_hidden_state = np.random.choice(self.hiddenStates, p = [self.emission_prob[current_state][cue] for cue in self.hiddenStates])
            hidden_states.append(next_hidden_state)
            current_state = next_state

        return hidden_states

    def generate_statesOntAdult(self, current_state, noOnt=10, noAdult = 1):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        hidden_states = []
        states = []
        noTotal = noOnt + noAdult
        for i in range(noTotal):

            next_state = self.next_state(current_state)
            next_hidden_state = np.random.choice(self.hiddenStates, p = [self.emission_prob[current_state][cue] for cue in self.hiddenStates])
            hidden_states.append(next_hidden_state)
            states.append(current_state)
            current_state = next_state

        return states[noOnt:],hidden_states[0:noOnt]

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
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
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
        posBeliefTracker = [0] * twinNum

        identTracker = cues
        if len(tValues) != 0:
            identTrackerDoppel = cues
        else:
            identTrackerDoppel = oppositeCues

        simValues = np.arange(0, twinNum, 1)

        for t in tValues:
            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            phenotypeTracker, posBeliefTracker = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker,
                                                                 identTracker)


            if t < T:
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)

                if adoptionType == "yoked":
                    oppositeCues = 1 - cues
                elif adoptionType == "oppositePatch":
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                    oppositeCues = np.array(oppositeCues)
                else:  # adoptionType = deprivation
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                    oppositeCues = np.array(oppositeCues)


            # update identity tracker for new cues
            identTracker = updateIdentiyTracker(identTracker, cues)
            if t != tValues[-1]:
                identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, cues)
            else:
                # last step is where we get yoked opposite cues
                identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)

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
            originalTwin, posBeliefOrg = updatePhenotype(subDF, originalTwin, simValues, cueTracker, identTracker)

            posBeliefTrackerOrg[:, t2 - tAdopt + 1] = posBeliefOrg

            doppelgaenger, posBeliefDG = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel,
                                                         identTrackerDoppel)
            posBeliefTrackerDG[:, t2 - tAdopt + 1] = posBeliefDG

            if t2 < T:
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

                identTracker = updateIdentiyTracker(identTracker,
                                                cuesOriginal)

                identTrackerDoppel = updateIdentiyTracker(identTrackerDoppel, oppositeCues)


                # update cue tracker
                cueTracker[:, 0] += (1 - cuesOriginal)
                cueTracker[:, 1] += cuesOriginal

                cueTrackerDoppel[:, 0] += (1 - oppositeCues)
                cueTrackerDoppel[:, 1] += oppositeCues

            # TODO reduce the amount of data stored for the posterior belief tracking
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

        # introduce identity tracker:
        identTracker = cuesSTart
        simValues = np.arange(0, twinNum, 1)
        for t in tValues:

            np.random.seed()
            subDF = policy[policy['time'] == t].reset_index(drop=True)

            # print identTracker
            phenotypeTracker, posBelief = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker, identTracker)
            phenotypeTrackerTemporal[:, :, t - 1] = np.copy(phenotypeTracker)
            posBeliefTrackerTemporal[:, t - 1] = np.copy(posBelief)

            if t < T:
                # now we have to recompute this for every timestep
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)
                # update cue tracker
                cueTracker[:, 0] += (1 - cues)
                cueTracker[:, 1] += cues
                identTracker = updateIdentiyTracker(identTracker, cues)



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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), n):
        yield lst[i:i + n]


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



def runTwinStudiesParallel(tAdopt, twinNum, env, pE0E0, pE1E1, pC1E1, adopt, T, resultsPath, argumentR, argumentP, adoptionType,
                           adultT):

    policyPath = os.path.join(resultsPath,
                              'runTest_%s%s_%s%s_%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE0E0, pE1E1, pC1E1))
    setGlobalPolicy(policyPath)

    # load the cue reliability array
    pC1E1 = D(str(pC1E1))
    pC0E1 = D(str(1))-pC1E1
    emissionProb = {  # these are the cue reliabilities
        'E0': {'0': pC1E1, '1': pC0E1},
        'E1': {'0': pC0E1, '1': pC1E1},
    }
    pE0E0,pE1E1 = D(str(pE0E0)),D(str(pE1E1))
    pE0E1 = D(str(1)) - pE0E0
    pE1E0 = D(str(1)) - pE1E1

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

    if not adultT:
        allCues, probabilities = generateCueSequencesAndProabilities(states, T, start_probability, transitionProb,
                                                                     emissionProb)
        if len(probabilities) <= twinNum:
            simulationChunk = chunks(allCues,
                                     12)  # this provides sublists of length 12 each, not exactly what I wanted bu perhaps it works

        else:
            simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

        pool = Pool(processes=12)


        # something is wrong in the first time step
        if adopt:
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

    else:
        HMM = HiddenMarkovChain(transitionProb, emissionProb)
        result = np.stack(
            [HMM.generate_statesOntAdult(startEnv, T, adultT) for x in np.arange(twinNum)])  # shape agentsxtime

        allStates, allCues = zip(*result)
        allCues = list(allCues)
        allStates = list(allStates)


        simulationChunk = chunks(allCues,12)

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

        return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), allStates


def calcEuclideanDistance(original, doppelgaenger):
    result = [np.sqrt(np.sum((x - y) ** 2)) for x, y in zip(original[:, 0:2], doppelgaenger[:, 0:2])]
    return np.array(result)


def runExperimentalAdoptionExperiment(T, numAgents, env, prior, cueReliability, resultsPath, argumentR, argumentP, lag,
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
        original, doppelgaenger = runExperimentalTwinStudiesParallel(t, numAgents, env, prior, cueReliability, lag, T,
                                                                     resultsPath, argumentR, argumentP, adoptionType,
                                                                     endOfExposure)
        results[t - 1, :] = calcEuclideanDistance(original, doppelgaenger)

    return results


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



def runAdoptionExperiment(T, numAgents, env, pE0E0, pE1E1, cueReliability, resultsPath, argumentR, argumentP, adoptionType):
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
        original, doppelgaenger, posBeliefOrg, posBeliefDG, originalTemp, doppelgaengerTemp,cueProbabilities = runTwinStudiesParallel(t,
                                                                                                                     numAgents,
                                                                                                                     env,
                                                                                                                     pE0E0,pE1E1,
                                                                                                                     cueReliability,
                                                                                                                     True,
                                                                                                                     T,
                                                                                                                     resultsPath,
                                                                                                                     argumentR,
                                                                                                                     argumentP,
                                                                                                                     adoptionType,
                                                                                                                     [])

        cueProbabilities = [float(elem) for elem in cueProbabilities]

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

                    uniqueFrac = [float(elem) for elem in uniqueFrac]
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
        dpi=600)

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

    plt.savefig(os.path.join(twinResultsAggregatedPath, 'fitnessDifferencesOverviewMerge%s.png'%levelsAutoCorrToPlot),bbox_inches='tight', dpi=600)
    plt.close()



def plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                   twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType, lag,
                                   endOfExposure,
                                   studyArg, priorArr, nameArg):
    T = T-1
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
        dpi=600)

    plt.close()



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
                uniqueFrac = [float(elem) for elem in uniqueFrac]
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

    plt.savefig(os.path.join(twinResultsPath, 'ternary_%s.png' % env), dpi=600)
    plt.close()


def calcFitnessSim(state, argumentR, argumentP,  T, beta, psi_weighting, adultStates):

    tfList = []  # this will hold all fitness values across the adult lifespan
    for adultState in adultStates:
        if adultState == 'E0':
            b0_D =1
            b1_D = 0
        else:
            b0_D = 0
            b1_D = 1
        tfCurr = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)
        tfList.append(tfCurr)

    tfList = np.array(tfList)
    return np.sum(tfList)








def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    y0, y1, yw = state

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


def calcFitness(state, argumentR, argumentP, adultT, markovChain, env, T, beta, psi_weighting):


    pE0E0, pE1E1 = markovChain
    pE0E0, pE1E1 = D(str(pE0E0)), D(str(pE1E1))
    P = np.array([[pE0E0, D(str(1)) - pE0E0], [D(str(1)) - pE1E1, pE1E1]])


    pE1 = D(str(env))
    pE0 = D(str(1))-D(str(env))
    b0_D = pE0
    b1_D = pE1


    tfList = []  # this will hold all fitness values across the adult lifespan
    for t in np.arange(1, adultT + 1, 1):
        tfCurr = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)
        tfList.append(tfCurr)
        # recalculate the distribution in the markov chain after one time step
        b0_D, b1_D = np.dot([pE0, pE1], np.linalg.matrix_power(P, t))

    tfList = np.array(tfList)
    return np.sum(tfList)



def fitnessDifferenceSim(markovProbabilities, cueValidityArr, policyPath, T, resultsPath, baselineFitness,
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
            maturePhenotypesEnv0, _, _, adultStatesEnv0 = \
                runTwinStudiesParallel(0, 5000, 0, pE0E0, pE1E1, cueReliability, False, T, policyPath,
                                       argumentR, argumentP, None, adultT)


            maturePhenotypesEnv1, _, _, adultStatesEnv1 = \
                runTwinStudiesParallel(0, 5000, 1, pE0E0, pE1E1, cueReliability, False, T, policyPath,
                                       argumentR, argumentP, None, adultT)

            prior = 1 - calcStationaryDist(markov_chain)

            OEnv1 = np.mean([calcFitnessSim(y, argumentR, argumentP, T, beta, 1,adultStatesCurr) for y,adultStatesCurr in
                          zip(maturePhenotypesEnv1, adultStatesEnv1)])
            OEnv0 = np.mean([calcFitnessSim(y, argumentR, argumentP, T, beta, 1,adultStatesCurr) for y,adultStatesCurr in
                          zip(maturePhenotypesEnv0, adultStatesEnv0)])

            OFitness = ((prior * float(OEnv1) + (1 - prior) * float(OEnv0)) - baselineFitness) / float(T*adultT)

            # next specialist Fitness
            if prior < 0.5:
                phenotypeS = np.array([T, 0, 0])
                SEnv1 = float(np.mean([calcFitnessSim(phenotypeS, argumentR, argumentP, T, beta, 1,adultStatesCurr) for adultStatesCurr in
                          adultStatesEnv1]))
                SEnv0 = float(np.mean([calcFitnessSim(phenotypeS, argumentR, argumentP, T, beta, 1,adultStatesCurr) for adultStatesCurr in
                          adultStatesEnv0]))



            else:
                resultLen = len(adultStatesEnv0)

                specialistPhenotypes = np.zeros((resultLen, 3))
                specialistPhenotypes[:, 0] = np.append(np.array([T] * int(resultLen / 2)),
                                                       np.array([0] * (resultLen - int(resultLen / 2))))

                specialistPhenotypes[:, 1] = np.append(np.array([0] * int(resultLen / 2)),
                                                       np.array([T] * (resultLen- int(resultLen / 2))))

                SEnv1 = float(np.mean(
                    [calcFitnessSim(y, argumentR, argumentP, T, beta, 1, adultStatesCurr) for y, adultStatesCurr in
                     zip(specialistPhenotypes, adultStatesEnv1)]))
                SEnv0 = float(np.mean(
                    [calcFitnessSim(y, argumentR, argumentP, T, beta, 1, adultStatesCurr) for y, adultStatesCurr in
                     zip(specialistPhenotypes, adultStatesEnv0)]))


            SFitness = ((prior * SEnv1 + (1 - prior) * SEnv0) - baselineFitness) / float(T*adultT)
            phenotypeG = np.array([T / float(2), T / float(2), 0])

            GEnv1 = float(np.mean(
                [calcFitnessSim(phenotypeG, argumentR, argumentP, T, beta, 1, adultStatesCurr) for adultStatesCurr in
                 adultStatesEnv1]))
            GEnv0 = float(np.mean(
                [calcFitnessSim(phenotypeG, argumentR, argumentP, T, beta, 1, adultStatesCurr) for adultStatesCurr in
                 adultStatesEnv0]))



            GFitness = ((prior * GEnv1 + (1 - prior) * GEnv0) - baselineFitness) / float(T*adultT)

            resultsDict[((pE0E0, pE1E1), cueReliability)] = np.array([SFitness, OFitness, GFitness])

    pickle.dump(resultsDict, open(os.path.join(resultsPath, "fitnessDifferences.p"), "wb"))








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

            #load the optimal policy
            policyPathData = os.path.join(policyPath,
                                      'runTest_%s%s_%s%s_%s/finalRaw.csv' % (
                                      argumentR[0], argumentP[0], pE0E0, pE1E1, cueReliability))
            policy = pd.read_csv(policyPathData, index_col=0).reset_index(drop=True)

            subDf = policy[policy['time'] == T].reset_index(drop=True)

            optimalDecision = subDf['cStar']

            optDecisionsNum = [
                int(a) if not '(' in str(a) else int(
                    np.random.choice(str(a).replace("(", "").replace(")", "").split(","))) for
                a in
                optimalDecision]

            tf = subDf['fitness'].values
            tf = [float(val) for val in tf]
            posE1 = subDf['pE1'].values
            posE1 = [float(val) for val in posE1]
            stateProb = subDf['stateProb'].values
            stateProb = [float(val) for val in stateProb]

            # fitness following the optimal policy
            OFitness = D(str(np.average(tf, weights=stateProb))) / D(str((adultT * T)))


            prior = D(str(1)) - D(str(calcStationaryDist(markov_chain)))  # this is prior E1

            # next specialist Fitness
            if prior < 0.5:
                specialistPhenotypes = np.array([T, 0, 0])
                SFitnessArr = [calcFitness(specialistPhenotypes, argumentR, argumentP, adultT, markov_chain, posE1Curr, T, beta, 1)
                                     for posE1Curr in posE1]
                SFitness = D(str(np.mean(SFitnessArr))) / D(str((adultT * T)))


            else:


                resultLen = numAgents

                specialistPhenotypes = np.zeros((resultLen, 3))
                specialistPhenotypes[:, 0] = np.append(np.array([T] * int(resultLen / 2)),
                                                       np.array([0] * (resultLen - int(resultLen / 2))))

                specialistPhenotypes[:, 1] = np.append(np.array([0] * int(resultLen / 2)),
                                                       np.array([T] * (resultLen- int(resultLen / 2))))

                SFitnessArr = [calcFitness(phenotypeS,  argumentR, argumentP, adultT, markov_chain, posE1Curr, T, beta, 1) for
                            phenotypeS,posE1Curr in zip(specialistPhenotypes,posE1)]


                SFitnessArr = [float(val) for val in SFitnessArr]
                SFitness = D(str(np.average(SFitnessArr, weights=stateProb))) / D(str((adultT * T)))

            phenotypeG = np.array([T / float(2), T / float(2), 0])
            GFitnessArr = [
                calcFitness(phenotypeG, argumentR, argumentP, adultT, markov_chain, posE1Curr, T, beta, 1)
                for posE1Curr in posE1]

            GFitnessArr = [float(val) for val in GFitnessArr]

            GFitness = D(str(np.average(GFitnessArr, weights=stateProb))) / D(str((adultT * T)))

            resultsDict[((pE0E0, pE1E1), cueReliability)] = np.array([SFitness, OFitness, GFitness])

    pickle.dump(resultsDict, open(os.path.join(resultsPath, "fitnessDifferences.p"), "wb"))
    del resultsDict


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
    plt.savefig(os.path.join(twinResultsPath, 'fitnessDifferences.png'), dpi=600)
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


def plotRankOrderStability(priorE0Arr, cueValidityArr, twinResultsPath, T, types):
    for distFun in types:
        plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun)


def createLABELS(T):
    labels = [" "] * T
    labels[0] = str(1)
    labels[T - 1] = str(T)
    labels[int(T / 2) - 1] = str(T / 2)
    return labels


def plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun):
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
    ranks = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRanks.p"), "rb"))

    # what do we want to plot?
    # could have a plot with the correlation coefficient between consecutive timesteps
    # or a whole correlation matrix, heatplot? start with this
    # want to represent the proportion of ties as well

    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    plt.subplots_adjust(top =0.92, bottom = 0.12)
    specialAx = fig.add_axes([.16, .040, .7, .01])
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for pE0 in priorE0Arr:
            rankMatrix = ranks[(pE0, cueVal)]

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
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            rankMatrix = ranks[(pE0, cueVal)]

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
                yLabel = 'ontogeny'

            # only negative rank switches

            # create a mask for the upper triangle
            mask = np.tri(sim.shape[0], k=0)




            if jx == len(priorE0Arr) - 1 and ix == 0:
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
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.set_xlabel('ontogeny', fontsize=20, labelpad=15)
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)

                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.tick_params(labelsize=15)
            else:
                #ax.get_xaxis().set_visible(False)
                ax2.set_yticks([])

            # if jx == 0:
            #     ax.yaxis.set_label_position("left")
            #     ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)
            #     ax2.set_yticks(np.arange(0, 1.1, 0.2))
            #     ax2.tick_params(labelsize=15)
            # else:
            #     ax2.set_yticks([])

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStability2%s.png' % distFun), dpi=1200)
    plt.close()

    # # second plot is for rank stability
    # fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    # specialAx = fig.add_axes([.16, .055, .7, .01])
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, "stable")
    #
    #         simRange += list(sim.flatten())
    #
    # boundary1 = min(simRange)
    # boundary2 = max(simRange)
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #             cmap = 'YlGnBu'
    #             yLabel = 'Cosine similarity'
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'stable')
    #             cmap = 'Greys'  # 'YlGnBu'
    #             yLabel = 'Time step'
    #
    #         # only negative rank switches
    #
    #         # create a mask for the upper triangle
    #         mask = np.tri(sim.shape[0], k=0)
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             cbar = True
    #             cbar_ax = specialAx
    #             cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
    #                         'label': "Proportion of stable ranks"}  # 'label':"Proportion of negative rank switches",
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #
    #             cbar = ax.collections[0].colorbar
    #             # here set the labelsize by 20
    #             cbar.ax.tick_params(labelsize=14)
    #             cbar.ax.xaxis.label.set_size(20)
    #         else:
    #             cbar = False
    #             cbar_ax = None
    #             cbar_kws = None
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #             ax.tick_params(labelsize=14)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.spines['left'].set_visible(False)
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=10)
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos1%s.png' % distFun), dpi=1200)
    # plt.close()
    #
    # # 3rd plot
    # fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         simRange += list(sim.flatten())
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)
    #
    #
    #         else:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(True)
    #         ax.spines['left'].set_visible(False)
    #
    #         ax.set_ylim(0, 1)
    #         plt.yticks([])
    #         plt.xticks([])
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #         #
    #         # if jx == 0:
    #         #     plt.title(str(cueVal), fontsize=30)
    #         #
    #         # if ix == 0 and jx == 0:
    #         #     ax.set_xlabel('Time', fontsize=30, labelpad=10)
    #         #     ax.spines['left'].set_visible(True)
    #         #     ax.yaxis.set_label_position("left")
    #         #     ax.set_ylabel('Proportion of rank switches', fontsize=30, labelpad=10)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('Proportion of rank switches', fontsize=20, labelpad=10)
    #
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos2%s.png' % distFun), dpi=1200)
    # plt.close()




def plotBeliefAndPhenotypeDivergence(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            relativeDistanceDiff = np.gradient(relativeDistance)


            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:, 5] #measured at the end of ontogeny after the last cue
            posBeliefDiffNoAverageDiff = np.gradient(posBeliefDiffNoAverage)


            plt.plot(tValues[0:], posBeliefDiffNoAverageDiff, color='grey', linestyle='solid', linewidth=2, markersize=5,
                         marker='D',
                         markerfacecolor='grey')

            plt.plot(tValues[0:], relativeDistanceDiff, color='black', linestyle='solid', linewidth=2, markersize=5,
                         marker='o',
                         markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.4, 0.05)
            plt.yticks([-0.3,0,0.05], fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('gradient of plasticity curves', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues)-0.5,max(tValues)+0.5+1,1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefEndOntogenyDivergence.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()


def plotBeliefDistances(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:, 5] #measured at the end of ontogeny after the last cue

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
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)


            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues)-0.5,max(tValues)+0.5+1,1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefEndOntogeny.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()

    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:,
                                         6]  # measured after each cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            relativeDistanceTemp = relativeDistanceDictTemp[(pE0, cueVal)]
            plt.plot(tValues, relativeDistanceTemp, color='black', linestyle='solid', linewidth=2, markersize=8,
                         marker='o',markerfacecolor='black')

            print "The current prior is %s and the cue reliability is %s" % ((1 - pE0), cueVal)
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
                plt.title(str(1 - pE0), fontsize=20)


            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize = 15)

            else:
                ax.get_xaxis().set_visible(False)


            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)


            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability
            tValNew = np.arange(min(tValues) - 0.5, max(tValues) + 0.5 + 1, 1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefAfterCue.png' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
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
        dpi=600)

    plt.close()



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
                print "currently working with prior: " + str(pE0E0) + " and cue reliability: " + str(cueReliability)

                startTime = time.time()
                resultsMat, resultsMatBeliefs, resultsMatTempPhenotypes, cueProbabilities = runAdoptionExperiment(T, numAgents, env, pE0E0,
                                                                                                pE1E1,
                                                                                                cueReliability,
                                                                                                resultsPath, argumentR,
                                                                                                argumentP, adoptionType)
                elapsedTime = time.time() - startTime
                print "Elapsed time: " + str(elapsedTime)

                # normalize resultsmat
                pickle.dump(resultsMat,
                            open(os.path.join(twinResultsPath,
                                              "resultsMat_%s%s_%s_%s.p" % (pE0E0, pE1E1, env, cueReliability)),
                                 "wb"))
                pickle.dump(cueReliability, open(os.path.join(twinResultsPath,
                                                              "cueProbability_%s%s_%s_%s.p" % (
                                                              pE0E0, pE1E1, env, cueReliability)),
                                                 "wb"))

                absoluteDistance, relativeDistance, absoluteDistanceVariance, relativeDistanceVariance = postProcessResultsMat(
                    resultsMat, T, endOfExposure, lag, cueProbabilities)

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
                                                                                         True, 1, cueProbabilities)
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

        del absoluteDistanceDict
        del relativeDistanceDict
        del absoluteDistanceDictVar
        del relativeDistanceDictVar

        del absoluteDistanceDictTemp
        del relativeDistanceDictTemp

        del beliefDict

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
        del maturePhenotypes
        del maturePhenotypesTemp
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
        # fitnessDifferenceSim(markovProbabilities, cueValidityC0E0Arr, resultsPath, T, twinResultsPath, baselineFitness,
        #                      argumentR,
        #                      argumentP, adultT, numAgents)

        fitnessDifference(markovProbabilities, cueValidityC0E0Arr, resultsPath, T, twinResultsPath, baselineFitness,
                          argumentR,
                          argumentP, adultT, numAgents)


    elif argument == 'FitnessDifferenceSim':
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
             argumentR, argumentP, lagArray, adoptionType, endOfExposure, plotArgs, plotVar,performSimulation, adultT,
             startingEnv):
    for arg in plotArgs:
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


def runAggregatePlotsMerge(markovProbabilities, cueValidityArr, T, adultTArr, startEnvList, autoCorrPathCurr,
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
    adultTArrSubset = adultTArr  # 1, 5, 20
    levelsAutoCorrToPlotArg = None

    r = 0.5  # 0.5  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
    minProb = 0  # minimal probability of reaching a state for those that are displayed
    lines = True

    adulthood = False

    # policyPlotReducedMergeBW(0.6, minProb, lines, cueValidityArr, T, 20, [0.2, 0.5, 0.8], autoCorrDictTotal,
    #                          twinResultsAggregatedPath, dataPath, argumentR, argumentP, True, False,
    #                          ['symmetric', 'asymmetric (E0)'])

    # policyPlotReducedMerge(r, minProb, lines, cueValidityArr, T, 5, [0.2, 0.5, 0.8], autoCorrDictTotal,
    #                        twinResultsAggregatedPath, dataPath, argumentR, argumentP, adulthood, False,
    #                        ['symmetric', 'asymmetric (E0)'])
    plotFitnessDifferenceOverviewMerge(cueValidityArr, T, adultTArrSubset, autoCorrDictTotal,
                                       twinResultsAggregatedPath, dataPath, argumentR, argumentP, [0.2, 0.5, 0.8],
                                       mergeArr)

    # ternaryMercedesMerge(cueValidityArr, T, adultTArrSubset, 0, autoCorrDictTotal,
    #                     twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag, endOfExposure,
    #                     studyArg,[0.2, 0.5, 0.8] , ['symmetric', 'asymmetric (E0)','asymmetric (E1)'],nameArg)



    # plotPlasticityCurvesOverview33(cueValidityArr, T+1, adultTArrSubset, 0, autoCorrDictTotal,
    #                                twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag,
    #                                endOfExposure, studyArg, mergeArr, nameArg)


def runAggregatePlotsAssymetries(markovProbabilities, cueValidityArr, T, adultTArr, startEnvList, autoCorrPathCurr,
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



    adultTArrSubset = adultTArr
    for startEnv in startEnvList:
        plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArrSubset, startEnv, autoCorrDictTotal,
                                       twinResultsAggregatedPath, dataPath, argumentR, argumentP, adoptionType, lag,
                                       endOfExposure,
                                       studyArg, mergeArr, nameArg)


def runAggregatePlots(markovProbabilities, cueValidityArr, T, adultTArr, startEnvList, autoCorrPath,
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
    adultTArrSubset = adultTArr
    r = 0.5#0.5  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
    minProb = 0  # minimal probability of reaching a state for those that are displayed
    lines = True

    adulthood = False
    policyPlotReducedOverview2(r, minProb, lines, cueValidityArr, T, adultTArrSubset,[0.2, 0.5, 0.8], autoCorrDict,
                             twinResultsAggregatedPath, dataPath, argumentR, argumentP, adulthood,False)

