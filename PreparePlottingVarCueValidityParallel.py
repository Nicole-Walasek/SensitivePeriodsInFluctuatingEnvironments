import cPickle as pickle
import numpy as np
import os
from ForwardPassVarCueValidityParallel import doForwardPass
import pandas as pd
import time as timer
from BTrees.OOBTree import OOBTree
import shelve
import multiprocessing
import itertools
import matplotlib
import matplotlib.pyplot as plt


# what we need
# set the current working directory

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    rel_tol = float(rel_tol)
    abs_tol = float(abs_tol)
    if type(a) != type(b):
        return False
    elif isinstance(a, tuple):
        return a == b
    else:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def allclose(tupleA, tupleB, rel_tol=1e-09, abs_tol=0.0):
    rel_tol = float(rel_tol)
    abs_tol = float(abs_tol)
    return np.all(
        [isclose(a, b, rel_tol, abs_tol) if not (isinstance(a, str) or isinstance(b, str)) else a == b for a, b in
         zip(tupleA, tupleB)])


def compareTupleToList(tuple, tupleList):
    # return true if tuple already in list
    return np.any([allclose(tuple, compareTuple) for compareTuple in tupleList])


def findIndexOfClosest(tuple, tupleList):
    for idx, val in enumerate([allclose(tuple, compareTuple) for compareTuple in tupleList]):
        if val:
            return idx

def listSplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


# in this version we marginalize over cue sets
# different order of cues sampled does not matter here
# agents might have the same posterior beliefs despite having sampled different cue sets
# these data will be used to generate plots of the following format:
# x-axis: time
# y-axis: pE1
# color: decision, pies represent situations in which agents with the same posterior made different decisions
# area: probability of reaching a state

def aggregareResultsMarginalizeOverCues(resultsPath, plottingPath, T):
    # create a folder in which to store the aggregated results in

    aggregatedPath = os.path.join(plottingPath, 'aggregatedResults')

    if not os.path.exists(aggregatedPath):
        os.makedirs(aggregatedPath)

    tValues = np.arange(1, T+1, 1)

    # time level
    for t in tValues:
        print 'Currently aggregating data for time step %s' % t
        # batch level


        #load the dataframe for the current time step
        resultsDF = pd.read_csv(os.path.join(resultsPath, '%s.csv' % (t)))
        # next sort the data frame by belief, decision and marker (in this order) to speed up the aggregation
        resultsDF = resultsDF.sort_values(by=['pE1', 'cStar', 'marker'])

        resultsDF = resultsDF.reset_index(drop=True)
        resultsDF= resultsDF.loc[:, ~resultsDF.columns.str.contains('^Unnamed')]


        time = []
        pE1 = []
        decision = []
        stateProb = []
        marker = []
        tupleTracker = []
        x0 = []
        x1 = []
        x0_2 = []
        x1_2 = []

        time2 = []
        pE12 = []
        tupleTracker2 = []

        startTime = timer.clock()

        for idx in range(len(resultsDF)):
            timeIDX, pE1IDX, decIDX, stateProbIDX, markerIDX, x0IDX, x1IDX = resultsDF.loc[
                idx, ['time', 'pE1', 'cStar', 'stateProb', 'marker','x0','x1']]

            timeIDX, markerIDX, x0IDX,x1IDX = int(timeIDX), int(markerIDX), int(x0IDX), int(x1IDX)


            pE1IDX = round(pE1IDX, 3)
            tupleIDX = (timeIDX, pE1IDX, decIDX, markerIDX)
            tupleIDX2 = (timeIDX, pE1IDX)

            # if we haven't stored those time x belief coordinates yet, append them
            if idx == 0 or not (tupleIDX2 in tupleTracker2):  # compareTupleToList(tupleIDX2, tupleTracker2):
                pE12.append(pE1IDX)
                tupleTracker2.append(tupleIDX2)
                time2.append(timeIDX)
                x0_2.append(x0IDX)
                x1_2.append(x1IDX)

            else:
                # otherwise look up the last entry with the same time  x belief coordinates
                updateIDX = tupleTracker2.index(tupleIDX2)

            # here we check time x belief x decision x marker tuples
            if idx == 0 or (tupleIDX not in tupleTracker):  # not compareTupleToList(tupleIDX, tupleTracker):   #
                time.append(timeIDX)
                pE1.append(pE1IDX)
                decision.append(decIDX)
                stateProb.append(stateProbIDX)
                marker.append(markerIDX)
                tupleTracker.append(tupleIDX)
                x0.append(x0IDX)
                x1.append(x1IDX)
            else:
                # else it might be that we already stored this particular combination
                updateIDX = tupleTracker.index(tupleIDX)
                # updateIDX = findIndexOfClosest(tupleIDX,tupleTracker)
                stateProb[updateIDX] += stateProbIDX

        aggregatdResultsDF = pd.DataFrame(
            {'time': time, 'pE1': pE1, 'cStar': decision, 'stateProb': stateProb, 'marker': marker,'x0': x0, 'x1':x1})
        aggregatdResultsDF.to_csv(os.path.join(aggregatedPath, 'aggregatedResults_%s.csv' % t))

        plottingResultsDF = pd.DataFrame([[a,b,c,d] for a,b,c,d in zip(time2,pE12,x0_2,x1_2)],
                                             columns=['time', 'pE1','x0','x1'])

        plottingResultsDF.to_csv(os.path.join(aggregatedPath, 'plottingResults_%s.csv' % t))
        elapsedTime = timer.clock() - startTime

        print "Elapsed time for time step %s:  %s" % (t, elapsedTime)


# the following functions are for multiprocessing purposes
def _apply_marker(cStarList):
    marker3 = [idx for idx, val in cStarList if (isinstance(val, tuple) and val == (0, 1))]
    marker4 = [idx for idx, val in cStarList if
               (isinstance(val, tuple) and not (val == (0, 1)))]
    return (marker3, marker4)


def marker_multiprocessing(cStarList, workers):
    pool = multiprocessing.Pool(processes=workers)
    # list split works like np.array_split and splits a list into n parts
    # just write a function that can do this for a list
    result = pool.map(_apply_marker, [statesSubset for statesSubset in listSplit(cStarList, workers)])
    pool.close()
    pool.join()
    a, b = zip(*result)
    return itertools.chain.from_iterable(a), itertools.chain.from_iterable(b)


def _apply_df(state):
    unSplit = [(s.split(";")[0], s.split(";")[1], s.split(";")[2], s.split(";")[3],
                s.split(";")[4], s.split(";")[5]) for s in state]
    return unSplit


def unsplit_multiprocessing(states, workers):
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [statesSubset for statesSubset in np.array_split(states, workers)])
    pool.close()
    pool.join()
    return itertools.chain.from_iterable(result)


"""
This is the core function that delegates and calls all the other functions
    it is the one that is called from the main script

"""

def preparePlotting(T, pE0, pE1, kwUncertainTime,finalTimeArr,emission_prob):

    tValues = np.arange(1, T+1, 1) #this needs to run from 1 until 20
    """
    The forward pass produces a paramCombination/plotting folder which contains one folder per time step
        each folder follows the same structure as the optimal policy, it has as many batches as the optimal policy 
        
        consequently the plotting will have to be handled in these batches as well 
    """

    doForwardPass(T, finalTimeArr)

    """
    Don't think I need the time to decision mapping right now
        for each time step it assigns the proportion of the population making a particular choice
    
    """
    """
    Procedure:
    loop over policy and state distribution files, assume standard locaions for both and that the working directory
    is set accordingly 
    """
    policyPath = os.path.join(os.getcwd(), 'fitness')
    plottingPath = os.path.join(os.getcwd(), 'plotting/StateDistribution')
    resultsPath = os.path.join(os.getcwd(), 'plotting/resultDataFrames')

    # create a resultsDataFrame folder in the current plotting folder
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    for t in tValues:


        print "Currently working on time step: %s" % t

        #loading the optimal policy
        currentPStar = pickle.load(open(os.path.join(policyPath, '%s/TF.p' % (t-1)),"rb"))
        #transform into data frame
        resultsDF = pd.DataFrame.from_dict(currentPStar, orient='index')
        # add this point we do not need optimal policy anymore
        del currentPStar
        resultsDF.columns = ['fitness', 'cStar', 'pC1', 'pE1']
        states = resultsDF.index
        resultsDF = resultsDF.reset_index(drop=True)
        resultsDF['states'] = states

        # next load the respective state distribution
        P = pickle.load(open(os.path.join(plottingPath, "%s/P.p" % (t-1)),"rb"))

        stateProb = [P[state] if state in P else -1000 for state in
                     states]
        del P
        resultsDF['stateProb'] = stateProb
        del stateProb
        del states

        """
        we preapred the results dataframe for this time step
        """
        # deleting zero entries to speed up processing and save memory, can commented out if not needed
        # might lead to empty dataframes

        # always leave one identifier
        resultsDF = resultsDF[resultsDF.stateProb != -1000].reset_index(drop=True)
        if len(resultsDF) != 0:
            statesToSplit = resultsDF.states
            unSplit = _apply_df(statesToSplit)

            resultsDF['x0'], resultsDF['x1'], resultsDF['y0'], resultsDF['y1'], resultsDF[
                'yw'], resultsDF['time'] = zip(*unSplit)
            del unSplit
            # dropping the states column now
            resultsDF = resultsDF.drop(columns=['states'])
            marker = np.array(resultsDF['cStar'])
            cStarList = list(enumerate(resultsDF['cStar']))
            marker3, marker4 = marker_multiprocessing(cStarList, 32)
            marker[list(marker3)] = 3
            marker[list(marker4)] = 4
            resultsDF['marker'] = list(marker)
            del marker
            del marker3
            del marker4
            #print resultsDF
            timeList = list(resultsDF['time'])
            timeList = np.array(map(int, timeList)) + 1
            timeList = list(map(str, timeList))
            resultsDF['time'] = timeList

            # do this ony for the very first time step
            if t == 1:
                new_row = ['-', '-', '-', pE1,1, 0, 0, 0, 0, 0, 0, -1]
                resultsDF.loc[len(resultsDF)] = new_row
            resultsDF = resultsDF.sort_values(['x0', 'y0', 'y1'], ascending=[1, 1, 1])
            resultsDF = resultsDF.reset_index(drop=True)
            resultsDF.to_csv(os.path.join(resultsPath, '%s.csv' % (t)))

        del resultsDF

    print 'Starting to aggregate the raw data for plotting'
    aggregareResultsMarginalizeOverCues(resultsPath, 'plotting', T)
