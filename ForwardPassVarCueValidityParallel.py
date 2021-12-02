"""
# implementing a forward pass through the optimal policy in order to calculate the
# state distribution matrix
# it indicates the probability of reaching this particular state, following the optimal policy
"""
import numpy as np
import cPickle as pickle
import time
from scipy.stats import binom
import os
import shelve
from multiprocessing import Pool
from progressbar import Bar, Percentage, ProgressBar, Timer, ETA
import itertools
import sys
from BTrees.OOBTree import OOBTree
from decimal import Decimal as D

def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()


def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

def set_global_P(setP):
    global P
    P = setP


def set_global_Policy(setcurrBatchPstar):
    global currBatchPstar
    currBatchPstar = setcurrBatchPstar


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def compressDict(originalDict):
    newDict = {}
    for key in originalDict:
        (_, a, b, _) = originalDict[key]
        newDict[key] = (a, b)
    t = OOBTree()
    t.update(newDict)
    del newDict

    return t


def BayesianUpdating(pE0, pE1, pDE0, pDE1):
    pE0, pE1, pDE0, pDE1 = float(pE0), float(pE1), float(pDE0), float(pDE1)
    # pE0 is the evolutionary prior vor environment 1
    # pE1 is the evolutionary prior for environment 2
    # pDE0 and pDE1 are the probabilities of obtaining the data given environment 0 or 1 respectively (likelihood)
    p_D = pDE0 * pE0 + pDE1 * pE1
    b0_D = (pDE0 * pE0) / p_D
    b1_D = (pDE1 * pE1) / p_D

    return b0_D, b1_D


def findMaxIdentifier(pStarBatch):
    identifers = [int(x.split(';')[0]) for x in pStarBatch]
    return (max(identifers),min(identifers))


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    if n == 0:
        yield l
    else:
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]

def calcForwardPass(currState, survivalProb):
    batchResult = OOBTree()

    ident, x0, x1, y0, y1, yw, t = currState.split(';')
    ident, x0, x1, y0, y1, yw, t = int(ident), int(x0), int(x1), int(y0), int(y1), int(yw), int(t)
    #
    (optDecision, pC1_D) = currBatchPstar[currState]


    pC1_D = float(pC1_D)
    nextT = t + 1
    pC0_D = 1 - pC1_D
    if currState in P:
        currentProb = P[currState]

        x0add1 = x0 + 1
        x1add1 = x1 + 1
        identx0 = ident * 2
        identx1 = ident * 2 + 1

        if isinstance(optDecision, tuple):
            currentLen = len(optDecision)
            for idx in optDecision:
                phenotype = [y0, y1, yw]
                phenotype[idx] += 1
                y0New, y1New, ywNew = phenotype
                # now tranform back to strings
                x0add1Key = '%s;%s;%s;%s;%s;%s;%s' % (identx0, x0add1, x1, y0New, y1New, ywNew, nextT)
                x1add1Key = '%s;%s;%s;%s;%s;%s;%s' % (identx1, x0, x1add1, y0New, y1New, ywNew, nextT)

                batchResult[x0add1Key] = 0
                batchResult[x1add1Key] = 0

                batchResult[x0add1Key] += currentProb * (1 / float(
                    currentLen)) * pC0_D * survivalProb
                batchResult[x1add1Key] += currentProb * (1 / float(currentLen)) * pC1_D * survivalProb

        else:

            # enumerate all possible successor phenotype states following the optimal policy
            phenotype = [y0, y1, yw]
            phenotype[optDecision] += 1
            y0New, y1New, ywNew = phenotype
            # now tranform back to strings for stroing
            x0add1Key = '%s;%s;%s;%s;%s;%s;%s' % (identx0, x0add1, x1, y0New, y1New, ywNew, nextT)
            x1add1Key = '%s;%s;%s;%s;%s;%s;%s' % (identx1, x0, x1add1, y0New, y1New, ywNew, nextT)
            # this is basically the initalization step
            batchResult[x0add1Key] = 0
            batchResult[x1add1Key] = 0
            batchResult[x0add1Key] += currentProb * pC0_D * survivalProb
            batchResult[x1add1Key] += currentProb * pC1_D * survivalProb

    return batchResult



def func_star(allArgs):
    return calcForwardPass(*allArgs)


def intialization(currBatchPstar, pC0E0, pC1E1, pE0, pE1, finalTime, stateDistPath, currLen):
    localP = {}
    # this is the initialization
    for currKey in currBatchPstar:
        localP[currKey] = 0

    currKey = '%s;%s;%s;%s;%s;%s;%s' % (0, 0, 0, 0, 0, 0, 0)
    localP[currKey] = 1  # this where the whole population starts


    tree_path = 'trees'
    trees = shelve.open(os.path.join(tree_path, "tree%s" % (0)))

    tree0 = dict(trees.items())['0']
    pC1D = tree0.pC1D
    pC0D = tree0.pC0D


    initialSurvProb = 1 - float(finalTime[1])

    localP['%s;%s;%s;%s;%s;%s;%s' % (1, 0, 1, 0, 0, 0, 0)] = float(pC1D) * localP[currKey] * initialSurvProb
    localP['%s;%s;%s;%s;%s;%s;%s' % (0, 1, 0, 0, 0, 0, 0)] = float(pC0D) * localP[currKey] * initialSurvProb
    # dump it in the respective folder
    myShelve = shelve.open(os.path.join(stateDistPath, "%s/P%s" % (0, currLen)))
    myShelve.update(localP)
    myShelve.close()


"""
This is the main function that is called 

    steps
    - make a plotting folder; check 
    - iterate over time and have one folder per time step
    - work in batches, i.e. define a batchsize parameter from the main script
    - always need t and t+1 in memory 
"""


def doForwardPass(T, fitnessPathHDD, pC0E0, pC1E1, pE0, pE1, finalTime, batchSize):
    # start the clock
    startTime = time.clock()
    # the current working directory is the folder for the respective parameter combination
    # a plotting folder exists
    # add a state distribution folder

    set_global_Policy({})
    stateDistPath = 'plotting/StateDistribution'
    fitnessPath = 'fitness'

    if not os.path.exists(stateDistPath):
        os.makedirs(stateDistPath)

    pC0E0, pC1E1, pE0, pE1 = float(pC0E0), float(pC1E1), float(pE0), float(pE1)

    tValues = np.arange(1,T+1, 1)

    """
    This is the initialization 
    - here we basically initialize a state distribution vector for t equals 0 and determine how the population starts 
    """
    print "start initialization"

    # make one folder per time step
    if not os.path.exists(os.path.join(stateDistPath, str(0))):
        os.makedirs(os.path.join(stateDistPath, str(0)))

    # what follows now should always be just one file in the directory
    # however I will make it general enough in case future models will deal with a larger state space
    for batchPstar in os.listdir('fitness/%s' % 0):  # directly navigating to the corresponding time step folder
        currBatchPstar = {}
        # either a pickled file or a shelve
        if batchPstar.endswith('.p'):
            currBatchPstar.update(pickle.load(open("fitness/0/%s" % batchPstar, 'rb')))
        else:  # must be a shelve
            currBatchPstar.update(shelve.open("fitness/0/%s" % batchPstar))

        currLen = batchPstar.replace('TF', '').replace('.p', '')
        # now the actual initialization
        intialization(currBatchPstar, pC0E0, pC1E1, pE0, pE1, finalTime, stateDistPath, currLen)


    del currBatchPstar
    print 'finished initialization'
    """
    Here, the actual calculation of the state distribution matrix will begin 
    - this will constitute a forward pass  
    - the resulting P matrices are a one-to-one mapping from policy states, if a state from the policy is not in the 
        in the P matrix, it's state distribution can be assumed to be zero 
    """

    # initialize a global dicts for P and policy star so that all parallel workers have access to it

    for t in tValues:
        global currBatchPstar
        currBatchPstar = {}
        global P
        P = {}
        #print "currently preparing time step: " + str(t)

        if t > 17:
            fitnessPath = fitnessPathHDD

        if not os.path.exists(os.path.join(stateDistPath, str(t))):
            os.makedirs(os.path.join(stateDistPath, str(t)))

        # make the respective path a variable here
        batchPstarList = [batchPstar for batchPstar in os.listdir(os.path.join(fitnessPath, '%s' % (t-1)))]
        batchPstarListSorted = [batchPstar for batchPstar in
                                sorted(batchPstarList, key=lambda x: int(x.replace('TF', '').replace('.p', '')))]

        down()
        widgets = ['Time step %s:' % t, Percentage(), ' ',
                   Bar(marker=('-'), left='[', right=']'), ' ', ETA(), ' ']
        pbarTimeStep = ProgressBar(widgets=widgets, maxval=int(len(batchPstarListSorted))).start()

        for batchPstar in pbarTimeStep(batchPstarListSorted):  # directly navigating to the corresponding time step folder
            up()
            localP = OOBTree()
            localBatchPstar = OOBTree()

            currLen = batchPstar.replace('TF', '').replace('.p', '')

            # first load the respective policy
            # either a pickled file or a shelve
            if batchPstar.endswith('.p'):

                localBatchPstar.update(pickle.load(open(os.path.join(fitnessPath, "%s/%s" % ((t-1), batchPstar)), 'rb')))
            else:  # must be a shelve
                localBatchPstar.update(shelve.open(os.path.join(fitnessPath, "%s/%s" % ((t-1), batchPstar))))

            # free up memory by deleting the local copies
            localBatchPstarCompressed = compressDict(localBatchPstar)
            del localBatchPstar
            set_global_Policy(localBatchPstarCompressed)
            del localBatchPstarCompressed

            if os.path.exists(os.path.join(stateDistPath, "%s/P%s" % ((t-1), currLen))):
                localP.update(shelve.open(os.path.join(stateDistPath, "%s/P%s" % ((t-1), currLen))))
            else:
                print "No such file" + str(os.path.join(stateDistPath, "%s/P%s" % ((t-1), currLen)))

            set_global_P(localP)
            del localP

            """
            Implement the parallelization 
            - this part will perform the actual calculation of the state distribution matrix 
            - for each state extract the relevant statistics from both the state distribution vector and the optimal
                policy
            - next use these informaton to do the forward updating
            - it should be possible to distribute that across workers 
            - should be inspired by the parallelization of the terminla fitness calculation?  
            """
            # now iterae through the optimal policy
            # structure:
            startCondition = int(currLen) - batchSize

            # prepare the respective pDicts
            # these will be the resultng P-values that I will need to fill
            identTimesTwo = startCondition * 2 + batchSize
            identTimesTwoPlusOne = identTimesTwo + batchSize
            cutOff1 = startCondition + (int(batchSize)/2 - 1)
            cutOff2 = cutOff1+ 1


            print "Currently working on batch: %s"  % currLen
            # this quantity is only dependent on t
            survivalProb = 1 - float(finalTime[t])

            # what is the maximim identifier?
            maxIdent,lowestIdent = findMaxIdentifier(currBatchPstar)
            subBatches = []
            # this step is in place to decide whether we want to initalize two outcome files or one
            if maxIdent >= (batchSize / 2):
                subBatches.append(identTimesTwo)
                subBatches.append(identTimesTwoPlusOne)
            else:
                subBatches.append(identTimesTwo)


            for elem in subBatches:
                finalResults = OOBTree()
                if elem == identTimesTwo:
                    parallelStates = [currState for currState in
                                      sorted(currBatchPstar.keys(), key=lambda x: int(x.split(';')[0])) if
                                      int(currState.split(";")[0]) <= cutOff1]
                else:
                    parallelStates = [currState for currState in
                                      sorted(currBatchPstar.keys(),
                                             key=lambda x: int(x.split(';')[0])) if
                                      int(currState.split(";")[0]) >= cutOff2]


                parallelStatesChunks = chunks(parallelStates,25000) # usually this should be the batchsize parameter
                del parallelStates


                for chunkStates in parallelStatesChunks:
                    # setting up the pool
                    pool = Pool(12)
                    results = pool.map(func_star,
                                       itertools.izip(chunkStates, itertools.repeat(survivalProb)))
                    pool.close()
                    pool.join()

                    for tempResult in results:
                        finalResults.update(tempResult)

                    del results

                # safe the results
                myShelve = shelve.open(os.path.join(stateDistPath, "%s/P%s" % (t, elem)))
                myShelve.update(finalResults)
                myShelve.close()


                del finalResults
                del parallelStatesChunks
            time.sleep(0.005)

            del currBatchPstar
            del P
    print "Elapsed time for thr forward pass: " + str(time.clock() - startTime)
