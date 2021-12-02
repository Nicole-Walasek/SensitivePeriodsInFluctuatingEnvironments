# approximate the autocorrelation of a markov chain

import numpy as np
import pandas as pd
import math
from MarkovChain import MarkovChain
from multiprocessing import Pool
import itertools
import math



def calcAutocorrelation(markovProbabilities, lag):
    autocorrelationDict = {}
    for markovChain in markovProbabilities:
        pE0E0, pE1E1 = markovChain
        pE0E1 = 1 - pE0E0
        pE1E0 = 1 - pE1E1
        # formula for symmetric markov chains
        autoCorrelation = None
        if pE0E0 == pE1E1:
            autoCorrelation = (2 * pE0E0 - 1) ** lag
            autocorrelationDict[markovChain] = round(autoCorrelation,2)
        else:
            autocorrelationDict[markovChain] = autoCorrelation
    return autocorrelationDict


def calcAutocorrelationSim(markovProbabilities):
    autocorrelationDict = {}
    for markovChain in markovProbabilities:
        pE0E0, pE1E1 = markovChain
        pE0E1 = 1 - pE0E0
        pE1E0 = 1 - pE1E1

        autocorrAvg = 1-(pE0E1+pE1E0)
        autocorrelationDict[markovChain] = round(autocorrAvg,2)

    return autocorrelationDict

