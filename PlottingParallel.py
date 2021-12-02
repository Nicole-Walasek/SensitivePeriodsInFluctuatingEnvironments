# waht we need: policy and state transition matrix
# combine those two into one ditcionary and read it out into a textfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from PlotCircles import circles
import multiprocessing
import itertools
import os
import random
import operator
from operator import itemgetter
from math import *
# set the current working directory

def markovSim(mc, adultT, startDist):
    resultArr = []

    pE0E0, pE1E1 = mc
    pE0E1 = float(1- pE0E0)
    pE1E0 = float(1- pE1E1)

    P = np.array([[pE0E0,pE0E1],[pE1E0,pE1E1]])
    resultArr.append(round(startDist[1],3))
    for t in np.arange(0,adultT,1):
        newDist = np.dot(startDist,np.linalg.matrix_power(P,(t+1)))
        newDist = np.array(newDist)/float(sum(newDist))
        resultArr.append(round(newDist[1],3))
    return resultArr


def chunks(l, n):
    if n == 0:
        yield l
    else:
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            if isinstance(l, list):
                yield l[i:i+n]
            else:

                yield l.loc[i:i+n-1].reset_index(drop = True)

def convertValues(valueArr, old_max ,old_min,new_max, new_min):
    minArr =old_min
    maxArr = old_max
    rangeArr = maxArr-minArr
    newRangeArr = new_max-new_min
    result = [((val - minArr)/float(rangeArr))*newRangeArr+new_min for val in valueArr]
    return result



def area_calc(probs, r):
    result = [np.sqrt(float(p))*r for p in probs]
    return result

def duplicates(n):
    counter=Counter(n) #{'1': 3, '3': 3, '2': 3}
    dups=[i for i in counter if counter[i]!=1] #['1','3','2']
    result={}
    for item in dups:
        result[item]=[i for i,j in enumerate(n) if j==item]
    return result

# hepler function for plotting the lines
def isReachable(currentIdent, nextIdentList):
    condition_a = currentIdent*2
    condition_b = condition_a+1
    yVals = [idx for idx,item in enumerate(nextIdentList) if (condition_a in item or condition_b in item)]
    yVals = list(set(yVals))
    return yVals


def joinIndidividualResultFiles(argument, tValues, dataPath):
    # need to provide the dataPath accordingly
    if argument == 'raw':

        resultsDFAll =[]
        for t in tValues:
            print 'Currently aggregating data for time step %s' % t
            # batch level
            resultDFList = [batchPstar for batchPstar in os.listdir(os.path.join(dataPath, '%s' % t))]
            resultDFListSorted = [batchPstar for batchPstar in
                                  sorted(resultDFList, key=lambda x: int(x.replace('.csv', '')))]

            # read and concatenate all csv file for one time step
            resultsDF = pd.concat(
                [pd.read_csv(os.path.join(dataPath, os.path.join('%s' % t, f)),index_col=0).reset_index(drop = True) for f in resultDFListSorted]).reset_index(drop = True)
            resultsDFAll.append(resultsDF)
        finalData = pd.concat(resultsDFAll).reset_index(drop = True)
        finalData.to_csv('finalRaw.csv')

        # # next get that df for the fitness functions
        # resultsDFAll = []
        # T = max(tValues)+1
        # print 'Currently aggregating data for time step %s' % T
        # # batch level
        # resultDFList = [batchPstar for batchPstar in os.listdir(os.path.join(dataPath, '%s' % T))]
        # resultDFListSorted = [batchPstar for batchPstar in
        #                       sorted(resultDFList, key=lambda x: int(x.replace('.csv', '')))]
        #
        # # read and concatenate all csv file for one time step
        # resultsDF = pd.concat(
        #     [pd.read_csv(os.path.join(dataPath, os.path.join('%s' % T, f)), index_col=0).reset_index(drop=True) for
        #      f in resultDFListSorted]).reset_index(drop=True)
        # resultsDFAll.append(resultsDF)
        # finalData = pd.concat(resultsDFAll).reset_index(drop=True)
        # finalData.to_csv('finalPhenotypes_Fitness.csv')


    elif argument == "aggregated":
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalAggregated.csv')
    elif argument == 'plotting':
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalPlotting.csv')

    else:
        print "Wrong argument"

# function for parallelization
def plotLinesCopy(subDF1Identifiers, nextIdentList):
    yvalsIDXAll = []
    if subDF1Identifiers:
        for identSubDF1 in subDF1Identifiers:  # as many lines as unique cue validities
            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
            subList.sort()
            subList2 = list(subList for subList, _ in itertools.groupby(subList))
            yvalsIDXAll.append(subList2)
            del subList
            del subList2
    return yvalsIDXAll

def plotLines(identSubDF1, nextIdentList):
    subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
    subList.sort()
    subList2 = list(subList for subList, _ in itertools.groupby(subList))
    del subList
    return subList2

def func_star(allArgs):
    return plotLines(*allArgs)


def cleanIdentifiers(oldIdentifiers):
    newIdent = [str(ident).replace('[', '').replace(']', '').split(',') for ident in oldIdentifiers]
    newIdent2 = [[int(str(a).replace('.0', '')) for a in subList] for subList in newIdent]

    return newIdent2


def policyPlotReducedOverview2(r, minProb, lines, cueValidityArr, T, adultTArr, levelsAutoCorrToPlot, autoCorrDict,
                               twinResultsAggregatedPath, mainPath, argumentR, argumentP, adulthood, useNames):
    ontT = T
    T = T + 1
    tValues = np.arange(1, T, 1)
    # first open the dictionary containing the results
    autoCorrDict_sorted = sorted(autoCorrDict.items(), key=operator.itemgetter(1))
    autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)

    """
    prepare autocorrelation values and columsn for plotting
    """
    # the next line find the indices of the closest autocorrelation values that match the user input
    idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
           levelsAutoCorrToPlot]
    autoCorrValSubset = np.array(autoCorrVal)[idx]
    autoCorrKeysDict = {autoCorrVal[idxx]: autoCorrKeys[idxx] for idxx in idx}
    nameDict = {autoCorrValSubset[0]: "low", autoCorrValSubset[1]: "moderate", autoCorrValSubset[2]: "high"}

    rowVec = []
    for currX in adultTArr:
        for currY in autoCorrValSubset:
            rowVec.append((currX, currY))

    fig, axes = plt.subplots(len(cueValidityArr), len(levelsAutoCorrToPlot) * len(adultTArr), sharex=True,
                             sharey=True)  # we will always plot three autocorrelation levels

    if len(adultTArr) == 3:
        fig.set_size_inches(30, 10)
    elif len(adultTArr) == 1:
        fig.set_size_inches(16, 16)
    else:
        fig.set_size_inches(32, 16)

    ax_list = fig.axes
    fig.set_facecolor("white")

    jx = 0
    for adultT, autoCorr in rowVec:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * (len(levelsAutoCorrToPlot) * len(adultTArr)) + jx]
            ax.set(aspect='equal')
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            pE0E0, pE1E1 = autoCorrKeysDict[autoCorr]
            dataPath1 = os.path.join(mainPath,
                                     '%s/runTest_%s%s_%s%s_%s' % (
                                         adultT, argumentR[0], argumentP[0], pE0E0, pE1E1, cueVal))
            dataPath2 = 'plotting/aggregatedResults'

            dataPath = os.path.join(dataPath1, dataPath2)
            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            if adulthood:
                convertVal = ontT + max(adultTArr)
            else:
                convertVal = ontT

            for t in tValues:
                # here is where the relevant files are loaded

                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t))
                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, convertVal, 1)
                aggregatedResultsDF = aggregatedResultsDF[
                    aggregatedResultsDF.stateProb > minProb]  # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] == t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t, key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])

                color_palette = {0: '#be0119', 1: '#448ee4', 2: '#000000', 3: '#98568d', 4: '#548d44', -1: '#d8dcd6'}
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])
                area = area_calc(aggregatedResultsDF['stateProb'], r)

                # now plot the developmental trajectories
                circles(aggregatedResultsDF['time'], aggregatedResultsDF['newpE1'], s=area, ax=ax, c=colors, alpha =0.5, zorder=2,
                        lw=0.05, ec = colors)


                if adulthood:
                    if t == ontT:
                        lastPE1 = aggregatedResultsDF['pE1']
                        lastArea = aggregatedResultsDF['stateProb']

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.clock()
                for t in np.arange(0, T - 1, 1):
                    print "Current time step: %s" % t
                    tNext = t + 1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t + 1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, convertVal, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)
                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time == 1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, convertVal, 1)
                        subDF2.reset_index(drop=True, inplace=True)

                        subDF2.drop(index=indices, inplace=True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    subDF1['Identifier'] = cleanIdentifiers(subDF1.Identifier)
                    subDF2['Identifier'] = cleanIdentifiers(subDF2.Identifier)
                    nextIdentList = subDF2['Identifier']

                    yvalsIDXAll = []
                    if t <= 11:  # otherwise the overhead for multiprocessing is slowing down the computation
                        for identSubDF1 in list(subDF1.Identifier):
                            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
                            subList.sort()
                            subList2 = list(subList for subList, _ in itertools.groupby(subList))
                            yvalsIDXAll.append(subList2)
                            del subList
                            del subList2
                    else:

                        for identSubDF1 in list(subDF1.Identifier):  # as many lines as unique cue validities
                            pool = multiprocessing.Pool(processes=32)
                            results = pool.map(func_star, itertools.izip(chunks(identSubDF1, 1000),
                                                                         itertools.repeat(nextIdentList)))
                            pool.close()
                            pool.join()
                            resultsUnchained = [item for sublist in results for item in sublist]
                            yvalsIDXAll.append(resultsUnchained)
                            del results
                            del resultsUnchained

                    # process the results
                    yArr = []
                    yArr = []
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                itertools.chain.from_iterable(yvalsIDXAll[subIDX])]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#dedacb', zorder=1, lw=0.3) for yArrr in
                         yArr]
                    del yArr
                elapsedTime = time.clock() - startTime
                print "Elapsed time plotting the lines: " + str(elapsedTime)
            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly
            radii = []
            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i) / sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR = np.sqrt(sum(stateProbPies[idx])) * r
                radii.append(currentR)
                pp, tt = ax.pie(pieFracs, colors=colorsPies, radius=currentR, center=coordinates[idx],
                                wedgeprops={'linewidth': 0.0, "edgecolor": "k", 'alpha':0.5 })
                [p.set_zorder(3 + len(coordinates) - idx) for p in pp]

            """ add adulthood here

            first only plot the circles and then plot the associated lines

            """
            if adulthood:
                markovChain = (pE0E0, pE1E1)

                adultTimeVec = []
                adultPos = []
                stateProbs = []
                for idxCurr, pE1 in enumerate(lastPE1):
                    stateProb = lastArea[idxCurr]
                    # first get data
                    results = markovSim(markovChain, adultT, [1 - pE1, pE1])
                    adultPos += results[1:]
                    [adultTimeVec.append(adultTcurr + ontT) for adultTcurr in np.arange(1, adultT + 1)]
                    [stateProbs.append(stateProb) for x in np.arange(adultT)]
                    # also draw the lines already:
                    xVec = np.arange(ontT, (ontT + adultT) + 1, 1)
                    yVec = convertValues(results, 1, 0, convertVal, 1)

                    ax.plot(xVec, yVec, ls='solid', marker=" ", color='#dbaea3', zorder=1, lw=0.3)

                # now plot the adult circles
                area = area_calc(stateProbs, r)

                posVecNewAdult = convertValues(adultPos, 1, 0, convertVal, 1)

                circles(adultTimeVec, posVecNewAdult, s=area, ax=ax, c='grey', zorder=2, lw=0.5)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.sca(ax)
            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (convertVal + 1) / float(2)
            yLabels = convertValues([1, midPoint, convertVal], convertVal, 1, 1, 0)
            # removing frame around the plot
            plt.ylim(0.4, convertVal + 0.5)
            plt.xlim(-0.6, convertVal + 0.5)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                # ax.spines['left'].set_visible(True)
                # ax.spines['bottom'].set_visible(True)

                plt.xlabel('ontogeny', fontsize=25, labelpad=10)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([])
                plt.yticks([1, midPoint, convertVal], yLabels)  # this doesn't make sense

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.yticks([1, midPoint, convertVal], yLabels, fontsize=20)

            if jx == (len(autoCorrValSubset) * len(adultTArr)) - 1:
                plt.ylabel(str(cueVal), fontsize=25, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            if ix == 0:
                if useNames:
                    plt.title("%s" % (nameDict[autoCorr]), fontsize=25, pad=15)
                else:
                    plt.title("%s" % round(autoCorr, 1), fontsize=25, pad=15)

            if (jx + 1) % len(autoCorrValSubset) == 0 and not jx == (len(adultTArr) * len(autoCorrValSubset)) - 1:
                paramVLine = convertVal + 1.75
                ax.vlines([paramVLine], -0.03, 1.03, transform=ax.get_xaxis_transform(), color='black', lw=2,
                          clip_on=False)

            ix += 1
        jx += 1

    if len(adultTArr) == 3:
        top = 0.7
        fig.text(0.95, 0.38, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        fig.text(0.08, 0.38, r'$P(E_1|D)$', fontsize=25, horizontalalignment='left', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        autoCorrCoord = 0.85

    elif len(adultTArr) == 1:
        top = 0.8
        fig.text(0.98, 0.42, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        fig.text(0.05, 0.42, r'$P(E_1|D)$', fontsize=25, horizontalalignment='left', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        autoCorrCoord = 0.9
    else:
        top = 0.8
        fig.text(0.95, 0.42, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        fig.text(0.08, 0.42, r'$P(E_1|D)$', fontsize=25, horizontalalignment='left', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        autoCorrCoord = 0.9

    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.05, top=top)

    fig.text(0.514, 0.95, 'adult life span', fontsize=25, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(adultTArr)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)

    for figCoord, adultT in zip(figVals, adultTArr):

        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.065
            else:
                figCoordF = figCoord - 0.035
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.085
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.514
            else:
                figCoordF = figCoord - 0.055
        fig.text(figCoordF, autoCorrCoord, '%s\n\nautocorrelation' % adultT, fontsize=25, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

        plt.savefig(os.path.join(twinResultsAggregatedPath, 'DevelopmentalTrajectoryReduced.png'), bbox_inches='tight',
                    dpi=600)


def policyPlotReducedMerge(r, minProb, lines, cueValidityArr, T, adultT, levelsAutoCorrToPlot, autoCorrDict,
                               twinResultsAggregatedPath, dataMainPath, argumentR, argumentP, adulthood, useNames, mergeArr):
    ontT = T
    T = T + 1
    tValues = np.arange(1, T, 1)


    """
    prepare autocorrelation values and columsn for plotting
    """

    nameList = ["low", "moderate", "high"]

    rowVec = []
    for currX in mergeArr:
        for currY in levelsAutoCorrToPlot:
            rowVec.append((currX, currY))

    fig, axes = plt.subplots(len(cueValidityArr), len(rowVec), sharex=True,
                             sharey=True)  # we will always plot three autocorrelation levels

    fig.set_size_inches(32, 16)

    ax_list = fig.axes
    fig.set_facecolor("white")

    jx = 0
    for symmArg, autoCorrCurr in rowVec:  # one column per adultT

        """
        select the correct data
        """
        # first open the dictionary containing the results
        autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        # the next line find the indices of the closest autocorrelation values that match the user input
        idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - autoCorrCurr))]
        autoCorrValSubset = np.array(autoCorrVal)[idx]
        autoCorr = autoCorrValSubset[0]
        autoCorrKeysDict = {autoCorrVal[idxx]: autoCorrKeys[idxx] for idxx in idx}
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect='equal')
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            pE0E0, pE1E1 = autoCorrKeysDict[autoCorr]

            mainPath = os.path.join(dataMainPath, str(symmArg))

            dataPath1 = os.path.join(mainPath,
                                     '%s/runTest_%s%s_%s%s_%s' % (
                                         adultT, argumentR[0], argumentP[0], pE0E0, pE1E1, cueVal))
            dataPath2 = 'plotting/aggregatedResults'

            dataPath = os.path.join(dataPath1, dataPath2)
            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            if adulthood:
                convertVal = ontT + adultT
            else:
                convertVal = ontT

            for t in tValues:
                # here is where the relevant files are loaded

                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t))
                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, convertVal, 1)
                aggregatedResultsDF = aggregatedResultsDF[
                    aggregatedResultsDF.stateProb > minProb]  # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] == t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t, key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])

                color_palette = {0: '#be0119', 1: '#448ee4', 2: '#000000', 3: '#98568d', 4: '#548d44', -1: '#d8dcd6'}
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])
                area = area_calc(aggregatedResultsDF['stateProb'], r)

                # now plot the developmental trajectories
                
                circles(aggregatedResultsDF['time'], aggregatedResultsDF['newpE1'], s=area, ax=ax, c=colors,alpha =0.5, zorder=2,
                        lw=0.05, ec = colors)
                if adulthood:
                    if t == ontT:
                        lastPE1 = aggregatedResultsDF['pE1']
                        lastArea = aggregatedResultsDF['stateProb']

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.clock()
                for t in np.arange(0, T - 1, 1):
                    print "Current time step: %s" % t
                    tNext = t + 1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t + 1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, convertVal, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)
                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time == 1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, convertVal, 1)
                        subDF2.reset_index(drop=True, inplace=True)

                        subDF2.drop(index=indices, inplace=True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    subDF1['Identifier'] = cleanIdentifiers(subDF1.Identifier)
                    subDF2['Identifier'] = cleanIdentifiers(subDF2.Identifier)
                    nextIdentList = subDF2['Identifier']

                    yvalsIDXAll = []
                    if t <= 11:  # otherwise the overhead for multiprocessing is slowing down the computation
                        for identSubDF1 in list(subDF1.Identifier):
                            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
                            subList.sort()
                            subList2 = list(subList for subList, _ in itertools.groupby(subList))
                            yvalsIDXAll.append(subList2)
                            del subList
                            del subList2
                    else:

                        for identSubDF1 in list(subDF1.Identifier):  # as many lines as unique cue validities
                            pool = multiprocessing.Pool(processes=32)
                            results = pool.map(func_star, itertools.izip(chunks(identSubDF1, 1000),
                                                                         itertools.repeat(nextIdentList)))
                            pool.close()
                            pool.join()
                            resultsUnchained = [item for sublist in results for item in sublist]
                            yvalsIDXAll.append(resultsUnchained)
                            del results
                            del resultsUnchained

                    # process the results
                    yArr = []
                    yArr = []
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                itertools.chain.from_iterable(yvalsIDXAll[subIDX])]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#dedacb', zorder=1, lw=0.3) for yArrr in
                         yArr] # #f7f3e2
                    del yArr
                elapsedTime = time.clock() - startTime
                print "Elapsed time plotting the lines: " + str(elapsedTime)


            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly
            radii = []
            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i) / sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR = np.sqrt(sum(stateProbPies[idx])) * r
                radii.append(currentR)
                pp, tt = ax.pie(pieFracs, colors=colorsPies, radius=currentR, center=coordinates[idx],
                                wedgeprops={'linewidth': 0.0, "edgecolor": "k", 'alpha':0.5 })
                [p.set_zorder(3 + len(coordinates) - idx) for p in pp]

            """ add adulthood here

            first only plot the circles and then plot the associated lines

            """
            if adulthood:
                markovChain = (pE0E0, pE1E1)

                adultTimeVec = []
                adultPos = []
                stateProbs = []
                for idxCurr, pE1 in enumerate(lastPE1):
                    stateProb = lastArea[idxCurr]
                    # first get data
                    results = markovSim(markovChain, adultT, [1 - pE1, pE1])
                    adultPos += results[1:]
                    [adultTimeVec.append(adultTcurr + ontT) for adultTcurr in np.arange(1, adultT + 1)]
                    [stateProbs.append(stateProb) for x in np.arange(adultT)]
                    # also draw the lines already:
                    xVec = np.arange(ontT, (ontT + adultT) + 1, 1)
                    yVec = convertValues(results, 1, 0, convertVal, 1)

                    ax.plot(xVec, yVec, ls='solid', marker=" ", color='#dbaea3', zorder=1, lw=0.3)

                # now plot the adult circles
                area = area_calc(stateProbs, r)

                posVecNewAdult = convertValues(adultPos, 1, 0, convertVal, 1)

                circles(adultTimeVec, posVecNewAdult, s=area, ax=ax, c='grey', zorder=2, lw=0.5)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (convertVal + 1) / float(2)
            yLabels = convertValues([1, midPoint, convertVal], convertVal, 1, 1, 0)
            # removing frame around the plot
            plt.ylim(0.4, convertVal + 0.5)
            plt.xlim(-0.6, convertVal + 0.5)


            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)


                plt.xlabel('ontogeny', fontsize=25, labelpad=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([])
                plt.yticks([1, midPoint, convertVal], yLabels)  # this doesn't make sense

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                plt.yticks([1, midPoint, convertVal], yLabels, fontsize=20)

            if jx == len(rowVec) - 1:
                plt.ylabel(str(cueVal), fontsize=25, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            if ix == 0:
                if useNames:
                    if autoCorr <= 0.3:
                        autoCorrIDX = 0
                    elif autoCorr > 0.3 and autoCorr <= 0.6:
                        autoCorrIDX = 1
                    else:
                        autoCorrIDX = 2
                    plt.title("%s" % (nameList[autoCorrIDX]), fontsize=25, pad=15)
                else:
                    plt.title("%s" % round(autoCorr,1), fontsize=25, pad=15)

            # if (jx + 1) % len(levelsAutoCorrToPlot) == 0 and not jx == len(rowVec) - 1:
            #     paramVLine = convertVal + 1.5
            #     ax.vlines([paramVLine], -0.03, 1.03, transform=ax.get_xaxis_transform(), color='black', lw=2,
            #               clip_on=False)

            ix += 1
        jx += 1



    top = 0.8
    fig.text(0.94, 0.42, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.08, 0.42, r'$P(E_1|D)$', fontsize=25, horizontalalignment='left', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.9

    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.05, top=top)

    fig.text(0.514, 0.95, 'adult life span = %s' %adultT, fontsize=25, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(mergeArr)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    newMergeArr = []
    for elem in mergeArr:
        if "(E0)" in elem:
            elem = elem.replace("(E0)","")
        elif "(E1)" in elem:
            elem = elem.replace("(E1)", "")
        newMergeArr.append(elem)
    for figCoord, adultT in zip(figVals, newMergeArr):

        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.065
            else:
                figCoordF = figCoord - 0.035
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.085
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.514
            else:
                figCoordF = figCoord - 0.055
        fig.text(figCoordF, autoCorrCoord, '%s transition probabilities\n\nautocorrelation' % adultT, fontsize=25, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')


        # draw rectangles around group of subplots
        rectLeft = plt.Rectangle(
            # (lower-left corner), width, height
            (0.07, 0.015), 0.435, 0.88, fill=False, color="k", lw=2,
            zorder=1000, transform=fig.transFigure, figure=fig
        )
        fig.patches.extend([rectLeft])

        rectRight = plt.Rectangle(
            # (lower-left corner), width, height
            (0.515, 0.015), 0.435, 0.88, fill=False, color="k", lw=2,
            zorder=1000, transform=fig.transFigure, figure=fig
        )
        fig.patches.extend([rectRight])

        # plt.savefig(os.path.join(twinResultsAggregatedPath,'DevelopmentalTrajectoryReduced_%s.png' %(startEnv)), dpi = 400)
        plt.savefig(os.path.join(twinResultsAggregatedPath, 'DevelopmentalTrajectoryReducedMergeTest.png'), bbox_inches='tight' ,dpi=600)


def policyPlotReducedMergeBW(r, minProb, lines, cueValidityArr, T, adultT, levelsAutoCorrToPlot, autoCorrDict,
                               twinResultsAggregatedPath, dataMainPath, argumentR, argumentP, adulthood, useNames, mergeArr):
    ontT = T-1
    #T = T + 1
    tValues = np.arange(1, T, 1)


    """
    prepare autocorrelation values and columsn for plotting
    """

    nameList = ["low", "moderate", "high"]

    rowVec = []
    for currX in mergeArr:
        for currY in levelsAutoCorrToPlot:
            rowVec.append((currX, currY))

    fig, axes = plt.subplots(len(cueValidityArr), len(rowVec), sharex=False,
                             sharey=True)  # we will always plot three autocorrelation levels

    fig.set_size_inches(32, 16)

    ax_list = fig.axes
    fig.set_facecolor("white")

    jx = 0
    for symmArg, autoCorrCurr in rowVec:  # one column per adultT

        """
        select the correct data
        """
        # first open the dictionary containing the results
        autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        # the next line find the indices of the closest autocorrelation values that match the user input
        idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - autoCorrCurr))]
        autoCorrValSubset = np.array(autoCorrVal)[idx]
        autoCorr = autoCorrValSubset[0]
        autoCorrKeysDict = {autoCorrVal[idxx]: autoCorrKeys[idxx] for idxx in idx}
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect='equal')
            plt.sca(ax)

            # get the data for this adult T and a specific cue reliability value
            pE0E0, pE1E1 = autoCorrKeysDict[autoCorr]

            mainPath = os.path.join(dataMainPath, str(symmArg))

            dataPath1 = os.path.join(mainPath,
                                     '%s/runTest_%s%s_%s%s_%s' % (
                                         adultT, argumentR[0], argumentP[0], pE0E0, pE1E1, cueVal))
            dataPath2 = 'plotting/aggregatedResults'

            dataPath = os.path.join(dataPath1, dataPath2)
            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            if adulthood:
                convertVal = ontT + adultT
            else:
                convertVal = ontT

            prior = 0
            for t in tValues:
                # here is where the relevant files are loaded

                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t))
                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, convertVal, 1)
                if t == 1:
                    prior = aggregatedResultsDF[aggregatedResultsDF['time'] == 0]['newpE1'][1]

                aggregatedResultsDF = aggregatedResultsDF[
                    aggregatedResultsDF.stateProb > minProb]  # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] == t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t, key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])

                colors = '#757575'#'black'
                area = area_calc(aggregatedResultsDF['stateProb'], r)

                # now plot the developmental trajectories
                circles(aggregatedResultsDF['time'], aggregatedResultsDF['newpE1'], s=area, ax=ax, c=colors, alpha =0.5, zorder=2,
                        lw=0.05, ec = colors)
                if adulthood:
                    if t == ontT:
                        lastPE1 = aggregatedResultsDF['pE1']
                        lastArea = aggregatedResultsDF['stateProb']

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.clock()
                for t in np.arange(0, T - 1, 1):
                    print "Current time step: %s" % t
                    tNext = t + 1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t + 1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, convertVal, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)
                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time == 1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, convertVal, 1)
                        subDF2.reset_index(drop=True, inplace=True)

                        subDF2.drop(index=indices, inplace=True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    subDF1['Identifier'] = cleanIdentifiers(subDF1.Identifier)
                    subDF2['Identifier'] = cleanIdentifiers(subDF2.Identifier)
                    nextIdentList = subDF2['Identifier']

                    yvalsIDXAll = []
                    if t <= 11:  # otherwise the overhead for multiprocessing is slowing down the computation
                        for identSubDF1 in list(subDF1.Identifier):
                            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
                            subList.sort()
                            subList2 = list(subList for subList, _ in itertools.groupby(subList))
                            yvalsIDXAll.append(subList2)
                            del subList
                            del subList2
                    else:

                        for identSubDF1 in list(subDF1.Identifier):  # as many lines as unique cue validities
                            pool = multiprocessing.Pool(processes=32)
                            results = pool.map(func_star, itertools.izip(chunks(identSubDF1, 1000),
                                                                         itertools.repeat(nextIdentList)))
                            pool.close()
                            pool.join()
                            resultsUnchained = [item for sublist in results for item in sublist]
                            yvalsIDXAll.append(resultsUnchained)
                            del results
                            del resultsUnchained

                    # process the results
                    yArr = []
                    yArr = []
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                itertools.chain.from_iterable(yvalsIDXAll[subIDX])]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#dedacb', zorder=1, lw=0.3) for yArrr in
                         yArr]
                    del yArr
                elapsedTime = time.clock() - startTime
                print "Elapsed time plotting the lines: " + str(elapsedTime)

            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly
            radii = []
            for idx in range(len(coordinates)):
                colorsPies = [colors for idj in decisionsPies[idx]]

                pieFracs = [float(i) / sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR = np.sqrt(sum(stateProbPies[idx])) * r
                radii.append(currentR)
                pp, tt = ax.pie(pieFracs, colors=colorsPies, radius=currentR, center=coordinates[idx],
                                wedgeprops={'linewidth': 0.0, "edgecolor": colors,  'alpha':0.5 })
                [p.set_zorder(3 + len(coordinates) - idx) for p in pp]

            """ add adulthood here

            first only plot the circles and then plot the associated lines

            """


            # adding the prior lines
            plt.plot(np.arange(-1.5, ontT +adultT +2.5, 1), [prior] * len(np.arange(-1.5,  ontT +adultT +2.5, 1)),
                     zorder=2, color='black', lw=2.5, ls='solid')

            if adulthood:
                colors = '#757575'
                markovChain = (pE0E0, pE1E1)

                adultTimeVec = []
                adultPos = []
                stateProbs = []
                for idxCurr, pE1 in enumerate(lastPE1):
                    stateProb = lastArea[idxCurr]
                    # first get data
                    results = markovSim(markovChain, adultT, [1 - pE1, pE1])
                    adultPos += results[1:]
                    [adultTimeVec.append(adultTcurr + ontT) for adultTcurr in np.arange(1, adultT + 1)]
                    [stateProbs.append(stateProb) for x in np.arange(adultT)]
                    # also draw the lines already:
                    xVec = np.arange(ontT, (ontT + adultT) + 1, 1)
                    yVec = convertValues(results, 1, 0, convertVal, 1)

                    ax.plot(xVec, yVec, ls='solid', marker=" ", color='#dedacb', zorder=1, lw=1.5) #D3D3D3

                # now plot the adult circles
                area = area_calc(stateProbs, r)

                posVecNewAdult = convertValues(adultPos, 1, 0, convertVal, 1)

                circles(adultTimeVec, posVecNewAdult, s=area, ax=ax, c=colors, zorder=3, lw=0.5,alpha = 0.5)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (convertVal + 1) / float(2)
            yLabels = convertValues([1, midPoint, convertVal], convertVal, 1, 1, 0)
            # removing frame around the plot
            plt.ylim(0, convertVal + 1.2)
            plt.xlim(-1.5, convertVal + 1)


            # plot lines for redability

            # first vertical lines; increase the size
            colorLines = 'black'
            plt.plot([ontT] * len(np.arange(0, convertVal + 1.2, 0.1)), np.arange(0, convertVal + 1.2, 0.1),
                     zorder=-1, color=colorLines, lw=1.5, ls='-')

            plt.plot([ontT + 1] * len(np.arange(1, convertVal + 0.2, 0.1)), np.arange(1, convertVal + 0.2, 0.1),
                     zorder=-1, color=colorLines, lw=2.5, ls='--')

            plt.plot([ontT+5]*len(np.arange(1, convertVal + 0.2, 0.1)), np.arange(1, convertVal + 0.2, 0.1),zorder = -1, color = colorLines, lw = 2.5,ls ='--')
            # plt.plot([ontT + 10] * len(np.arange(1, convertVal + 0.2, 0.1)), np.arange(1, convertVal + 0.2, 0.1),
            #          zorder=-1, color=colorLines, lw=2.5, ls='--')
            plt.plot([ontT + 20] * len(np.arange(1, convertVal + 0.2, 0.1)), np.arange(1, convertVal + 0.2, 0.1),
                     zorder=-1, color=colorLines, lw=2.5, ls='--')



            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                plt.xlabel('ontogeny & adulthood', fontsize=25, labelpad=15)
                plt.xticks([11, 15, 30], [1, 5,20], fontsize=20)
            else:
                plt.xticks([])
                ax.get_xaxis().set_visible(False)

            if ix == len(cueValidityArr) - 1:
                plt.yticks([1, midPoint, convertVal], yLabels)  # this doesn't make sense


            if jx == 0:
                plt.yticks([1, midPoint, convertVal], yLabels, fontsize=20)

            if jx == len(rowVec) - 1:
                plt.ylabel(str(cueVal), fontsize=25, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            if ix == 0:
                if useNames:
                    if autoCorr <= 0.3:
                        autoCorrIDX = 0
                    elif autoCorr > 0.3 and autoCorr <= 0.6:
                        autoCorrIDX = 1
                    else:
                        autoCorrIDX = 2
                    plt.title("%s" % (nameList[autoCorrIDX]), fontsize=25, pad=15)
                else:
                    plt.title("%s" % round(autoCorr,1), fontsize=25, pad=15)



            # if (jx + 1) % len(levelsAutoCorrToPlot) == 0 and not jx == len(rowVec) - 1:
            #     paramVLine = convertVal + 5.5
            #     ax.vlines([paramVLine], -0.125, 1.125, transform=ax.get_xaxis_transform(), color='black', lw=2,
            #               clip_on=False)

            ix += 1
        jx += 1



    top = 0.8
    fig.text(0.94, 0.42, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.08, 0.42, r'$P(E_1|D)$', fontsize=25, horizontalalignment='left', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.9

    plt.subplots_adjust(wspace=0.3, hspace=0.2, bottom=0.075, top=top)


    figVal = 1 / float((len(mergeArr)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    newMergeArr = []
    for elem in mergeArr:
        if "(E0)" in elem:
            elem = elem.replace("(E0)","")
        elif "(E1)" in elem:
            elem = elem.replace("(E1)", "")
        newMergeArr.append(elem)
    for figCoord, adultT in zip(figVals, newMergeArr):

        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.065
            else:
                figCoordF = figCoord - 0.035
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.085
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.514
            else:
                figCoordF = figCoord - 0.055
        fig.text(figCoordF, autoCorrCoord, '%s transition probabilities\n\nautocorrelation' % adultT, fontsize=25, horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')


        # draw rectangles around group of subplots
        rectLeft = plt.Rectangle(
            # (lower-left corner), width, height
            (0.07, 0.015), 0.435, 0.88, fill=False, color="k", lw=2,
            zorder=1000, transform=fig.transFigure, figure=fig
        )
        fig.patches.extend([rectLeft])

        rectRight = plt.Rectangle(
            # (lower-left corner), width, height
            (0.515, 0.015), 0.435, 0.88, fill=False, color="k", lw=2,
            zorder=1000, transform=fig.transFigure, figure=fig
        )
        fig.patches.extend([rectRight])


        # plt.savefig(os.path.join(twinResultsAggregatedPath,'DevelopmentalTrajectoryReduced_%s.png' %(startEnv)), dpi = 400)
        plt.savefig(os.path.join(twinResultsAggregatedPath, 'DevelopmentalTrajectoryReducedMergeBWV3.png'),bbox_inches='tight',dpi=800)




def policyPlotReduced(T,r,markovProbabilities,pC0E0Arr, tValues, dataPath, lines, argumentR, argumentP, minProb,mainPath, plottingPath):
    # preparing the subplot
    fig, axes = plt.subplots(len(pC0E0Arr), len(markovProbabilities), sharex= True, sharey= True)
    fig.set_size_inches(16, 16)
    fig.set_facecolor("white")
    ax_list = fig.axes

    # looping over the paramter space
    iX = 0
    for cueVal in pC0E0Arr: # for each cue validity
        jX = 0
        for pE0E0, pE1E1 in markovProbabilities: # for each prior
            # set the working directory for the current parameter combination
            os.chdir(os.path.join(mainPath,"runTest_%s%s_%s%s_%s" % (argumentR[0], argumentP[0], pE0E0,pE1E1, cueVal)))
            ax = ax_list[iX * len(markovProbabilities) + jX]
            plt.sca(ax)

            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            for t in tValues:
                # here is where the relevant files are loaded
                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' %t))
                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, T - 1, 1)
                aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.stateProb >minProb] # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] ==t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t,key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])


                color_palette = {0:'#be0119', 1:'#448ee4', 2:'#000000', 3: '#98568d', 4: '#548d44', -1: '#d8dcd6'}
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])
                area = area_calc(aggregatedResultsDF['stateProb'], r)

                # now plot the developmental trajectories
                circles(aggregatedResultsDF['time'],aggregatedResultsDF['newpE1'], s =area, ax = ax,c = colors, alpha =0.5, zorder=2,
                        lw=0.05, ec = colors)

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.clock()
                for t in np.arange(0,T-1,1):
                    print "Current time step: %s" % t
                    tNext = t+1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t+1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, T - 1, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)
                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time ==1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop = True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' %tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, T - 1, 1)
                        subDF2.reset_index(drop=True, inplace= True)

                        subDF2.drop(index = indices, inplace= True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    subDF1['Identifier'] = cleanIdentifiers(subDF1.Identifier)
                    subDF2['Identifier'] = cleanIdentifiers(subDF2.Identifier)
                    nextIdentList = subDF2['Identifier']

                    yvalsIDXAll = []
                    if t <= 11: # otherwise the overhead for multiprocessing is slowing down the computation
                        for identSubDF1 in list(subDF1.Identifier):
                            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
                            subList.sort()
                            subList2 = list(subList for subList, _ in itertools.groupby(subList))
                            yvalsIDXAll.append(subList2)
                            del subList
                            del subList2
                    else:

                        for identSubDF1 in list(subDF1.Identifier):  # as many lines as unique cue validities
                            pool = multiprocessing.Pool(processes=32)
                            results = pool.map(func_star, itertools.izip(chunks(identSubDF1, 1000), itertools.repeat(nextIdentList)))
                            pool.close()
                            pool.join()
                            resultsUnchained = [item for sublist in results for item in sublist]
                            yvalsIDXAll.append(resultsUnchained)
                            del results
                            del resultsUnchained

                        # process the results
                    yArr = []
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                itertools.chain.from_iterable(yvalsIDXAll[subIDX])]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#dedacb', zorder=1, lw=0.3) for yArrr in
                         yArr]
                    del yArr
                elapsedTime = time.clock()-startTime
                print "Elapsed time plotting the lines: " + str(elapsedTime)
            #
            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly
            xTuple = [current[0] for current in coordinates]
            yTuple = [current[1] for current in coordinates]
            radii = []
            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i)/sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR= np.sqrt(sum(stateProbPies[idx]))*r
                radii.append(currentR)
                pp,tt = ax.pie(pieFracs,colors = colorsPies, radius = currentR ,center = coordinates[idx], wedgeprops= {'linewidth':0.0, "edgecolor":"k",  'alpha':0.5 })
                [p.set_zorder(3+len(coordinates)-idx) for p in pp]

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (T) / float(2)
            yLabels = convertValues([1, midPoint, T - 1], T - 1, 1, 1, 0)
            # removing frame around the plot
            plt.ylim(0.4, T - 1 + 0.5)
            plt.xlim(-0.6, T - 1 + 0.5)

            if iX == 0:
                plt.title("%s, %s" % (pE0E0, pE1E1), fontsize=20)

            if iX == len(pC0E0Arr) - 1:
                plt.xticks([])
                plt.yticks([1, midPoint, T - 1], yLabels)  # this doesn't make sense

            else:
                ax.get_xaxis().set_visible(False)
            if jX == 0:
                plt.yticks([1, midPoint, T - 1], yLabels, fontsize=15)
                plt.ylabel(r'$P(E_1|D)$', fontsize=20, labelpad=10)

            if jX == len(markovProbabilities) - 1:
                plt.ylabel(str(cueVal), fontsize=20, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            ax.set_aspect('equal')
            jX += 1
        iX += 1
        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        resultPath = os.path.join(mainPath, plottingPath)
        plt.savefig(os.path.join(resultPath, 'DevelopmentalTrajectoryReduced.png'), dpi=400)
        # plt.sav

















