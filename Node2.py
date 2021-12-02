import numpy as np
from scipy.stats import binom
import time
import cPickle as pickle
import os
import shelve
import sys
from decimal import *
from decimal import Decimal as D

getcontext().prec = 10


class Edge():
    def __init__(self, parent, cue):
        self.parent = parent
        self.child = None
        self.cue = cue

    def add_child(self, child):
        self.child = child

    def has_child(self):
        if self.child:
            return True
        else:
            return False

    def get_parent(self):
        return self.parent

    def get_child(self):
        return self.child

    def get_cue(self):
        return self.cue


class Node():
    def __init__(self, pE0, pE1, pC0D, pC1D, x0, x1):
        self.outgoing = []
        self.pE0 = pE0
        self.pE1 = pE1
        self.pC0D = pC0D  # these will be the updated cue validities; weights of the arcs
        self.pC1D = pC1D  # these will be the updated cue validities
        self.x0 = x0
        self.x1 = x1
        self.state = (x0, x1)

    def add_outgoingEdges(self, edge):
        self.outgoing.append(edge)

    def has_edges(self):
        if self.outgoing:
            return True
        else:
            return False

    def get_outgoing(self):
        return self.outgoing

    def get_state(self):
        return self.sate

    def return_last_level(self):
        level = []
        if self.has_edges():
            for edge in self.outgoing:
                if edge.has_child():
                    level += edge.child.return_last_level()
                else:
                    level.append(edge)
        else:
            level.append(self)

        return level


# TODO rewrite this to include changing states



def BayesianUpdating(pE0, pE1, pDE0, pDE1, transition_probability):

    # pE0 is the evolutionary prior vor environment 1
    # pE1 is the evolutionary prior for environment 2
    # pDE0 and pDE1 are the probabilities of obtaining the data given environment 0 or 1 respectively (likelihood)
    p_D = pDE0 * (pE0 * transition_probability['E0']['E0'] + pE1 * transition_probability['E1']['E0']) + pDE1 * (
                pE1 * transition_probability['E1']['E1'] + pE0*transition_probability['E0']['E1'])


    b0_D = (pDE0 * (pE0 * transition_probability['E0']['E0'] + pE1 * transition_probability['E1']['E0'])) / p_D
    b1_D = (pDE1 * (
                pE1 * transition_probability['E1']['E1'] + pE0*transition_probability['E0']['E1'])) / p_D


    return b0_D, b1_D


# traverse the tree:
def traverseTree(tree):
    print "root"
    print str(tree.x0) + " " + str(tree.x1) + " " + str(tree.y0) + " " + str(tree.y1) + " " + str(tree.yw) + "\n"
    if tree.has_edges():
        for edge in tree.get_outgoing():
            # print "\n"
            if edge.has_child():
                child = edge.get_child()
                print "child"
                traverseTree(child)


def is_Edge(obj):
    return isinstance(obj, Edge)


def is_Node(obj):
    return isinstance(obj, Node)


def add_edges(tree, cueSet):
    for node in tree.return_last_level():
        if is_Node(node):
            for cue in cueSet:
                node.add_outgoingEdges(Edge(node, cue))
        else:
            print "Something went wrong, expected node"
            exit(1)


def add_node_layer(tree, pC0E0, pC1E1, pC1E0, pC0E1, tree_dict, cueSet, t, T, edge_counter,
                   transition_probability):


    miniCounter = 0
    for edge in tree.return_last_level():
        if is_Edge(edge):
            prevNode = edge.get_parent()
            old_cueSet = [prevNode.x0, prevNode.x1]
            old_cueSet[edge.get_cue()] += 1

            # need to determine the updated posterior and updated conditional cue reliability
            # in case we see a cue for environment 0


            if edge.get_cue() == 0:
                posE0, posE1 = BayesianUpdating(prevNode.pE0, prevNode.pE1, pC0E0, pC0E1,
                                                transition_probability)
            # in case we see a cue for enviroment 1
            else:  # received a cue for environment 1
                posE0, posE1 = BayesianUpdating(prevNode.pE0, prevNode.pE1, pC1E0, pC1E1,
                                                transition_probability)

            newx0, newx1 = old_cueSet


            # pC0D = pC0E0 * posE0 + pC0E1 * posE1
            # pC1D = pC1E0 * posE0 + pC1E1 * posE1

            superPosE0 = posE0 * transition_probability['E0']['E0'] + posE1 * transition_probability['E1']['E0']
            superPosE1 = posE0 * transition_probability['E0']['E1'] + posE1 * transition_probability['E1']['E1']

            pC0D = pC0E0 * superPosE0 + pC0E1 * superPosE1
            pC1D = pC1E0 * superPosE0 + pC1E1 * superPosE1

            # instead of adding the children to the existing edges, make them new root nodes

            node = Node(posE0, posE1, pC0D, pC1D, newx0, newx1)
            # immediateley set edges
            if t < T:
                add_edges(node, cueSet)
            tree_dict[str(edge_counter * 2 + miniCounter)] = node
            miniCounter += 1
            del node
        else:
            print "Something went wrong, expected edge"
            exit(1)
    return tree_dict


def treeBuildingRoutine(t, T, pC0E0t, cueSet, pC1E1t, tree_path, transition_probability):
    pC0E0 = D(str(pC0E0t[t]))
    pC1E1 = D(str(pC1E1t[t]))

    pC1E0 = D(str(1)) - pC0E0
    pC0E1 = D(str(1)) - pC1E1


    print t
    # initialize new tree dict
    tree_dict = {}
    # first step: load previous tree
    prevTrees = shelve.open(os.path.join(tree_path, "tree%s" % (t - 1)))
    for identifier, prevTree in prevTrees.iteritems():  # scope for parallelization?
        # generate two new trees
        tree_dict = add_node_layer(prevTree, pC0E0, pC1E1, pC1E0, pC0E1, tree_dict, cueSet, t, T,
                                   int(identifier), transition_probability)

    # dump the tree
    currentFile = os.path.join(tree_path, "tree%s" % t)

    myShelve = shelve.open(currentFile)
    myShelve.update(tree_dict)
    myShelve.close()

    # pickle.dump(tree_dict, open(currentFile, "wb"))
    del tree_dict


def buildForwardTree(T, cueSet, pE0, pE1, pC0E0t, pC1E1t, tree_path, transition_probability):
    # check whether tree directory exists; if not creates one
    if not os.path.exists(tree_path):
        os.makedirs(tree_path)

    # initialize the tree
    tValuesForward = np.arange(1, T+1, 1)
    startTime = time.clock()
    pC0E0 = Decimal(str(pC0E0t[1]))  # have to think about this in the long run, should be a zero
    pC1E0 = D(str(1)) - pC0E0

    pC1E1 = Decimal(str(pC1E1t[1]))
    pC0E1 = D(str(1)) - pC1E1

    superPosE0 = pE0 * transition_probability['E0']['E0'] + pE1 * transition_probability['E1']['E0']
    superPosE1 = pE0 * transition_probability['E0']['E1'] + pE1 * transition_probability['E1']['E1']

    pC0D = pC0E0 * pE0 + pC0E1 * pE1
    pC1D = pC1E0 * pE0 + pC1E1 * pE1

    root = Node(pE0, pE1, pC0D, pC1D, 0, 0)  # cue validities don't matter here because we don't get a cue here
    root.add_outgoingEdges(Edge(root, 0))
    root.add_outgoingEdges(Edge(root, 1))
    tree_dict = {}
    tree_dict[str(0)] = root
    currentFile = os.path.join(tree_path, "tree0")

    myShelve = shelve.open(currentFile)
    myShelve.update(tree_dict)
    myShelve.close()

    # free memory psace
    del root
    del tree_dict

    for t in tValuesForward:
        treeBuildingRoutine(t, T, pC0E0t, cueSet, pC1E1t, tree_path,  transition_probability)
    elapsedTime = time.clock() - startTime

    print "elapsed time for bulding the tree: " + str(elapsedTime)
