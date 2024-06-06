# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Graph Utilities
=========================================================================
This file contains a host of utility functions for NetworkX graphs.

.. autofunction:: generate_K_string_matrix
.. autofunction:: generate_edges
.. autofunction:: retrieve_rate_matrix
.. autofunction:: add_node_attribute
.. autofunction:: add_graph_attribute
.. autofunction:: find_all_unique_cycles
.. autofunction:: get_ccw_cycle
"""


import numpy as np
import networkx as nx

from kda.exceptions import CycleError


def generate_edges(G, K):
    """
    Generates weighted edges for an input MultiDiGraph where the edge
    weights are stored under the edge attribute 'weight'.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    K : array
        'NxN' array where 'N' is the number of nodes in the diagram G.
        Adjacency matrix for G where each element kij is the edge weight
        (i.e. transition rate constant). For example, for a 2-state model
        with `k12=3` and `k21=4`, `K=[[0, 3], [4, 0]]`.

    """
    np.fill_diagonal(K, 0)  # Make sure diagonal elements are set to zero

    for i, row in enumerate(K):
        for j, elem in enumerate(row):
            if elem != 0:
                G.add_edge(i, j, weight=elem)


def retrieve_rate_matrix(G):
    """
    Retrieves rate matrix from edge data stored in input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram

    Returns
    -------
    rate_matrix : array
        'NxN' array where 'N' is the number of nodes in the diagram G.
        Adjacency matrix for G where each element kij is the edge weight
        (i.e. transition rate constant). For example, for a 2-state model
        with `k12=3` and `k21=4`, `K=[[0, 3], [4, 0]]`.
    """
    # sort the nodes in increasing order
    nodelist = sorted(G.nodes())
    rate_matrix = nx.to_numpy_array(G, nodelist=nodelist)
    return rate_matrix


def add_node_attribute(G, data, label):
    """
    Sequentially add attributes to nodes from array of values, i.e. state
    probabilities.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    data : array like
        Array or list of length 'n', where 'n' is the number of nodes in the
        diagram G. Elements must be in order, i.e. [x1, x2, x3, ..., xn]
    label : str
        Name of new attribute to be assigned to nodes.
    """
    for i in range(G.number_of_nodes()):
        G.nodes[i][label] = data[i]


def add_graph_attribute(G, data, label):
    """
    Add attribute to graph G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    data : anything
        Data to be assigned as an attribute to G.
    label : str
        Name of new attribute to be assigned to nodes.
    """
    G.graph[label] = data


def find_all_unique_cycles(G):
    """
    Finds all unique cycles for an input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph object
        Diagram of interest.

    Returns
    -------
    unique_cycles : list of lists of int
        List of cycles, where each cycle is a list of nodes in that cycle.
    """
    temp = []
    unique_cycles = []
    for cycle in nx.simple_cycles(G):
        temp.append(cycle)
        reverse_cycle = [cycle[0]] + cycle[len(G.nodes) : 0 : -1]
        if not reverse_cycle in temp:
            unique_cycles.append(cycle)
    return unique_cycles


def _is_ccw(cycle, start, end):
    """
    Function for determining if a cycle is CCW based on a pair of nodes in the
    cycle. For example, for a cycle [0, 1, 2], if moving from node 0 to 1 was
    in the CCW direction, one would input 'start=0' and 'end=1'. Function
    returns a value of 'True' if the input cycle is in the CCW direction.

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    start : int
        Node used as initial reference point.
    end : int
        Node used as final reference point.
    """
    double = 2 * cycle
    for i in range(len(double) - 1):
        if (double[i], double[i + 1]) == (start, end):
            return True
    return False


def get_ccw_cycle(cycle, order):
    """
    Function used for obtaining the CCW version of an input cycle, primarily
    used for kda.calculate_pi_difference() and kda.calculate_thermo_force().

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    order : list of int
        List of integers of length 2, where the integers must be nodes in the
        input cycle. This pair of nodes is used to determine which direction is
        CCW.
    """
    if not all(i in cycle for i in order):
        raise CycleError(f"Input node indices {order} do not exist in cycle {cycle}")
    if _is_ccw(cycle, order[0], order[1]):
        return cycle
    else:
        return cycle[::-1]
