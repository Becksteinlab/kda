# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Graph Utilities
=========================================================================
This file contains a host of utility functions for NetworkX graphs.

Functions
=========
.. autofunction:: generate_edges
.. autofunction:: find_all_unique_cycles
.. autofunction:: generate_K_string_matrix
.. autofunction:: retrieve_rate_matrix
.. autofunction:: get_ccw_cycle

"""


import numpy as np
import networkx as nx

from kda.exceptions import CycleError


def generate_K_string_matrix(N_states):
    """
    Creates the string variant of the K-matrix based on the number of states
    in a diagram.

    Parameters
    ==========
    N_states : int
        The number of states in a diagram used to create a `NxN` matrix of
        strings.

    Returns
    =======
    K_string : NumPy array
        An `NxN` array of strings where `N` is the number of states in a
        diagram and the diaganol values of the array are zeros.
    """
    K_string = []
    for i in range(N_states):
        for j in range(N_states):
            K_string.append(f"k{i+1}{j+1}")
    K_string = np.array(K_string).reshape((N_states, N_states))
    np.fill_diagonal(K_string, val=0)
    return K_string


def generate_edges(G, vals, names=None, val_key="val", name_key="name"):
    """
    Generate edges with attributes 'val' and 'name'.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    vals : array
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the values associated with the attribute names in 'names'. For example,
        assuming k12 and k21 had already been assigned values, for a 2 state
        diagram 'vals' = [[0, k12], [k21, 0]].
    names : array (optional)
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the names of all of the attributes corresponding to the values in
        'vals' as strings, i.e. [[0, "k12"], ["k21", 0]].
    val_key : str (optional)
        Key used to retrieve variable values in 'vals'. Default is 'val'.
    name_key : str (optional)
        Key used to retrieve variable names in 'names'. Default is 'name'.
    """
    if names is None:
        names = generate_K_string_matrix(vals.shape[0])
    if isinstance(vals[0, 0], str):
        raise TypeError(
            "Values entered for 'vals' must be integers or floats, not strings."
        )
    if not isinstance(names[0, 0], str):
        raise TypeError("Labels entered for 'names' must be strings.")
    np.fill_diagonal(vals, 0)  # Make sure diagonal elements are set to zero

    for i, row in enumerate(vals):
        for j, elem in enumerate(row):
            if not elem == 0:
                attrs = {name_key: names[i, j], val_key: elem}
                G.add_edge(i, j, **attrs)


def retrieve_rate_matrix(G, key="val"):
    """
    Retrieves rate matrix from edge data stored in input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    val : str (optional)
        Key used to retrieve values from edges. Default is 'val'.

    Returns
    -------
    rate_matrix : array
        'NxN' array where 'N' is the number of nodes/states in the diagram G.
        Contains the values/rates for each edge.
    """
    # get the number of states
    n_states = G.number_of_nodes()
    # create `n_states x n_states` zero-array
    rate_matrix = np.zeros(shape=(n_states, n_states), dtype=float)
    # iterate over edges
    for edge in G.edges(data=True):
        i, j, data = edge
        # use edge indices and edge values to construct rate matrix
        rate_matrix[i, j] = data[key]
    return rate_matrix


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
