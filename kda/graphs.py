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
.. autofunction:: add_node_attribute
.. autofunction:: add_graph_attribute
.. autofunction:: find_all_unique_cycles
.. autofunction:: get_ccw_cycle
"""


import numpy as np
import networkx as nx


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
    elif not isinstance(names[0, 0], str):
        raise TypeError("Labels entered for 'names' must be strings.")
    np.fill_diagonal(vals, 0)  # Make sure diagonal elements are set to zero

    for i, row in enumerate(vals):
        for j, elem in enumerate(row):
            if not elem == 0:
                attrs = {name_key: names[i, j], val_key: elem}
                G.add_edge(i, j, **attrs)


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
    if len(unique_cycles) == 1:
        print("Only 1 cycle found: {}".format(unique_cycles[0]))
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
    return None


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
    CCW = _is_ccw(cycle, order[0], order[1])
    if CCW == True:
        return cycle
    elif not CCW:
        return cycle[::-1]
    else:
        raise CycleError("Direction of cycle {} could not be determined.".format(cycle))
