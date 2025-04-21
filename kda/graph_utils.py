# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Graph Utilities
=========================================================================
This file contains a host of utility functions for ``NetworkX`` graphs.

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
        The number of states in a diagram used to create a ``NxN``
        matrix of strings.

    Returns
    =======
    K_string : ndarray
        An ``NxN`` array of strings where ``N`` is the number of states
        in a diagram and the diagonal values of the array are zeros.
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
    Generate edges for an input kinetic diagram ``G``, where edges have
    attributes ``"name"`` (for rate constant variable names, e.g.
    ``"k12"``) and ``"val"`` (for the rate constant values, e.g. ``100``).

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    vals : ndarray
        ``NxN`` array where ``N`` is the number of nodes in ``G``. Contains
        the kinetic rate *values* for each transition in ``G``. For example,
        assuming we have some values ``k12_val`` and ``k21_val``, for a
        2-state diagram ``vals = [[0, k12_val], [k21_val, 0]]``.
    names : ndarray, optional
        ``NxN`` array where ``N`` is the number of nodes in ``G``. Contains
        the kinetic rate *variable names* (as strings) for each transition
        in ``G``. For example, for a 2-state diagram
        ``names = [[0, "k12"], ["k21", 0]]``.
    val_key : str, optional
        Attribute key used to retrieve kinetic rate *values* from the
        edge data stored in ``G.edges``. The default is ``"val"``.
    name_key : str, optional
        Attribute key used to retrieve kinetic rate *variable names* from
        the edge data stored in ``G.edges``. The default is ``"name"``.
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
    Retrieves rate matrix from a kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    key : str, optional
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"val"``.

    Returns
    -------
    rate_matrix : ndarray
        ``NxN`` array where ``N`` is the number of nodes/states in the
        diagram ``G``. Contains the values/rates for each edge.
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
    Finds all unique cycles in a kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram

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
    cycle. For example, for ``cycle = [0, 1, 2]``, if moving from node ``0``
    to ``1`` was in the CCW direction, one would input ``start=0`` and
    ``end=1``. Function returns a value of ``True`` if the input cycle is
    in the CCW direction.

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
    used for :func:`~kda.calculations.calculate_pi_difference()` and
    :func:`~kda.calculations.calculate_thermo_force()`.

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    order : list of int
        List of integers of length 2 (e.g. ``[0, 1]``), where the integers are
        nodes in ``cycle``. The pair of nodes should be ordered such that
        a counter-clockwise path is followed.
    """
    if not all(i in cycle for i in order):
        raise CycleError(f"Input node indices {order} do not exist in cycle {cycle}")
    if _is_ccw(cycle, order[0], order[1]):
        return cycle
    else:
        return cycle[::-1]
