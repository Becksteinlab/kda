# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Diagram Generation
=========================================================================
This file contains a host of functions aimed at the analysis of biochemical
kinetic diagrams, using the methods of Hill.

.. autofunction:: enumerate_partial_diagrams
.. autofunction:: generate_partial_diagrams
.. autofunction:: generate_directional_partial_diagrams
.. autofunction:: generate_flux_diagrams
.. autofunction:: generate_all_flux_diagrams
"""

import functools
import itertools
import copy
import numpy as np
import networkx as nx

from kda import graph_utils


def _find_unique_edges(G):
    """
    Creates list of unique edges for input diagram G. Effectively removes
    duplicate edges such as '(1, 0)' from [(0, 1), (1, 0)].

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    """
    edges = list(G.edges)  # Get list of edges
    sorted_edges = np.sort(edges)  # Sort list of edges
    tuples = [
        (sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))
    ]  # Make list of edge tuples
    return list(set(tuples))


def combine(x, y):
    """
    Used to reduce list of dictionaries into single dictionary where the keys
    are the indices of states, and the values are lists of neighbors for each
    state.
    """
    x.update(y)
    return x


def _get_directional_connections(target, unique_edges):
    """
    Recursively constructs dictionary of directional connections for a given
    target state, {0: [1, 5], 1: [2], ...}. This allows for iterating through a
    given partial diagram in _get_directional_edges() to construct the list of
    directional edges for each directional partial diagram.

    Parameters
    ----------
    target : int
        Index of target state
    unique_edges : list
        List of edges (2-tuples) that are unique to the diagram,
        [(0, 1), (1, 2), ...].
    """
    edges = [
        i for i in unique_edges if target in i
    ]  # Find edges that connect to target state
    neighbors = [
        [j for j in i if not j == target][0] for i in edges
    ]  # Find states neighboring target state
    if not neighbors:
        return {}
    unique_edges = [
        k for k in unique_edges if not k in edges
    ]  # Make new list of unique edges that does not contain original unique edges
    return functools.reduce(
        combine,
        [{target: neighbors}]
        + [_get_directional_connections(i, unique_edges) for i in neighbors],
    )


def _get_directional_edges(cons):
    """
    Iterates through a dictionary of connections to construct the list
    of directional edges for each directional partial diagram.

    Parameters
    ----------
    cons : dict
        Dictionary of directional connections for a given target state,
        {0: [1, 5], 1: [2], ...}.

    Returns
    -------
    values : list
        List of edges (3-tuples) corresponding to a single directional partial
        diagram, [(1, 0, 0), (5, 0, 0), ...].
    """
    values = []
    for target in cons.keys():
        for neighb in cons[target]:
            values.append((neighb, target, 0))
    return values


def _construct_cycle_edges(cycle):
    """
    Constucts edge tuples in a cycle using the node indices in the cycle. It
    is important for the cycle to be in the correct order and not sorted, as
    a sorted cycle list will return incorrect edges.

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    reverse_list : list of tuples
        List of edge tuples corresponding to the input cycle.
    """
    reverse_list = list(zip(cycle[:-1], cycle[1:], np.zeros(len(cycle), dtype=int)))
    reverse_list.append((cycle[-1], cycle[0], 0))
    return reverse_list


def _find_node_edges(cycle, r=2):
    """
    Function for generating all possible edges that contain 2 nodes. First finds
    all combinations of pairs of nodes, then appends the reverse edges.

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    r : int (optional)
        Length of tuples to be generated, default is 2.

    Returns
    -------
    node_edges : list of tuples
        List of all possible pairs of nodes, where node pairs are in tuple form.
    """
    node_edges = list(itertools.combinations(cycle, r))  # generate node edges
    for edge in node_edges.copy():
        node_edges.append((edge[1], edge[0]))  # append reverse edge
    return node_edges


def _find_unique_uncommon_edges(G_edges, cycle):
    """
    Function for removing cycle edges for flux diagram generation.

    Parameters
    ----------
    G_edges : list of tuples
        List of all edges in a diagram G. For general use, this list should have
        only unique edges (not both forward and reverse).
    cycle : list of int
        Cycle to generate edge list from. Order of node indices does not matter.

    Returns
    -------
    valid_non_cycle_edges : list of tuples
        List of uncommon edges between input list "G_edges" and "cycle_edges".
        Since these should be unique edges (no reverse edges), these are the
        unique uncommon edges between two diagrams (normal use case).
    """
    cycle_edges = _construct_cycle_edges(cycle)
    sorted_G_edges = [sorted(edge) for edge in G_edges]
    sorted_cycle_edges = [sorted(edge) for edge in cycle_edges]
    non_cycle_edges = [
        tuple(edge) for edge in sorted_G_edges if not edge in sorted_cycle_edges
    ]
    node_edges = _find_node_edges(cycle)  # generate edges that only contain 2 nodes
    valid_non_cycle_edges = [
        edge for edge in non_cycle_edges if not edge in node_edges
    ]  # remove node edges
    return valid_non_cycle_edges


def _flux_edge_conditions(edge_list, N):
    """
    Conditions that need to be true for flux edges to be valid.

    Parameters
    ----------
    edge_list : list of tuples
        List of edges (tuples) to be checked. Common cases that need to be
        filtered out are where half the edges are the same but reversed, or
        there are simply too many edges to be a flux diagram.
    N : int
        Number of unique edges in given flux diagram. Defined as the difference
        between the number of nodes in the diagram of interest and the number of
        unique cycle edges in the cycle of interest.
    """
    sorted_edges = np.sort(edge_list)
    tuples = [
        (sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))
    ]
    unique_edges = list(set(tuples))
    # the number of edges must equal the number of edges in the list
    if len(edge_list) == N:
        if len(unique_edges) == len(edge_list):
            return True
        else:
            return False
    else:
        return False


def _append_reverse_edges(edge_list):
    """
    Returns a list that contains original edges and reverse edges.
    Parameters
    ----------
    edge_list : list of edge tuples
        List of unidirectional edges to have reverse edges appended to.
    Returns
    -------
    new_edge_list : list of edge tuples
        List of edge tuples with both forward and reverse edges.
    """
    new_edge_list = []
    for edge in edge_list:
        if not edge in new_edge_list:
            new_edge_list.append(edge)
            new_edge_list.append((edge[1], edge[0], edge[2]))
    return new_edge_list


def _get_cofactor_matrix(K_laplace):
    """
    Helper function for `enumerate_partial_diagrams()`. Uses singular value
    decomposition to get the cofactor matrix for the input Laplacian matrix.

    Parameters
    ==========
    K_laplace : array
        `NxN` Laplacian matrix, where 'N' is the number of nodes.

    Returns
    =======
    K_cof : array
        Cofactor matrix for the input Laplacian matrix.
    """
    U, w, Vt = np.linalg.svd(K_laplace)
    N = len(w)
    g = np.tile(w, N)
    g[:: (N + 1)] = 1
    G = np.diag(-((-1) ** N) * np.product(np.reshape(g, (N, N)), 1))
    K_cof = U @ G @ Vt
    K_cof = np.asarray(np.round(K_cof, decimals=0), dtype=int)
    return K_cof


def enumerate_partial_diagrams(K0):
    """
    Quantifies the number of partial diagrams/spanning trees that can be
    generated from an input graph, represented by a rate matrix. This implements
    Kirchhoff's theroem by generating the adjacency matrix from the input
    matrix, generating the Laplacian matrix from the adjacency matrix, then
    getting the cofactor matrix of the Laplacian matrix.

    Parameters
    ==========
    K0 : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input K matrix.

    Returns
    =======
    n_partials : int
        The number of unique partial diagrams (spanning trees) that can be
        generated from a graph represented by the input rate matrix.
    """
    # make a copy of the input rate matrix to operate on
    K = copy.deepcopy(K0)
    # get the adjacency matrix for K
    K_adj = np.asarray(K != 0, dtype=int)
    # use the adjacency matrix to generate the Laplacian matrix
    # multiply through by -1
    K_laplace = -1 * K_adj
    # now assign the degree of the ith node to the ith diagonal value
    # NOTE: the degree of each node should be the sum of each row or column
    # in the adjacency matrix
    for i in range(len(K_adj)):
        K_laplace[i, i] = np.sum(K_adj[i])
    # get the cofactor matrix
    K_cof = _get_cofactor_matrix(K_laplace=K_laplace)
    # check that all values in the cofactor matrix are the same by
    # checking if minimum and maximum values are equal
    assert K_cof.min() == K_cof.max()
    # just take the first value from the cofactor matrix since they are all
    # the same value
    n_partials = np.abs(K_cof[0][0])
    return n_partials


def generate_partial_diagrams(G):
    """
    Generates all partial diagrams for input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram

    Returns
    -------
    valid_partials : list
        List of NetworkX MultiDiGraphs where each graph is a unique partial
        diagram with no loops.
    """
    # Calculate number of edges needed for each partial diagram
    N_partial_edges = G.number_of_nodes() - 1
    # Get list of all possible combinations of unique edges (N choose n)
    combinations = list(itertools.combinations(_find_unique_edges(G), N_partial_edges))
    # Using combinations, generate all possible partial diagrams (including closed loops)
    partials_all = []
    for i in combinations:
        diag = G.copy()
        diag.remove_edges_from(list(G.edges()))
        edges = []
        for j in i:
            edges.append((j[0], j[1]))  # Add edge from combinations
            edges.append((j[1], j[0]))  # Add reverse edge
        diag.add_edges_from(edges)
        partials_all.append(diag)
    # Remove unwanted (closed loop) diagrams
    valid_partials = []
    for i in partials_all:
        if len(list(nx.simple_cycles(i))) == N_partial_edges:
            valid_partials.append(i)
    return valid_partials


def generate_directional_partial_diagrams(G):
    """
    Generates all directional partial diagrams for input diagram G.

    Parameters
    ----------
    partials : list
        List of NetworkX MultiDiGraphs where each graph is a unique partial
        diagram with no loops.

    Returns
    -------
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    """
    partials = generate_partial_diagrams(G)
    targets = np.sort(list(G.nodes))
    dir_partials = []
    for target in targets:
        for i in range(len(partials)):
            partial = partials[i].copy()  # Make a copy of directional partial diagram
            unique_edges = _find_unique_edges(
                partial
            )  # Find unique edges of that diagram
            cons = _get_directional_connections(
                target, unique_edges
            )  # Get dictionary of connections
            dir_edges = _get_directional_edges(
                cons
            )  # Get directional edges from connections
            partial.remove_edges_from(
                list(partial.edges())
            )  # Remove all edges from partial diagram
            for e in dir_edges:
                partial.add_edge(
                    e[0], e[1], e[2]
                )  # Add relevant edges to partial diagram
            for t in targets:  # Add node attrbutes to label target nodes
                if t == target:
                    partial.nodes[t]["is_target"] = True
                else:
                    partial.nodes[t]["is_target"] = False
            dir_partials.append(partial)  # Append to list of partial diagrams
    return dir_partials


def generate_flux_diagrams(G, cycle):
    """
    Creates all of the directional flux diagrams for the given cycle in the
    diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    flux_diagrams : list of NetworkX MultiDiGraph objects
        List of directional flux diagrams. Diagrams contain the input cycle
        where remaining edges follow path pointing to cycle. Cycle nodes are
        labeled by attribute 'is_target'.
    """
    if sorted(cycle) == sorted(list(G.nodes)):
        print(
            f"Cycle {cycle} contains all nodes in G, no flux diagrams can be"
            f" generated. Value of None Returned."
        )
    else:
        cycle_edges = _construct_cycle_edges(cycle)
        G_edges = _find_unique_edges(G)
        # get edges that are uncommon between cycle and G
        non_cycle_edges = _find_unique_uncommon_edges(G_edges, cycle)
        # number of non-cycle edges in flux diagram
        N = G.number_of_nodes() - len(cycle_edges)
        # all combinations of valid edges
        # generates too many edge lists: some create cycles, some use both forward and reverse edges
        flux_edge_lists = list(itertools.combinations(non_cycle_edges, r=N))
        flux_diagrams = []
        for edge_list in flux_edge_lists:
            dir_edges = []
            for target in cycle:
                cons = _get_directional_connections(target, edge_list)
                if not len(cons) == 0:
                    dir_edges.append(_get_directional_edges(cons))
            dir_edges_flat = [edge for edges in dir_edges for edge in edges]
            if _flux_edge_conditions(dir_edges_flat, N) == True:
                diag = G.copy()
                diag.remove_edges_from(G.edges)
                for edge in dir_edges_flat:
                    diag.add_edge(edge[0], edge[1], edge[2])
                for edge in cycle_edges:
                    diag.add_edge(edge[0], edge[1], 0)
                    diag.add_edge(edge[1], edge[0], 0)
                included_nodes = np.unique(diag.edges)
                if sorted(G.nodes) == sorted(included_nodes):
                    for target in diag.nodes():
                        if target in cycle:
                            diag.nodes[target]["is_target"] = True
                        else:
                            diag.nodes[target]["is_target"] = False
                    flux_diagrams.append(diag)
                else:
                    continue
            else:
                continue
        return flux_diagrams


def generate_all_flux_diagrams(G):
    """
    Creates all of the directional flux diagrams for the diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram

    Returns
    -------
    all_flux_diagrams : list of lists of NetworkX MultiDiGraph objects
        List of lists of flux diagrams, where each list is for a different cycle
        in G.
    """
    all_cycles = graph_utils.find_all_unique_cycles(G)
    all_flux_diagrams = []
    for cycle in all_cycles:
        flux_diagrams = generate_flux_diagrams(G, cycle)
        if flux_diagrams == None:
            continue
        else:
            for diag in flux_diagrams:
                if (
                    len(graph_utils.find_all_unique_cycles(diag)) == 1
                ):  # check if there is only 1 unique cycle
                    continue
                else:
                    raise CycleError(
                        "Flux diagram has more than 1 closed loop for cycle {}.".format(
                            cycle
                        )
                    )
            all_flux_diagrams.append(flux_diagrams)
    return all_flux_diagrams
