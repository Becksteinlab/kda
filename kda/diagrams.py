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
.. autofunction:: generate_directional_diagrams
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


def _combine(x, y):
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
    unique_edges : array
        Array of edges (made from 2-tuples) that are unique to the diagram,
        [[0, 1], [1, 2], ...].
    """
    # get the indices for each edge pair that contains the target state
    adj_idx = np.nonzero(unique_edges == target)[0]
    # collect the edges that contain the target state
    adj_edges = unique_edges[adj_idx]
    # from the adjacent edges, get the neighbors of the target state
    neighbors = adj_edges[np.nonzero(adj_edges != target)]
    # if there are neighbors, continue
    if neighbors.size:
        # get the list of all possible indices
        all_idx = np.arange(unique_edges.shape[0])
        # get the indices for the edges that are
        # not connected to the target state
        nonadj_mask = np.ones(all_idx.size, dtype=bool)
        nonadj_mask[adj_idx] = False
        nonadj_idx = all_idx[nonadj_mask]
        # collect the edges that do not contain the target state
        nonadj_edges = unique_edges[nonadj_idx]
        # get the unique neighbors
        neighbors = np.unique(neighbors)
        # recursively generate a dictionary of all connections
        con_dict = functools.reduce(
            _combine,
            [{target: neighbors}]
            + [_get_directional_connections(i, nonadj_edges) for i in neighbors],
        )
        return con_dict
    else:
        # if there are no neighbors, return empty dictionary
        return {}


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
            edge_tuple = (neighb, target, 0)
            if not edge_tuple in values:
                values.append(edge_tuple)
    return values


def _construct_cycle_edges(cycle):
    """
    Constucts edge tuples in a cycle using the node indices in the cycle. It
    is important for the cycle to be in the correct order and not sorted, as
    a sorted cycle list will return incorrect edges.

    Parameters
    ----------
    cycle : list of int
        List of node indices for cycle of interest, index zero.

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


def _flux_edge_conditions(edge_list, n_flux_edges):
    """
    Conditions that need to be true for flux edges to be valid.

    Parameters
    ----------
    edge_list : list of tuples
        List of edges (tuples) to be checked. Common cases that need to be
        filtered out are where half the edges are the same but reversed, or
        there are simply too many edges to be a flux diagram.
    n_flux_edges : int
        Number of unique edges in given flux diagram. Defined as the difference
        between the number of nodes in the diagram of interest and the number of
        unique cycle edges in the cycle of interest.
    """
    # count the edges
    n_input_edges = len(edge_list)
    # check if the number of input edges are the
    # correct number of flux diagram edges
    if n_input_edges == n_flux_edges:
        # sort the list of edges into arrays of increasing order
        sorted_edges = np.sort(edge_list)
        tuples = [
            (sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(n_input_edges)
        ]
        # count how many of the sorted edges are unique
        n_unique_edges = len(set(tuples))
        if n_unique_edges == n_input_edges:
            # if there are no redundant edges, return True
            return True

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
    new_edge_list = [(e[1], e[0], e[2]) for e in edge_list]
    new_edge_list.extend(edge_list)
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


def enumerate_partial_diagrams(G):
    """
    Quantifies the number of partial diagrams/spanning trees that can be
    generated from an input graph. This implements Kirchhoff's theroem by
    generating the adjacency matrix from the input matrix, generating the
    Laplacian matrix from the adjacency matrix, then getting the cofactor
    matrix of the Laplacian matrix.

    Parameters
    ==========
    G : NetworkX MultiDiGraph
        Input diagram

    Returns
    =======
    n_partials : int
        The number of unique partial diagrams (spanning trees) that can be
        generated from a graph represented by the input rate matrix.
    """
    # get the adjacency matrix for K
    K_adj = nx.to_numpy_array(G)
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


def generate_partial_diagrams(G, return_edges=False):
    """
    Generates all partial diagrams for input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    return_edges : bool
        Binary used for determining whether to return NetworkX diagram objects
        (primarily for plotting) or the edge tuples (generally for
        calculations).

    Returns
    -------
    partials : array
        Array of NetworkX MultiDiGraphs where each graph is a unique
        partial diagram with no loops (return_edges=False), or a nested
        array of unique edges for valid partial diagrams (return_edges=True).
    """
    # calculate number of edges needed for each partial diagram
    n_edges = G.number_of_nodes() - 1
    # collect the nodes for the partial diagrams
    base_nodes = G.nodes()
    # get the unique edges in G
    unique_edges = _find_unique_edges(G)
    # calculate the number of expected partial diagrams
    n_partials = enumerate_partial_diagrams(G)
    # preallocate arrays for storing partial diagrams edges/graphs
    # and initialize a counter
    i = 0
    if return_edges:
        partials = np.empty((n_partials, n_edges, 2), dtype=np.int32)
    else:
        partials = np.empty(n_partials, dtype=object)
    # get list of possible combinations of unique edges (N choose N-1)
    # and iterate over each unique combination
    for edge_list in itertools.combinations(unique_edges, n_edges):
        # make a base partial graph
        partial = nx.Graph()
        partial.add_nodes_from(base_nodes)
        partial.add_edges_from(edge_list)
        # for the tree to be valid, it must have N-1 edges and
        # be connected. Since we already have the edge list
        # generated for N-1 edges, just check if it is connected
        if nx.is_connected(partial):
            if return_edges:
                partials[i, :] = edge_list
            else:
                partials[i] = partial
            i += 1

    return partials


def generate_directional_diagrams(G, return_edges=False):
    """
    Generates all directional diagrams for input diagram G.

    Parameters
    ----------
    partials : list
        List of NetworkX MultiDiGraphs where each graph is a unique partial
        diagram with no loops.
    return_edges : bool
        Binary used for determining whether to return NetworkX diagram objects
        (primarily for plotting) or the edge tuples (generally for
        calculations).

    Returns
    -------
    directional_partial_diagrams : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    directional_partial_diagram_edges : array
        Array of edges (made from 2-tuples) for valid directional partial
        diagrams.
    """
    partial_diagram_edges = generate_partial_diagrams(G, return_edges=True)

    base_nodes = G.nodes()
    n_states = G.number_of_nodes()
    n_partials = len(partial_diagram_edges)
    n_dir_diags = n_states * n_partials

    if return_edges:
        directional_diagrams = np.empty((n_dir_diags, n_states - 1, 3), dtype=np.int32)
    else:
        directional_diagrams = np.empty((n_dir_diags,), dtype=object)

    targets = np.sort(list(G.nodes))
    for i, target in enumerate(targets):
        for j, partial_edges in enumerate(partial_diagram_edges):
            # get dictionary of connections
            cons = _get_directional_connections(target, partial_edges)
            # get directional edges from connections
            dir_edges = _get_directional_edges(cons)
            if return_edges:
                directional_diagrams[j + i*n_partials] = dir_edges
            else:
                directional_diagram = nx.MultiDiGraph()
                directional_diagram.add_nodes_from(base_nodes)
                directional_diagram.add_edges_from(dir_edges)
                # set "is_target" to False for all nodes
                nx.set_node_attributes(directional_diagram, False, "is_target")
                # set target node to True
                directional_diagram.nodes[target]["is_target"] = True
                # add to array of directional partial diagrams
                directional_diagrams[j + i*n_partials] = directional_diagram

    return directional_diagrams


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
    if sorted(cycle) == sorted(G.nodes):
        print(
            f"Cycle {cycle} contains all nodes in G. No flux diagrams generated."
        )
        return None
    # get the edge tuples created by the input cycle
    cycle_edges = _construct_cycle_edges(cycle)
    cycle_edges = _append_reverse_edges(cycle_edges)
    # get all of the unique edges in the input diagram
    G_edges = _find_unique_edges(G)
    # get edges that are uncommon between cycle and G
    non_cycle_edges = _find_unique_uncommon_edges(G_edges, cycle)
    # number of non-cycle edges in flux diagram
    n_non_cycle_edges = G.number_of_nodes() - len(cycle)
    # generate all combinations of valid edges
    # TODO: generates too many edge lists: some create cycles, some
    # use both forward and reverse edges
    flux_diagrams = []
    for edge_list in itertools.combinations(non_cycle_edges, r=n_non_cycle_edges):
        # convert each edge list into numpy array
        edge_list = np.asarray(edge_list, dtype=int)
        # initialize empty list for storing uni-directional edges
        dir_edges = []
        for target in cycle:
            # for each node in the cycle, collect the directional
            # connection dictionary
            cons = _get_directional_connections(target, edge_list)
            if cons:
                # if there are directional connections, generate and
                # store the directional edges
                dir_edges.extend(_get_directional_edges(cons))
        if _flux_edge_conditions(dir_edges, n_non_cycle_edges):
            # initialize a graph object
            flux_diag = nx.MultiDiGraph()
            # add cycle edges to list of directional edges
            dir_edges.extend(cycle_edges)
            # add all edges to flux diagram
            flux_diag.add_edges_from(dir_edges)
            # collect all nodes in the potential flux
            # diagram from the diagram edges
            included_nodes = np.unique(flux_diag.edges)
            # if the diagram contains all nodes it is still valid
            if included_nodes.size == G.number_of_nodes():
                # count how many cycles are in the flux diagram by removing
                # 2-node cycles
                # NetworkX stores forward/reverse cycles, so if 2 remain there
                # is 1 unique cycle for this flux diagram
                contains_1_cycle = (
                    len([c for c in nx.simple_cycles(flux_diag) if len(c) > 2]) == 2
                )
                # if there is exactly 1 unique cycle in the
                # generated diagram, it is valid
                if contains_1_cycle:
                    # set "is_target" to False for all nodes
                    nx.set_node_attributes(flux_diag, False, "is_target")
                    for target in cycle:
                        # for nodes in cycle, mark True
                        flux_diag.nodes[target]["is_target"] = True
                    # append valid flux diagram to list
                    flux_diagrams.append(flux_diag)
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
    n_nodes = G.number_of_nodes()
    all_flux_diagrams = []
    for cycle in all_cycles:
        if len(cycle) == n_nodes:
            # for all-node cycles just append None since
            # they cannot have any flux diagrams
            all_flux_diagrams.append(None)
            continue
        flux_diagrams = generate_flux_diagrams(G, cycle)
        all_flux_diagrams.append(flux_diagrams)
    return all_flux_diagrams
