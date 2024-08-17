# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Diagram Generation
=========================================================================
The :mod:`~kda.diagrams` module contains code to generate partial diagrams
(undirected spanning trees), directional diagrams, and flux diagrams using the
diagram method developed by King and Altman :footcite:`king_schematic_1956`
and Hill :footcite:`hill_studies_1966`.

.. autofunction:: enumerate_partial_diagrams
.. autofunction:: generate_partial_diagrams
.. autofunction:: generate_directional_diagrams
.. autofunction:: generate_flux_diagrams
.. autofunction:: generate_all_flux_diagrams

References
==========
.. footbibliography::

"""

import functools
import itertools
import copy
import numpy as np
import networkx as nx

from kda import graph_utils


def _find_unique_edges(G):
    """
    Creates list of unique edges for input diagram ``G``. Effectively removes
    duplicate edges such as ``(1, 0)`` from ``[(0, 1), (1, 0)]``.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        Input diagram
    """
    # since non-directional graphs cannot contain forward/reverse edges,
    # simply create a simple graph and collect its (unique) set of edges
    G_temp = nx.Graph()
    G_temp.add_edges_from(G.edges())
    return list(G_temp.edges())


def _combine(x, y):
    """
    Used to reduce list of dictionaries into single dictionary where the keys
    are the indices of states, and the values are lists of neighbors for each
    state.
    """
    x.update(y)
    return x


def _get_neighbor_dict(target, unique_edges):
    """
    Recursively constructs dictionary containing neighbor connectivity
    information for a set of target states in a diagram.

    Parameters
    ----------
    target : int
        Index of target state
    unique_edges : ndarray
        Array of edges (made from 2-tuples) that are unique
        to the diagram, (e.g. ``[[0, 1], [1, 2], ...]``).

    Returns
    -------
    Dictionary of directional connections, where node
    indices are mapped to a list of their respective
    neighbor node indices (e.g. ``{0: [1, 5], 1: [2], ...}``).
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
            + [_get_neighbor_dict(i, nonadj_edges) for i in neighbors],
        )
        return con_dict
    else:
        # if there are no neighbors, return empty dictionary
        return {}


def _get_flux_path_edges(target, unique_edges):
    """
    Constructs edges for all paths leading to a target
    state using input collection of unique edge tuples.

    Parameters
    ----------
    target : int
        Target state.
    unique_edges : array
        Array of edges (made from 2-tuples) that are unique to the
        diagram (e.g. ``(1, 2)`` and ``(2, 1)`` are considered the same).

    Returns
    -------
    path_edges : list
        List of edge tuples (e.g. ``[(0, 1, 0), (1, 2, 0), ...]``).
    """
    neighbors = _get_neighbor_dict(target, unique_edges)
    path_edges = [(nbr, tgt, 0) for tgt in neighbors for nbr in neighbors[tgt]]
    return list(set(path_edges))


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
    cycle_edges : list of tuples
        List of edge tuples corresponding to the input cycle.
    """
    # slice cycle to generate edge tuples using consecutive nodes
    cycle_edges = list(zip(cycle[:-1], cycle[1:], np.zeros(len(cycle), dtype=int)))
    # append tuple connecting first/last nodes in input cycle
    cycle_edges.append((cycle[-1], cycle[0], 0))
    return cycle_edges


def _find_unique_uncommon_edges(G, cycle_edges):
    """
    Collects the set of unique non-cycle edges for a cycle in the input diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle_edges : list of tuples
        List of edge tuples for a cycle of interest. Both forward and
        reverse edges should be included (e.g. ``(1, 0)`` and ``(0, 1)``).

    Returns
    -------
    edges : list of tuples
        List of uncommon edges between ``G`` and ``cycle_edges``.
        Since these should be unique edges (no reverse edges), these are the
        unique uncommon edges between two diagrams (normal use case).
    """
    G_temp = nx.MultiDiGraph()
    G_temp.add_edges_from(G.edges())
    G_temp.remove_edges_from(cycle_edges)
    edges = _find_unique_edges(G_temp)
    return edges


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
        sorted_edges = np.sort(edge_list)[:, 1:]
        # count how many of the sorted edges are unique
        n_unique_edges = len(set(map(tuple, sorted_edges)))
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
    Helper function for :meth:`~kda.diagrams.enumerate_partial_diagrams()`.
    Uses singular value decomposition to get the cofactor matrix for
    the input Laplacian matrix.

    Parameters
    ----------
    K_laplace : ndarray
        ``NxN`` Laplacian matrix, where ``N`` is the number of nodes.

    Returns
    -------
    K_cof : ndarray
        Cofactor matrix for the input Laplacian matrix.
    """
    U, w, Vt = np.linalg.svd(K_laplace)
    N = len(w)
    g = np.tile(w, N)
    g[:: (N + 1)] = 1
    G = np.diag(-((-1) ** N) * np.prod(np.reshape(g, (N, N)), 1))
    K_cof = U @ G @ Vt
    K_cof = np.asarray(np.round(K_cof, decimals=0), dtype=int)
    return K_cof


def enumerate_partial_diagrams(G):
    """
    Quantifies the number of partial diagrams (undirected spanning
    trees) that can be generated from a kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram

    Returns
    -------
    n_partials : int
        The number of unique partial diagrams (spanning trees) that can be
        generated from a graph represented by the input rate matrix.

    Notes
    -----
    This implements Kirchhoff's matrix theroem
    :footcite:`chakraborty_algorithms_2019` by generating the adjacency
    matrix from the input diagram, generating the Laplacian matrix
    from the adjacency matrix, then getting the cofactor matrix of
    the Laplacian matrix. All cofactors are equal and equal to the
    number of undirected spanning trees.

    A more sophistocated version of this function is available in
    the ``NetworkX`` library :footcite:`hagberg_exploring_2008`
    (see `here <https://networkx.org/documentation/stable/reference/
    algorithms/generated/networkx.algorithms.tree.mst.
    number_of_spanning_trees.html>`_).

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
    Generates all partial diagrams (undirected spanning trees)
    for a kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    return_edges : bool
        Binary used for determining whether to return ``NetworkX``
        diagram objects (primarily for plotting) or the edge
        tuples (generally for calculations).

    Returns
    -------
    partials : ndarray of ``NetworkX.Graph``
        Array of ``NetworkX.Graph`` where each graph is a unique
        partial diagram with no loops (``return_edges=False``), or
        a nested array of unique edges for valid partial diagrams
        (``return_edges=True``).
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
    Generates all directional diagrams for a kinetic diagram
    using depth-first-search algorithm.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    return_edges : bool
        Binary used for determining whether to return ``NetworkX`` graph
        objects (primarily for plotting) or the edge tuples (generally
        for calculations).

    Returns
    -------
    directional_diagrams : ndarray or ndarray of ``NetworkX.DiGraph``
        Array of all directional diagram edges made from 3-tuples
        (``return_edges=True``) or array of all directional
        diagrams (``return_edges=False``) for ``G``.
    """
    partial_diagrams = generate_partial_diagrams(G, return_edges=False)

    n_states = G.number_of_nodes()
    n_partials = len(partial_diagrams)
    n_dir_diags = n_states * n_partials

    if return_edges:
        directional_diagrams = np.empty((n_dir_diags, n_states - 1, 3), dtype=np.int32)
    else:
        directional_diagrams = np.empty((n_dir_diags,), dtype=object)

    # get the set of target nodes in ascending order
    # so all directional diagrams for each state are
    # generated in order
    targets = np.sort(list(G.nodes))
    for i, target in enumerate(targets):
        for j, G_partial in enumerate(partial_diagrams):
            # apply depth-first-search to partial diagram to create
            # a directed spanning tree where the edges are directed
            # from the target node to the leaf nodes
            G_dfs = nx.dfs_tree(G_partial, source=target)
            if return_edges:
                # collect the edges from the directed spanning tree
                # and reverse the direction of the edges to get the correct
                # edges for a directional diagram
                dir_edges = np.fliplr(np.asarray(G_dfs.edges(), dtype=np.int32))
                # add in the zero column for now
                # TODO: change downstream functions so we
                # don't have to keep these unnecessary zeros
                dir_edges = np.column_stack((dir_edges, np.zeros(dir_edges.shape[0])))
                directional_diagrams[j + i*n_partials] = dir_edges
            else:
                # make a copy of the `nx.DiGraph` with reversed
                # edges to get the directional diagram
                G_directional = G_dfs.reverse(copy=True)
                # set "is_target" to False for all nodes
                nx.set_node_attributes(G_directional, False, "is_target")
                # set target node to True
                G_directional.nodes[target]["is_target"] = True
                # add to array of directional diagrams
                directional_diagrams[j + i*n_partials] = G_directional

    return directional_diagrams


def generate_flux_diagrams(G, cycle):
    """
    Generates all flux diagrams for a specific cycle in the kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    flux_diagrams : list of ``NetworkX.MultiDiGraph``
        List of flux diagrams. Diagrams contain the input cycle
        where remaining edges follow path pointing to ``cycle``.
        Cycle nodes are labeled by attribute ``'is_target'``.
    """
    if sorted(cycle) == sorted(G.nodes):
        print(
            f"Cycle {cycle} contains all nodes in G. No flux diagrams generated."
        )
        return None
    # get the edge tuples created by the input cycle
    cycle_edges = _construct_cycle_edges(cycle)
    cycle_edges = _append_reverse_edges(cycle_edges)
    # get edges that are uncommon between cycle and G
    non_cycle_edges = _find_unique_uncommon_edges(G, cycle_edges)
    # number of non-cycle edges in flux diagram
    n_non_cycle_edges = G.number_of_nodes() - len(cycle)
    # generate all combinations of valid edges
    # TODO: generates too many edge lists: some create cycles, some
    # use both forward and reverse edges
    flux_diagrams = []
    for edge_list in itertools.combinations(non_cycle_edges, r=n_non_cycle_edges):
        # convert each edge list into numpy array
        edge_list = np.asarray(edge_list, dtype=int)
        # collect the directional edges
        dir_edges = []
        for target in cycle:
            path_edges = _get_flux_path_edges(target, edge_list)
            dir_edges.extend(path_edges)
        if _flux_edge_conditions(dir_edges, n_non_cycle_edges):
            # initialize a graph object
            flux_diag = nx.MultiDiGraph()
            # add cycle edges to list of directional edges
            dir_edges.extend(cycle_edges)
            # add all edges to flux diagram
            flux_diag.add_edges_from(dir_edges)
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
    Generates all flux diagrams for a kinetic diagram.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram

    Returns
    -------
    all_flux_diagrams : list of lists of ``NetworkX.MultiDiGraph``
        List of lists of flux diagrams, where each list
        is for a different cycle in ``G``.
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
