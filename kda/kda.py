# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analyzer (kda)
=========================================================================
This file contains a host of functions aimed at the analysis of biochemical
kinetic diagrams, using the methods of Hill.

"""

import numpy as np
import networkx as nx
import scipy.integrate
import sympy
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import functools
import itertools


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
    combinations = list(itertools.combinations(find_unique_edges(G), N_partial_edges))
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
            partial = partials[i].copy()               # Make a copy of directional partial diagram
            unique_edges = find_unique_edges(partial)      # Find unique edges of that diagram
            cons = get_directional_connections(target, unique_edges)   # Get dictionary of connections
            dir_edges = get_directional_edges(cons)                    # Get directional edges from connections
            partial.remove_edges_from(list(partial.edges()))                 # Remove all edges from partial diagram
            for e in dir_edges:
                partial.add_edge(e[0], e[1], e[2])                        # Add relevant edges to partial diagram
            for t in targets:                    # Add node attrbutes to label target nodes
                if t == target:
                    partial.nodes[t]['is_target'] = True
                else:
                    partial.nodes[t]['is_target'] = False
            dir_partials.append(partial)                                  # Append to list of partial diagrams
    return dir_partials

def calc_state_probabilities(G, dir_partials, key, output_strings=False):
    """
    Calculates state probabilities and generates analytic function strings for
    input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic state multplicity and normalization functions.

    Returns
    -------
    state_probabilities : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    state_mults : list
        List of analytic state multiplicity functions in string form.
    norm : str
        Analytic state multiplicity function normalization function in
        string form. This is the sum of all multiplicty functions.
    """
    N = G.number_of_nodes() # Number of nodes/states
    partial_mults = []
    edges = list(G.edges)
    if output_strings == False:
        if isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise Exception("To enter variable strings set parameter output_strings=True.")
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = 1          # assign initial value of 1
            for e in edge_list:                                 # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]][key]    # multiply the rate of each edge in edge_list
            partial_mults.append(products)
        N_terms = np.int(len(dir_partials)/N) # calculate the number of terms to be summed for given state, s
        state_mults = []
        partial_mults = np.array(partial_mults)
        for s in range(N):    # iterate over number of states, "s"
            state_mults.append(partial_mults[N_terms*s:N_terms*s+N_terms].sum(axis=0))
        state_mults = np.array(state_mults)
        state_probs = state_mults/state_mults.sum(axis=0)
        if any(elem < 0 for elem in state_probs) == True:
            raise Exception("Calculated negative state probabilities, overflow or underflow occurred.")
        return state_probs
    elif output_strings == True:
        if not isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise Exception("To enter variable values set parameter output_strings=False.")
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = []
            for e in edge_list:
                products.append(G.edges[e[0], e[1], e[2]][key]) # append rate constant names from dir_par to list
            partial_mults.append(products)
        N_terms = np.int(len(dir_partials)/N) # calculate the number of terms to be summed for given state, s
        state_mults = []
        term_list = []  # create empty list to put products of rate constants (terms) in
        for k in partial_mults:
            term_list.append("*".join(k))    # join rate constants for each dir_par by delimeter "*"
        for s in range(N):    # iterate over number of states, "s"
            state_mults.append("+".join(term_list[N_terms*s:N_terms*s+N_terms]))    # join appropriate terms for each state by delimeter "+"
        norm = "+".join(state_mults)    # sum all terms to get normalization factor
        return state_mults, norm

def construct_sympy_prob_funcs(state_mult_funcs, norm_func):
    """
    Constructs analytic state probability SymPy functions

    Parameters
    ----------
    state_mult_funcs : list of str
        List of length 'N', where N is the number of states, that contains the
        analytic multiplicity function for each state
    norm_func : str
        Sum of all state multiplicity functions, the normalization factor to
        calculate the state probabilities

    Returns
    -------
    sympy_funcs : list
        List of analytic state probability SymPy functions.
    """
    sympy_funcs = []   # create empty list to fill with state probability functions
    for func in state_mult_funcs:
        prob_func = parse_expr(func)/parse_expr(norm_func) # convert strings into SymPy functions, normalize
        sympy_funcs.append(prob_func)
    return sympy_funcs

def construct_lambdify_funcs(sympy_funcs, rate_names):
    """
    Constructs lambdified functions.

    Parameters
    ----------
    sympy_funcs : list of SymPy functions
        List of SymPy functions.
    rate_names : list
        List of strings, where each element is the name of the variables for
        the input probability functions, ["x12", "x21", "x23", ...].

    Returns
    -------
    state_prob_funcs : list
        List of lambdified analytic state probability functions.
    """
    if isinstance(sympy_funcs, Mul) == True:
        return lambdify(rate_names, sympy_funcs, "numpy")
    elif isinstance(sympy_funcs, list) == True:
        state_prob_funcs = []   # create empty list to fill with state probability functions
        for func in sympy_funcs:
            state_prob_funcs.append(lambdify(rate_names, func, "numpy"))    # convert into "lambdified" functions that work with NumPy arrays
        return state_prob_funcs

def generate_edges(G, vals, names=[None], val_key='val', name_key='name'):
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
    if isinstance(vals[0, 0], str):
        raise Exception("Values entered for 'vals' must be integers or floats, not strings.")
    elif not isinstance(names[0, 0], str):
        raise Exception("Labels entered for 'names' must be strings.")
    np.fill_diagonal(vals, 0) # Make sure diagonal elements are set to zero
    if len(names) == 1:
        for i, row in enumerate(vals):
            for j, elem in enumerate(row):
                if not elem == 0:
                    attrs = {val_key : elem}
                    G.add_edge(i, j, **attrs)
    else:
        for i, row in enumerate(vals):
            for j, elem in enumerate(row):
                if not elem == 0:
                    attrs = {name_key : names[i, j], val_key : elem}
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

def solve_ODE(P, K, t_max, tol=1e-16, **options):
    """
    Integrates state probability ODE's to find steady state probabilities.

    Parameters
    ----------
    P : array
        'Nx1' matrix of initial state probabilities.
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input k
        matrix.
    t_max : int
        Length of time for integrator to run, in seconds.
    tol : float (optional)
        Tolerance value used as convergence criteria. Once all dp/dt values for
        each state are less than the tolerance the integrator will terminate.
        Default is 1e-16.
    options
        Options passed to scipy.integrate.solve_ivp().

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at t.

    Note:
    For all parameters and returns, view the SciPy.integrate.solve_ivp()
    documentation: "https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.integrate.solve_ivp.html"
    """
    def convert_K(K):
        """
        Sets the diagonal elements of the input k matrix to zero, then takes the
        transpose, and lastly sets the diagonals equal to the negative of the
        sum of the column elements.
        """
        N = len(K)
        np.fill_diagonal(K, 0)
        K = K.T
        for i in range(N):
            K[i, i] = -K[:, i].sum(axis=0)
        return K

    def KdotP(t, y):
        """
        y = [p1, p2, p3, ... , pn]
        """
        return np.matmul(k, y, dtype=np.float64)

    def terminate(t, y):
        y_prime = np.matmul(k, y, dtype=np.float64)
        if all(elem < tol for elem in y_prime) == True:
            return False
        else:
            return True

    terminate.terminal = True
    k = convert_K(K)
    y0 = np.array(P, dtype=np.float64)
    return scipy.integrate.solve_ivp(fun=KdotP, t_span=(0, t_max), y0=y0,
                                     events=[terminate], **options)

def find_unique_edges(G):
    """
    Creates list of unique edges for input diagram G. Effectively removes
    duplicate edges such as '(1, 0)' from [(0, 1), (1, 0)].

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    """
    edges = list(G.edges)           # Get list of edges
    sorted_edges = np.sort(edges)      # Sort list of edges
    tuples = [(sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))]   # Make list of edges tuples
    return list(set(tuples))

def combine(x, y):
    """
    Used to reduce list of dictionaries into single dictionary where the keys
    are the indices of states, and the values are lists of neighbors for each
    state.
    """
    x.update(y)
    return x

def get_directional_connections(target, unique_edges):
    """
    Recursively constructs dictionary of directional connections for a given
    target state, {0: [1, 5], 1: [2], ...}. This allows for iterating through a
    given partial diagram in get_directional_edges() to construct the list of
    directional edges for each directional partial diagram.

    Parameters
    ----------
    target : int
        Index of target state
    unique_edges : list
        List of edges (2-tuples) that are unique to the diagram,
        [(0, 1), (1, 2), ...].
    """
    edges = [i for i in unique_edges if target in i]    # Find edges that connect to target state
    neighbors = [[j for j in i if not j == target][0] for i in edges] # Find states neighboring target state
    if not neighbors:
        return {}
    unique_edges = [k for k in unique_edges if not k in edges]  # Make new list of unique edges that does not contain original unique edges
    return functools.reduce(combine, [{target: neighbors}] + [get_directional_connections(i, unique_edges) for i in neighbors])

def get_directional_edges(cons):
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

def get_unique_uncommon_edges(G, diagrams):
    """
    Finds unique uncommon edges between G and each diagram in diagrams.

    Parameters
    ----------
    G : NetworkX MultiDiGraph object
        Input diagram
    diagrams : list of NetworkX MultiDiGraph objects
        List of diagrams of interest.

    Returns
    -------
    unique_uncommon_edges : list of lists of tuples
        List of lists where each list contains the edges that are unique
        between G and the given diagram.
    """
    all_uncommon_edges = []
    for diag in diagrams:
        uncommon_edges = []
        for edge in list(G.edges()):
            if not edge in list(diag.edges()):
                uncommon_edges.append(edge)
        all_uncommon_edges.append(uncommon_edges)
    unique_uncommon_edges = []
    for edges in all_uncommon_edges:
        unique_uncommon_edges.append(list(set([(e[0], e[1]) for e in np.sort(edges)])))
    return unique_uncommon_edges

def get_indices_of_flux_diagrams(cycles, flux_diagrams):
    """
    Finds the indices of the input list of flux diagrams that correspond to the
    input cycle(s). If a single cycle is input it will only return a list of
    the corresponding flux diagram list indices. If a list of cycles is input
    it will return a list of lists of indices, where each list contains the
    indices associated with that cycle.

    Parameters
    ----------
    cycles : list of int or list of lists of int
        List of cycles where each cycle is a list of nodes corresponding to a
        cycle in a diagram.
    flux_diagrams : list of NetworkX MultiDiGraph objects
        List of flux diagrams corresponding to the input cycles.

    Returns
    -------
    cycle_idx : list of int
        List of input flux diagram list indices.
    all_cycle_idx : list of lists of int
        List of lists of indices of input flux diagram list, where each index
        corresponds to a flux diagram that contains the cycle in that list.
    """
    if isinstance(cycles[0], int):
        cycle_idx = []
        for i, diag in enumerate(flux_diagrams):
            main_cycles = [set(c) for c in list(nx.simple_cycles(diag)) if len(c) > 2]
            if set(cycles) in main_cycles:
                cycle_idx.append(i)
        return cycle_idx
    else:
        all_cycle_idx = []
        for cycle in cycles:
            cycle_idx = []
            for i, diag in enumerate(flux_diagrams):
                main_cycles = [set(c) for c in list(nx.simple_cycles(diag)) if len(c) > 2]
                if set(cycle) in main_cycles:
                    cycle_idx.append(i)
            all_cycle_idx.append(cycle_idx)
        return all_cycle_idx

def find_unique_cycles(diagram):
    """
    Finds unique cycles for an input diagram.

    Parameters
    ----------
    diagram : NetworkX MultiDiGraph object
        Diagram of interest.

    Returns
    -------
    unique_cycles : list of lists of int
        List of cycles, where each cycle is a list of nodes in that cycle.
    """
    cycles = [c for c in nx.simple_cycles(diagram) if len(c) > 2]
    unique_cycles = []
    unique_cycles_ordered = []
    for cycle in cycles:
        ordered_cycle = set(cycle)
        if not ordered_cycle in unique_cycles_ordered:
            unique_cycles_ordered.append(ordered_cycle)
            unique_cycles.append(cycle)
    unique_cycles = [list(c) for c in unique_cycles]
    return unique_cycles

def construct_cycle_edges(cycle):
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

def get_ordered_cycle(G, cycle):
    """
    Takes in arbitrary list of nodes and returns list of nodes in correct order.
    Can be used in conjunction with construct_cycle_edges() to generate list of
    edge tuples for an arbitrary input cycle. Assumes input cycle only exists
    once in the input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    ordered_cycle : list of int
        Ordered list of integers for the input cycle.
    """
    ordered_cycles = find_unique_cycles(G)
    for ordered_cycle in ordered_cycles:
        if set(ordered_cycle) == set(cycle):
            return ordered_cycle

def generate_two_way_flux_diagrams(G):
    """
    Creates two-way flux diagrams for the input diagram G. Created by adding one
    more edge per diagram than are added for partial diagrams.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram

    Returns
    -------
    valid_flux_diags : list of NetworkX MultiDiGraph objects
        List of diagrams with the same number of unique edges as nodes, and
        only 1 complete cycle.
    """
    # Calculate number of edges needed for each flux diagram
    N_flux_edges = G.number_of_nodes()
    # Get list of all possible combinations of unique edges (N choose n)
    combinations = list(itertools.combinations(find_unique_edges(G), N_flux_edges))
    # Using combinations, generate all possible partial diagrams (including closed loops)
    flux_diags_all = []
    for i in combinations:
        diag = G.copy()
        diag.remove_edges_from(list(G.edges()))
        edges = []
        for j in i:
            edges.append((j[0], j[1]))  # Add edge from combinations
            edges.append((j[1], j[0]))  # Add reverse edge
        diag.add_edges_from(edges)
        flux_diags_all.append(diag)
    # Remove unwanted diagrams (more than 1 closed loop)
    valid_flux_diags = []
    for diag in flux_diags_all:
        n_cycles = len(find_unique_cycles(diag))
        if n_cycles == 1:
            valid_flux_diags.append(diag)
    return valid_flux_diags

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
    flux_diagram : NetworkX MultiDiGraph object
        Flux diagram returned in the event that there is only one directional
        flux diagram for the input cycle in the input diagram G. Cycle nodes
        are labeled by attribute 'is_target'.
    directional_flux_diagrams : list of NetworkX MultiDiGraph objects
        List of directional flux diagrams. Diagrams contain the input cycle
        where remaining edges follow path pointing to cycle. Cycle nodes are
        labeled by attribute 'is_target'.
    """
    two_way_flux_diagrams = generate_two_way_flux_diagrams(G)
    cycle_idx = get_indices_of_flux_diagrams(cycle, two_way_flux_diagrams)
    relevant_flux_diags = [two_way_flux_diagrams[i] for i in cycle_idx]
    if len(relevant_flux_diags) == 1:
        print("Only 1 flux diagram detected for cycle ({}). Sigma K value is 1.".format(cycle))
        for target in relevant_flux_diags[0].nodes():
            flux_diagram = relevant_flux_diags[0]
            if target in cycle:
                flux_diagram.nodes[target]['is_target'] = True
            else:
                flux_diagram.nodes[target]['is_target'] = False
        return flux_diagram
    else:
        diag_cycles = [c for c in list(nx.simple_cycles(relevant_flux_diags[0])) if len(c) > 2]
        cycle_edges = [construct_cycle_edges(cyc) for cyc in diag_cycles]
        cycle_edges_flat = [edge for edges in cycle_edges for edge in edges]
        directional_flux_diagrams = []
        for diagram in relevant_flux_diags:
            diag = diagram.copy()
            edges = find_unique_edges(diag)
            unique_edges = [edge for edge in edges if not edge in cycle_edges_flat]
            dir_edges = []
            for target in cycle:
                cons = get_directional_connections(target, unique_edges)
                if not len(cons) == 0:
                    dir_edges.append(get_directional_edges(cons))
            dir_edges_flat = [edge for edges in dir_edges for edge in edges]
            diag.remove_edges_from(list(diag.edges()))
            for edge in dir_edges_flat:
                diag.add_edge(edge[0], edge[1], edge[2])
            for edge in cycle_edges_flat:
                diag.add_edge(edge[0], edge[1], 0)
            for target in diag.nodes():
                if target in cycle:
                    diag.nodes[target]['is_target'] = True
                else:
                    diag.nodes[target]['is_target'] = False
            directional_flux_diagrams.append(diag)
        return directional_flux_diagrams

def calc_sigma(G, dir_partials, key, output_strings=False):
    """
    Calculates sigma, the normalization factor for calculating state
    probabilities and cycle fluxes for a given diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for the input diagram G.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic cycle flux function.
    Returns
    -------
    sigma : float
        Normalization factor for state probabilities.
    sigma_str : str
        Sum of rate products of all directional partial diagrams for input
        diagram G, in string form.
    """
    N = G.number_of_nodes() # Number of nodes/states
    partial_mults = []
    edges = list(G.edges)
    if output_strings == False:
        if isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise Exception("To enter variable strings set parameter output_strings=True.")
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = 1          # assign initial value of 1
            for e in edge_list:                                 # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]][key]    # multiply the rate of each edge in edge_list
            partial_mults.append(products)
        sigma = np.array(partial_mults).sum(axis=0)
        return sigma
    elif output_strings == True:
        if not isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise Exception("To enter variable values set parameter output_strings=False.")
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = []
            for e in edge_list:
                products.append(G.edges[e[0], e[1], e[2]][key]) # append rate constant names from dir_par to list
            partial_mults.append(products)
        state_mults = []
        term_list = []  # create empty list to put products of rate constants (terms) in
        for k in partial_mults:
            term_list.append("*".join(k))    # join rate constants for each dir_par by delimeter "*"
        sigma_str = "+".join(term_list)    # sum all terms to get normalization factor
        return sigma_str

def calculate_sigma_K(G, cycle, flux_diags, key, output_strings=False):
    """
    Calculates sigma_K, the sum of all directional flux diagrams.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    flux_diags : list
        List of relevant directional flux diagrams for input cycle.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    sigma_K : float
        Sum of rate products of directional flux diagram edges pointing to
        input cycle.
    sigma_K_str : str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle in string form.
    """
    if isinstance(flux_diags, list) == False:
        print("Only 1 flux diagram detected for cycle ({}). Sigma K value is 1.".format(cycle))
        return 1
    else:
        ordered_cycle = get_ordered_cycle(G, cycle)
        cycle_edges = construct_cycle_edges(ordered_cycle)
        if output_strings == False:
            if isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
                raise Exception("To enter variable strings set parameter output_strings=True.")
            rate_products = []
            for diagram in flux_diags:
                diag = diagram.copy()
                for edge in cycle_edges:
                    diag.remove_edge(edge[0], edge[1], edge[2])
                    diag.remove_edge(edge[1], edge[0], edge[2])
                vals = 1
                for edge in diag.edges:
                    vals *= G.edges[edge[0], edge[1], edge[2]][key]
                rate_products.append(vals)
            sigma_K = np.array(rate_products).sum(axis=0)
            return sigma_K
        elif output_strings == True:
            if not isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
                raise Exception("To enter variable values set parameter output_strings=False.")
            rate_products = []
            for diagram in flux_diags:
                diag = diagram.copy()
                for edge in cycle_edges:
                    diag.remove_edge(edge[0], edge[1], edge[2])
                    diag.remove_edge(edge[1], edge[0], edge[2])
                rates = []
                for edge in diag.edges:
                    rates.append(G.edges[edge[0], edge[1], edge[2]][key])
                rate_products.append("*".join(rates))
            sigma_K_str = "+".join(rate_products)
            return sigma_K_str

def calculate_pi_difference(G, cycle, key, output_strings=False):
    """
    Calculates the difference of the forward and reverse rate products for a
    given cycle, where forward rates are defined as counter clockwise.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    pi_diff : float
        Difference of product of counter clockwise cycle rates and clockwise
        cycle rates.
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    """
    cycle_edges = construct_cycle_edges(get_ordered_cycle(G, cycle))
    if output_strings == False:
        if isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
            raise Exception("To enter variable strings set parameter output_strings=True.")
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        pi_difference = ccw_rates - cw_rates
        return pi_difference
    elif output_strings == True:
        if not isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
            raise Exception("To enter variable values set parameter output_strings=False.")
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        pi_difference = "-".join(["*".join(ccw_rates), "*".join(cw_rates)])
        return pi_difference

def calc_state_probs(G, key, output_strings=False):
    """
    Calculates state probabilities directly.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic state multplicity and normalization function.

    Returns
    -------
    state_probs : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    state_mults : list of str
        List of analytic state multiplicity functions in string form.
    norm : str
        Analytic state multiplicity function normalization function in
        string form. This is the sum of all multiplicty functions.
    """
    dir_pars = generate_directional_partial_diagrams(G)
    if output_strings == False:
        state_probs = calc_state_probabilities(G, dir_pars, key, output_strings=output_strings)
        return state_probs
    if output_strings == True:
        state_mults, norm = calc_state_probabilities(G, dir_pars, key, output_strings=output_strings)
        return state_mults, norm

def calc_cycle_flux(G, cycle, key, output_strings=False):
    """
    Calculates cycle flux for a given cycle in diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic cycle flux function.

    Returns
    -------
    cycle_flux : float
        Cycle flux for input cycle.
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    sigma_K : str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle in string form.
    sigma_str : str
        Sum of rate products of all directional partial diagrams for input
        diagram G, in string form.
    """
    dir_pars = generate_directional_partial_diagrams(G)
    flux_diags = generate_flux_diagrams(G, cycle)
    if output_strings == False:
        pi_diff = calculate_pi_difference(G, cycle, key, output_strings=output_strings)
        sigma_K = calculate_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma = calc_sigma(G, dir_pars, key, output_strings=output_strings)
        cycle_flux = pi_diff*sigma_K/sigma
        return cycle_flux
    if output_strings == True:
        pi_diff_str = calculate_pi_difference(G, cycle, key, output_strings=output_strings)
        sigma_K_str = calculate_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma_str = calc_sigma(G, dir_pars, key, output_strings=output_strings)
        return pi_diff_str, sigma_K_str, sigma_str

def construct_sympy_cycle_flux_funcs(pi_diff_str, sigma_K_str, sigma_str):
    """
    Creates the analytic cycle flux SymPy function for a given cycle.

    Parameters
    ----------
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    sigma_K_str : str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle in string form.
    sigma_str : str
        Sum of rate products of all directional partial diagrams for input
        diagram G, in string form.

    Returns
    -------
    cycle_flux_func : SymPy object
        Analytic cycle flux SymPy function
    """
    if sigma_K_str == 1:
        cycle_flux_func = parse_expr(pi_diff_str)/parse_expr(sigma_str)
        return cycle_flux_func
    else:
        cycle_flux_func = (parse_expr(pi_diff_str)*parse_expr(sigma_K_str))/parse_expr(sigma_str)
        return cycle_flux_func


def SVD(K, tol=1e-12):
    """
    Calculates the steady-state probabilities for an N-state model using
    singular value decomposition.

    Parameters
    ----------
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input K matrix.
    tol : float (optional)
        Tolerance used for singular value determination. Values are considered
        singular if they are less than the input tolerance. Default is 1e-12.

    Returns
    -------
    state_probs : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    """
    N = len(K)                  # get number of states
    Kc = K.copy()                # Make a copy of input matrix K
    np.fill_diagonal(Kc, 0)      # fill the diagonal elements with zeros
    Kc = Kc.T                    # take the transpose
    for i in range(N):
        Kc[i, i] = -Kc[:, i].sum(axis=0)    # set the diagonal elements equal to the negative sum of the columns
    prob_norm = np.ones(N)                  # create array of ones
    Kcs = np.vstack((Kc, prob_norm))        # stack ODE equations with probability equation
    U, w, VT = np.linalg.svd(Kcs, full_matrices=False)  # use SVD to generate U, w, and V.T matrices
    singular_vals = np.abs(w) < tol                     # find any singular values in w matrix
    inv_w = 1/w                                         # Take the inverse of the w matrix
    inv_w[singular_vals] = 0                            # Set any singular values to zero
    Kcs_inv = VT.T.dot(np.diag(inv_w)).dot(U.T)         # construct the pseudo inverse of Kcs
    pdot = np.zeros(N+1)                                # create steady state solution matrix (pdot = 0), add additional entry for probaility equation
    pdot[-1] = 1                                        # set last value to 1 for probability normalization
    state_probs = Kcs_inv.dot(pdot)                     # dot Kcs and pdot matrices together
    return state_probs
