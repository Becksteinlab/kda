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

import functools

import itertools

import sympy

from sympy import *

from sympy.parsing.sympy_parser import parse_expr


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

def generate_directional_partial_diagrams(partials):
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
    N_targets = partials[0].number_of_nodes()
    dir_partials = []
    for target in range(N_targets):
        for i in range(len(partials)):
            diag = partials[i].copy()               # Make a copy of directional partial diagram
            unique_edges = find_unique_edges(diag)      # Find unique edges of that diagram
            cons = get_directional_connections(target, unique_edges)   # Get dictionary of connections
            dir_edges = get_directional_edges(cons)                    # Get directional edges from connections
            diag.remove_edges_from(list(diag.edges()))
            for e in dir_edges:
                diag.add_edge(e[0], e[1], e[2])
            dir_partials.append(diag)
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
    for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
        edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
        if output_strings == False:
            products = 1          # generate an array with a value of 1
            for e in edge_list:                                 # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]][key]    # multiply the rate of each edge in edge_list
        if output_strings == True:
            products = []
            for e in edge_list:
                products.append(G.edges[e[0], e[1], e[2]][key]) # append rate constant names from dir_par to list
        partial_mults.append(products)
    N_terms = np.int(len(dir_partials)/N) # calculate the number of terms to be summed for given state, s
    state_mults = []
    if output_strings == False:
        partial_mults = np.array(partial_mults)
        for s in range(N):    # iterate over number of states, "s"
            state_mults.append(partial_mults[N_terms*s:N_terms*s+N_terms].sum(axis=0))
        state_mults = np.array(state_mults)
        state_probs = state_mults/state_mults.sum(axis=0)
        return state_probs
    if output_strings == True:
        term_list = []  # create empty list to put products of rate constants (terms) in
        for vars in partial_mults:
            term_list.append("*".join(vars))    # join rate constants for each dir_par by delimeter "*"
        for s in range(N):    # iterate over number of states, "s"
            state_mults.append("+".join(term_list[N_terms*s:N_terms*s+N_terms]))    # join appropriate terms for each state by delimeter "+"
        norm = "+".join(state_mults)    # sum all terms to get normalization factor
        return state_mults, norm

def construct_sympy_funcs(state_mult_funcs, norm_func):
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
    Constructs lambdified analytic state probability functions.

    Parameters
    ----------
    sympy_funcs : list of SymPy functions
        List of analytic state probability SymPy functions.
    rate_names : list
        List of strings, where each element is the name of the variables for
        the input probability functions, ["x12", "x21", "x23", ...].

    Returns
    -------
    state_prob_funcs : list
        List of lambdified analytic state probability functions.
    """
    state_prob_funcs = []   # create empty list to fill with state probability functions
    for func in sympy_funcs:
        state_prob_funcs.append(lambdify(rate_names, func, "numpy"))    # convert into "lambdified" functions that work with NumPy arrays
    return state_prob_funcs

def generate_edges(G, names, vals, name_key='name', val_key='val'):
    """
    Generate edges with attributes 'name' and 'val'.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    names : array
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the names of all of the attributes corresponding to the values in
        'vals' as strings, i.e. [[0, "k12"], ["k21", 0]].
    vals : array
        'NxN' array where 'N' is the number of nodes in the diagram G. Contains
        the values associated with the attribute names in 'names'. For example,
        assuming k12 and k21 had already been assigned values, for a 2 state
        diagram 'vals' = [[0, k12], [k21, 0]].
    name_key : str (optional)
        Key used to retrieve variable names in 'names'. Default is 'name'.
    val_key : str (optional)
        Key used to retrieve variable values in 'vals'. Default is 'val'.
    """
    for i, row in enumerate(vals):
        for j, elem in enumerate(row):
            if not elem == 0:
                attrs = {name_key : names[i, j], val_key : elem}
                G.add_edge(i, j, **attrs)

def add_node_attribute(G, vals, label):
    """
    Sequentially add attributes to nodes from array of values, i.e. state
    probabilities.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    vals : array like
        Array or list of values of length 'n', where 'n' is the number of nodes
        in the diagram G. Elements must be in order, i.e. [x1, x2, x3, ..., xn]
    label : str
        Name of new attribute to be assigned to nodes.
    """
    for i in range(G.number_of_nodes()):
        G.nodes[i][label] = vals[i]

def solve_ODE(P, K, t_max, method='RK45', t_eval=None,
              dense_output=False, events=None, vectorized=False):
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
    max_step : int
        Maximum step size for integrator.

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

    def func(t, y):
        """
        y = [p1, p2, p3, ... , pn]
        """
        return np.dot(k, y)

    k = convert_K(K)
    time = (0, t_max)
    y0 = np.array(P, dtype=np.float64)
    return scipy.integrate.solve_ivp(fun=func, t_span=time, y0=y0,
                                     method=method, t_eval=t_eval,
                                     dense_output=dense_output,
                                     events=events, vectorized=vectorized)

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
