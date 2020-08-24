# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis (kda)
=========================================================================
This file contains a host of functions aimed at the analysis of biochemical
kinetic diagrams, using the methods of Hill.

.. autofunction:: generate_edges
.. autofunction:: generate_partial_diagrams
.. autofunction:: generate_directional_partial_diagrams
.. autofunction:: generate_flux_diagrams
.. autofunction:: generate_all_flux_diagrams
.. autofunction:: calc_state_probs
.. autofunction:: calc_cycle_flux
.. autofunction:: calculate_sigma
.. autofunction:: calculate_sigma_K
.. autofunction:: calculate_pi_difference
.. autofunction:: calculate_thermo_force
.. autofunction:: calc_state_probabilities
.. autofunction:: construct_cycle_edges
.. autofunction:: construct_sympy_prob_funcs
.. autofunction:: construct_sympy_cycle_flux_func
.. autofunction:: construct_lambdify_funcs
.. autofunction:: solve_ODE
.. autofunction:: SVD
.. autofunction:: add_node_attribute
.. autofunction:: add_graph_attribute
.. autofunction:: find_node_edges
.. autofunction:: find_unique_edges
.. autofunction:: find_uncommon_edges
.. autofunction:: find_all_unique_cycles
.. autofunction:: find_unique_uncommon_edges
.. autofunction:: get_ordered_cycle
.. autofunction:: get_CCW_cycle
.. autofunction:: get_directional_edges
.. autofunction:: get_directional_connections
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
    tuples = [(sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))]   # Make list of edge tuples
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
        reverse_cycle = [cycle[0]] + cycle[len(G.nodes):0:-1]
        if not reverse_cycle in temp:
            unique_cycles.append(cycle)
    if len(unique_cycles) == 1:
        print("Only 1 cycle found: {}".format(unique_cycles[0]))
    return unique_cycles

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
    ordered_cycles : list of int, list of lists of int
        Ordered list of nodes for the input cycle, or if several cycles are
        found, a list of lists of nodes for the input cycle.
    """
    ordered_cycles = []
    for cyc in find_all_unique_cycles(G):
        if sorted(cyc) == sorted(cycle):
            ordered_cycles.append(cyc)
    if ordered_cycles == []:
        print("No cycles found for nodes {}.".format(cycle))
    elif len(ordered_cycles) > 1:
        return ordered_cycles
    else:
        return ordered_cycles[0]

def is_CCW(cycle, start, end):
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
    double = 2*cycle
    for i in range(len(double)-1):
        if (double[i], double[i+1]) == (start, end):
            return True
    return None

def get_CCW_cycle(cycle, order):
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
    CCW = is_CCW(cycle, order[0], order[1])
    if CCW == True:
        return cycle
    elif not CCW:
        return cycle[::-1]
    else:
        raise Exception("Direction of cycle {} could not be determined.".format(cycle))

def append_reverse_edges(edge_list):
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

def find_uncommon_edges(edges1, edges2):
    """
    Function for removing edges when both lists contain forward and reverse
    edges. All input edges should be either 2-tuples or 3-tuples.

    Parameters
    ----------
    edges1 : list of tuples
        List of edge tuples to be compared. This list should be the longer list
        of the two.
    edges2 : list of tuples
        List of edge tuples to be compared. This list should be the shorter
        list of the two.

    Returns
    -------
    uncommon_edges : list of tuples
        List of uncommon edges between input list "edges1" and "edges2".
    """
    all_uncommon_edges = []
    for edge in edges1:
        if not edge in edges2:
            all_uncommon_edges.append(edge)
    uncommon_edges = []
    temp = []
    for edge in all_uncommon_edges:
        temp.append((edge[1], edge[0], 0))
        if not edge in temp:
            uncommon_edges.append(edge)
    return uncommon_edges

def find_unique_uncommon_edges(G_edges, cycle):
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
    cycle_edges = construct_cycle_edges(cycle)
    sorted_G_edges = [sorted(edge) for edge in G_edges]
    sorted_cycle_edges = [sorted(edge) for edge in cycle_edges]
    non_cycle_edges = [tuple(edge) for edge in sorted_G_edges if not edge in sorted_cycle_edges]
    node_edges = find_node_edges(cycle) # generate edges that only contain 2 nodes
    valid_non_cycle_edges = [edge for edge in non_cycle_edges if not edge in node_edges] # remove node edges
    return valid_non_cycle_edges

def find_node_edges(cycle, r=2):
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
    node_edges = list(itertools.combinations(cycle, r)) # generate node edges
    for edge in node_edges.copy():
        node_edges.append((edge[1], edge[0])) # append reverse edge
    return node_edges

def flux_edge_conditions(edge_list, N):
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
    tuples = [(sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))]
    unique_edges = list(set(tuples))
    if len(edge_list) == N:  # the number of edges must equal the number of edges in the list
        if len(unique_edges) == len(edge_list):
            return True
        else:
            return False
    else:
        return False

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
        print("""Cycle {} contains all nodes in G, no flux
        diagrams can be generated. Value of None Returned.""".format(cycle))
    else:
        cycle_edges = construct_cycle_edges(cycle)
        G_edges = find_unique_edges(G)
        non_cycle_edges = find_unique_uncommon_edges(G_edges, cycle) # get edges that are uncommon between cycle and G
        N = G.number_of_nodes() - len(cycle_edges) # number of non-cycle edges in flux diagram
        flux_edge_lists = list(itertools.combinations(non_cycle_edges, r=N)) # all combinations of valid edges
        # generates too many edge lists: some create cycles, some use both forward and reverse edges
        flux_diagrams = []
        for edge_list in flux_edge_lists:
            dir_edges = []
            for target in cycle:
                cons = get_directional_connections(target, edge_list)
                if not len(cons) == 0:
                    dir_edges.append(get_directional_edges(cons))
            dir_edges_flat = [edge for edges in dir_edges for edge in edges]
            if flux_edge_conditions(dir_edges_flat, N) == True:
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
                            diag.nodes[target]['is_target'] = True
                        else:
                            diag.nodes[target]['is_target'] = False
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
    all_cycles = find_all_unique_cycles(G)
    all_flux_diagrams = []
    for cycle in all_cycles:
        flux_diagrams = generate_flux_diagrams(G, cycle)
        if flux_diagrams == None:
            continue
        else:
            for diag in flux_diagrams:
                if len(find_all_unique_cycles(diag)) == 1: # check if there is only 1 unique cycle
                    continue
                else:
                    raise Exception("Flux diagram has more than 1 closed loop for cycle {}.".format(cycle))
            all_flux_diagrams.append(flux_diagrams)
    return all_flux_diagrams

def calculate_sigma(G, dir_partials, key, output_strings=False):
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
        is False, which tells the function to calculate the normalization factor
        using numbers. If True, this will assume the input
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
        indices does not matter but should not contain all nodes.
    flux_diags : list
        List of relevant directional flux diagrams for input cycle.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the sum of all
        directional flux diagrams using numbers. If True, this will assume the
        input 'key' will return strings of variable names to join into the
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
        print("No flux diagrams detected for cycle {}. Sigma K value is 1.".format(cycle))
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

def calculate_pi_difference(G, cycle, order, key, output_strings=False):
    """
    Calculates the difference of the forward and reverse rate products for a
    given cycle, where forward rates are defined as counter clockwise.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2, where the integers must be nodes in the
        input cycle. This pair of nodes is used to determine which direction is
        CCW.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the difference using
        numbers. If True, this will assume the input 'key' will return strings
        of variable names to join into the analytic function.

    Returns
    -------
    pi_diff : float
        Difference of product of counter clockwise cycle rates and clockwise
        cycle rates.
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    """
    cycle_count = 0
    for cyc in find_all_unique_cycles(G):
        if sorted(cycle) == sorted(cyc):
            cycle_count += 1
    if cycle_count > 1:     # for all-node cycles
        CCW_cycle = get_CCW_cycle(cycle, order)
        cycle_edges = construct_cycle_edges(CCW_cycle)
    elif cycle_count == 1:  # for all other cycles
        ordered_cycle = get_ordered_cycle(G, cycle)
        CCW_cycle = get_CCW_cycle(ordered_cycle, order)
        cycle_edges = construct_cycle_edges(CCW_cycle)
    else:
        raise Exception("Cycle {} could not be found in G.".format(cycle))
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

def calculate_thermo_force(G, cycle, order, key, output_strings=False):
    """
    Calculates the thermodynamic driving force for a given cycle in diagram G.
    The driving force is calculated as the natural log of the ratio of the
    forward rate product and the reverse rate product in the cycle, where the
    forward direction is defined as counter clockwise. The value returned should
    be multiplied by 'kT' to obtain the actual thermodynamic force, in units of
    energy.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2, where the integers must be nodes in the
        input cycle. This pair of nodes is used to determine which direction is
        CCW.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the thermodynamic force
        using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    thermo_force : float
        The calculated thermodynamic force for the input cycle. This value is
        unitless and should be multiplied by 'kT'.
    parsed_thermo_force_str : SymPy function
        The thermodynamic force equation in SymPy function form. Should be
        multiplied by 'kT' to get actual thermodynamic force.
    """
    cycle_count = 0
    for cyc in find_all_unique_cycles(G):
        if sorted(cycle) == sorted(cyc):
            cycle_count += 1
    if cycle_count > 1:     # for all-node cycles
        CCW_cycle = get_CCW_cycle(cycle, order)
        cycle_edges = construct_cycle_edges(CCW_cycle)
    elif cycle_count == 1:  # for all other cycles
        ordered_cycle = get_ordered_cycle(G, cycle)
        CCW_cycle = get_CCW_cycle(ordered_cycle, order)
        cycle_edges = construct_cycle_edges(CCW_cycle)
    else:
        raise Exception("Cycle {} could not be found in G.".format(cycle))
    if output_strings == False:
        if isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
            raise Exception("To enter variable strings set parameter output_strings=True.")
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        thermo_force = np.log(ccw_rates/cw_rates)
        return thermo_force
    elif output_strings == True:
        if not isinstance(G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str):
            raise Exception("To enter variable values set parameter output_strings=False.")
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        thermo_force_str = 'ln(' + "*".join(ccw_rates) + ') - ln(' + "*".join(cw_rates) + ')'
        parsed_thermo_force_str = logcombine(parse_expr(thermo_force_str), force=True)
        return parsed_thermo_force_str

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
    state_probs_sympy : SymPy object
        List of analytic SymPy state probability functions.
    """
    dir_pars = generate_directional_partial_diagrams(G)
    if output_strings == False:
        state_probs = calc_state_probabilities(G, dir_pars, key, output_strings=output_strings)
        return state_probs
    if output_strings == True:
        state_mults, norm = calc_state_probabilities(G, dir_pars, key, output_strings=output_strings)
        state_probs_sympy = construct_sympy_prob_funcs(state_mults, norm)
        return state_probs_sympy

def calc_cycle_flux(G, cycle, order, key, output_strings=False):
    """
    Calculates cycle flux for a given cycle in diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram.
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the cycle flux using
        numbers. If True, this will assume the input 'key' will return strings
        of variable names to join into the analytic cycle flux function.

    Returns
    -------
    cycle_flux : float
        Cycle flux for input cycle.
    cycle_flux_func : SymPy object
        Analytic cycle flux SymPy function.
    """
    dir_pars = generate_directional_partial_diagrams(G)
    flux_diags = generate_flux_diagrams(G, cycle)
    if output_strings == False:
        pi_diff = calculate_pi_difference(G, cycle, order, key, output_strings=output_strings)
        sigma_K = calculate_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma = calculate_sigma(G, dir_pars, key, output_strings=output_strings)
        cycle_flux = pi_diff*sigma_K/sigma
        return cycle_flux
    if output_strings == True:
        pi_diff_str = calculate_pi_difference(G, cycle, order, key, output_strings=output_strings)
        sigma_K_str = calculate_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma_str = calculate_sigma(G, dir_pars, key, output_strings=output_strings)
        sympy_cycle_flux_func = construct_sympy_cycle_flux_func(pi_diff_str, sigma_K_str, sigma_str)
        return sympy_cycle_flux_func

def construct_sympy_cycle_flux_func(pi_diff_str, sigma_K_str, sigma_str):
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
