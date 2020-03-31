# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Kinetic Diagram Analyzer

import numpy as np
import networkx as nx
import scipy.integrate
import functools
import itertools
import sympy
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

#===============================================================================
#== Functions ==================================================================
#===============================================================================

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
            [diag.add_edge(e[0], e[1], e[2]) for e in dir_edges]
            dir_partials.append(diag)
    return dir_partials


def calc_state_probabilities(G, dir_partials, key='k'):
    """
    Calculates state probabilities for N states in diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    key : str (optional)
        Definition of key in NetworkX diagram edges, used to call rate values.
        Default is 'k'. This needs to match the key used for the rate constants
        in the input diagram G.

    Returns
    -------
    state_probabilities : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    """
    N = G.number_of_nodes() # Number of nodes/states
    state_multiplicities = np.zeros(N)
    for s in range(N):    # iterate over number of states, "s"
        partial_multiplicities = np.zeros(len(dir_partials))    # generate zero array of length # of directional partial diagrams
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = np.array([1])          # generate an array with a value of 1
            for e in edge_list:                                 # iterate over the edges in the given directional partial diagram i
                products *= G[e[0]][e[1]][e[2]][key]     # multiply the rate of each edge in edge_list
            partial_multiplicities[i] = products[0]                 # for directional partial diagram i, assign product to partial multiplicity array
        N_terms = np.int(len(dir_partials)/N) # calculate the number of terms to be summed for given state, s
        for j in range(N):                            # iterate over number of states
             # sum appropriate parts of partial multiplicity array for given state, s
            state_multiplicities[j] = partial_multiplicities[N_terms*j:N_terms*j+N_terms].sum(axis=0)
            # this gives you an array of all the state multiplicites
    # calculate the state probabilities by normalizing over the sum of all state multiplicites
    state_probabilities = state_multiplicities/state_multiplicities.sum(axis=0)
    return state_probabilities


def construct_string_funcs(G, dir_partials, rates, rate_names, key='k'):
    """
    Constructs analytic state multiplicity and normalization function strings
    for the input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    rates : list
        List of rate values associated with the edges of the input diagram G.
        Each element should be the corresponding value for the input list of
        strings 'rate_names', [x12, x21, x23...].
    rate_names : list
        List of strings, where each element is the name of the variable in the
        input list 'rates', ["x12", "x21", "x23", ...].
    key : str (optional)
        Definition of key in NetworkX diagram edges, used to call rate values.
        Default is 'k'. This needs to match the key used for the rate constants
        in the input diagram G.

    Returns
    -------
    state_mult_funcs : list of str
        List of length 'N', where N is the number of states, that contains the
        analytic multiplicity function for each state
    norm_func : str
        Sum of all state multiplicity functions, the normalization factor to
        calculate the state probabilities
    """
    N = G.number_of_nodes() # Number of nodes/states
    var_dict = dict.fromkeys(rates, {}) # generate dictionary with rates as keys
    for i in range(len(rates)):
        var_dict[rates[i]] = rate_names[i]  # assign appropriate rate names to rate values
    state_mult_funcs = []    # create empty list to fill with summed terms
    for s in range(N):    # iterate over number of states, "N"
        part_mults = []    # generate empty list to put partial multiplicities in
        for i in range(len(dir_partials)):      # iterate over the directional partial diagrams
            edge_list = list(dir_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = []          # generate an empty list to store individual variable names for each product
            for edge in edge_list:
                products.append(var_dict[G.edges[edge[0], edge[1], edge[2]][key]]) # append rate constant names from dir_par to list
            part_mults.append(products)     # append list of rate constant names to part_mults
    N_terms = int(len(part_mults)/N) # number of terms per state
    term_list = []  # create empty list to put products of rate constants (terms) in
    for vars in part_mults:
        term_list.append("*".join(vars))    # join rate constants for each dir_par by delimeter "*"
    for j in range(N):
        state_mult_funcs.append("+".join(term_list[N_terms*j:N_terms*j+N_terms]))    # join appropriate terms for each state by delimeter "+"
    norm_func = "+".join(state_mult_funcs)    # sum all terms to get normalization factor
    return state_mult_funcs, norm_func


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


def calc_cycle_fluxes(dir_pars):
    return NotImplementedError


def assign_probs_and_analytic_functions_to_G(dir_pars):
    return NotImplementedError


def solve_ODE(P, K, t_max, max_step):
    """
    Integrates state probabilities to find steady state probabilities.

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
    return scipy.integrate.solve_ivp(func, time, y0, max_step=max_step)
