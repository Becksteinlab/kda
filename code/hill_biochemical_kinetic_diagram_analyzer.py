# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Hill Biochemical Kinetic Diagram Analyzer

import numpy as np
import networkx as nx
import functools
import itertools
import sympy as sp
from sympy import *

#===============================================================================
#== Functions ==================================================================
#===============================================================================

def find_unique_edges(G):
    edges = list(G.edges)           # Get list of edges
    sorted_edges = np.sort(edges)      # Sort list of edges
    tuples = [(sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))]   # Make list of edges tuples
    return list(set(tuples))


def combine(x,y):
    x.update(y)
    return x


def generate_directional_connections(target, unique_edges):
    edges = [i for i in unique_edges if target in i]    # Find edges that connect to target state
    neighbors = [[j for j in i if not j == target][0] for i in edges] # Find states neighboring target state
    if not neighbors:
        return {}
    unique_edges = [k for k in unique_edges if not k in edges]  # Make new list of unique edges that does not contain original unique edges
    return functools.reduce(combine, [{target: neighbors}] + [generate_directional_connections(i, unique_edges) for i in neighbors])


def generate_directional_edges(cons):
    values = []
    for i in cons.keys():
        for j in cons[i]:
            values.append((j, i, 0))
    return values


def generate_partial_diagrams(G):
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
    N_targets = partials[0].number_of_nodes()
    dir_par_diags = []
    for target in range(N_targets):
        for i in range(len(partials)):
            diag = partials[i].copy()               # Make a copy of directional partial diagram
            unique_edges = find_unique_edges(diag)      # Find unique edges of that diagram
            cons = generate_directional_connections(target, unique_edges)   # Get dictionary of connections
            dir_edges = generate_directional_edges(cons)                    # Get directional edges from connections
            diag.remove_edges_from(list(diag.edges()))
            [diag.add_edge(e[0], e[1], e[2]) for e in dir_edges]
            dir_par_diags.append(diag)
    return dir_par_diags


def calc_state_probabilities(G, directional_partials, state_mults=None):
    state_multiplicities = np.zeros(G.number_of_nodes())
    for s in range(G.number_of_nodes()):    # iterate over number of states, "s"
        partial_multiplicities = np.zeros(len(directional_partials))    # generate zero array of length # of directional partial diagrams
        for i in range(len(directional_partials)):      # iterate over the directional partial diagrams
            edge_list = list(directional_partials[i].edges)     # get a list of all edges for partial directional diagram i
            products = np.array([1])          # generate an array with a value of 1
            for e in edge_list:                                 # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]]['weight']     # multiply the weight of each edge in edge_list
            partial_multiplicities[i] = products[0]                 # for directional partial diagram i, assign product to partial multiplicity array
        N_terms = np.int(len(directional_partials)/G.number_of_nodes()) # calculate the number of terms to be summed for given state, s
        for j in range(G.number_of_nodes()):                            # iterate over number of states
             # sum appropriate parts of partial multiplicity array for given state, s
            state_multiplicities[j] = partial_multiplicities[N_terms*j:N_terms*j+N_terms].sum(axis=0)
            # this gives you an array of all the state multiplicites
    # calculate the state probabilities by normalizing over the sum of all state multiplicites
    state_probabilities = state_multiplicities/state_multiplicities.sum(axis=0)
    if state_mults == True:
        return state_probabilities, state_multiplicities
    else:
        return state_probabilities


def generate_rate_dict(rates, rate_names):
    """
    Generates dictionary where rate constant values are keys and rate constant
    names are the values, or vice-versa.
    """
    var_dict = dict.fromkeys(rates, {})
    for i in range(len(rates)):
        var_dict[rates[i]] = rate_names[i]
    return var_dict


def construct_analytic_functions(G, dir_parts, var_dict, rate_names, sym_funcs=None):
    """
    This function will input a list of all directional partial diagrams and
    output a list of analytic functions for the steady-state probability of
    each state in the original diagram.

    Parameters
    ----------
    G : networkx diagram object
    dir_parts : list of networkx diagram objects
        List of all directional partial diagrams for the given diagram "G"
    var_dict : dict
        Dictionary where the rate constant values are the keys and the rate
        constant names are the values
    rate_names : list
        List of strings of variable names for the model ["x12", "x21", "x23"...]

    Returns
    -------
    state_prob_funcs : list
        List of lambdified functions for each state [p1, p2, p3,...pn]
    state_mult_funcs : list
        List of length 'N', where N is the number of states, that contains the
        analytic multiplicity function for each state
    norm_func : str
        Sum of all state multiplicity functions, the normalization factor to
        calculate the state probabilities
    """
    state_mult_funcs = []    # create empty list to fill with summed terms
    for s in range(G.number_of_nodes()):    # iterate over number of states, "s"
        part_mults = []    # generate empty list to put partial multiplicities in
        for i in range(len(dir_parts)):      # iterate over the directional partial diagrams
            edge_list = list(dir_parts[i].edges)     # get a list of all edges for partial directional diagram i
            products = []          # generate an empty list to store individual variable names for each product
            for edge in edge_list:
                products.append(var_dict[G.edges[edge[0], edge[1], edge[2]]['weight']]) # append rate constant names from dir_par to list
            part_mults.append(products)     # append list of rate constant names to part_mults
    N_terms = int(len(part_mults)/G.number_of_nodes()) # number of terms per state
    term_list = []  # create empty list to put products of rate constants (terms) in
    for vars in part_mults:
        term_list.append("*".join(vars))    # join rate constants for each dir_par by delimeter "*"
    for j in range(G.number_of_nodes()):
        state_mult_funcs.append("+".join(term_list[N_terms*j:N_terms*j+N_terms]))    # join appropriate terms for each state by delimeter "+"
    norm_func = "+".join(state_mult_funcs)    # sum all terms to get normalization factor
    state_prob_funcs = []   # create empty list to fill with state probability functions
    for i in range(G.number_of_nodes()):
        state_func = sp.parsing.sympy_parser.parse_expr(state_mult_funcs[i]) # convert strings into SymPy data type
        prob_func = state_func/sp.parsing.sympy_parser.parse_expr(norm_func)     # normalize probabilities
        state_prob_funcs.append(lambdify(rate_names, prob_func, "numpy"))    # convert into "lambdified" functions that work with NumPy arrays
    if sym_funcs == True:
        return state_prob_funcs, state_mult_funcs, norm_funcs
    else:
        return state_prob_funcs


def calc_cycle_fluxes(dir_pars):
    return NotImplementedError


def assign_probs_and_analytic_functions_to_G(dir_pars):
    return NotImplementedError
