# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Hill Biochemical Kinetic Diagram Analyzer

import numpy as np
import networkx as nx
import functools
import itertools

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

def calc_state_probabilities(G, directional_partials):
    state_multiplicities = np.zeros(G.number_of_nodes())
    for s in range(G.number_of_nodes()):
        partial_multiplicities = np.zeros(len(directional_partials))
        for i in range(len(directional_partials)):
            edge_list = list(directional_partials[i].edges)
            products = np.array([1], dtype=np.float64)
            for e in edge_list:
                products *= G.edges[e[0], e[1], e[2]]['weight']
            partial_multiplicities[i] = products[0]
        N_terms = np.int(len(directional_partials)/G.number_of_nodes())
        for j in range(G.number_of_nodes()):
            state_multiplicities[j] = partial_multiplicities[N_terms*j:N_terms*j+N_terms].sum(axis=0)
    state_probabilities = state_multiplicities/state_multiplicities.sum(axis=0)
    return state_multiplicities, state_probabilities

def assign_probs_to_G(dir_partials):
    return NotImplementedError
