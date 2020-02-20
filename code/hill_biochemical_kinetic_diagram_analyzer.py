# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/14/2020
# Hill Biochemical Kinetic Diagram Analyzer

import numpy as np
import networkx as nx
import functools

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

def generate_directional_connections(target, cons):
    edges = [i for i in cons if target in i]
    neighbors = [[j for j in i if not j == target][0] for i in edges]
    if not neighbors:
        return {}
    # if not any(map(lambda x: target in x, cons)):
    #     raise ValueError("Could not find {0} in connections".format(target))
    cons = [k for k in cons if not k in edges]
    return functools.reduce(combine, [{target: neighbors}] + [generate_directional_connections(i, cons) for i in neighbors])

def generate_directional_edges(cons):
    values = []
    for i in cons.keys():
        for j in cons[i]:
            values.append((j, i, 0))
    return values

def generate_partial_diagrams(G):
    unique_edges = find_unique_edges(G)         # Get list of unique edges
    N_partial_edges = G.number_of_nodes() - 1   # Number of edges in each partial diagram
    diagrams = []
    for i in range(len(unique_edges)):
        diag = G.copy()
        diag.remove_edge(unique_edges[i][0], unique_edges[i][1], 0)
        diag.remove_edge(unique_edges[i][1], unique_edges[i][0], 0)
        diagrams.append(diag)
    return diagrams

# What happens when you start by generating all the diagrams that one would get
# if they simply removed one of every edge?

def generate_directional_partial_diagrams(partials):
    N_targets = partials[0].number_of_nodes()
    dir_par_diags = []
    for target in range(N_targets):
        for i in range(N_targets):
            diag = partials[i].copy()               # Make a copy of directional partial diagram
            unique_edges = find_unique_edges(diag)      # Find unique edges of that diagram
            cons = generate_directional_connections(target, unique_edges)   # Get dictionary of connections
            dir_edges = generate_directional_edges(cons)                    # Get directional edges from connections
            edges = list(diag.edges())
            diag.remove_edges_from(edges)
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
        for j in range(N_terms):
            state_multiplicities[j] = partial_multiplicities[N_terms*j:N_terms*j+N_terms].sum(axis=0)
    state_probabilities = state_multiplicities/state_multiplicities.sum(axis=0)
    return state_probabilities

def assign_probs_to_G(dir_partials):
    return NotImplementedError
#===============================================================================

#===============================================================================
#== Run Method =================================================================
#===============================================================================

# NEED TO GET LABELS AND POSTIIONS FROM G
#
# partials = generate_partial_diagrams(G)
# directional_partials = generate_directional_partial_diagrams(partials)
# state_probs = calc_state_probabilities(G, directional_partials)
# print(state_probs)
# print(state_probs.sum(axis=0))
#
# date = '02_13_2020'
# run = 'test'
# model = 'NHE'
# pc = "laptop"     # 'home', 'laptop' or 'work'
# plot = False
# save = False
#
# if pc == "home":
#     path = "/c/Users/Nikolaus/phy495/antiporter-model/data"
# elif pc == "laptop":
# 	path = "C:/Users/nikol/phy495/antiporter-model/data"
# elif pc == "work":
#     path = "/nfs/homes/nawtrey/Documents/PHY495/data"
#
# if plot == True:
#     plot_G(G, pos)
#     plot_partials(partials, pos)
#     plot_dir_partials(directional_partials, pos)
# plt.show()
