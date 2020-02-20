# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/13/2020
# Kinetic Model Analysis Testing

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import functools

def generate_nodes(center=[0, 0], radius=10, N=3):
    """Generates positions for nodes of a hexagon
    """
    angle = np.pi*np.array([1/2, 7/6, 11/6])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((3, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(3):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def generate_edges(G, rates):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (0, 2, rates[4]),
                               (2, 0, rates[5])]) # Alternatively, one could make an attribute 'k' and assign these to that

def plot_G(G, pos):
    fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
    fig1.add_subplot(111)
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=[0, 1, 2], node_color='grey')
    nx.draw_networkx_edges(G, pos, width=4, arrow_style='->', arrowsize=15)
    labels = {}
    for i in range(3):
        labels[i] = r"${}$".format(i+1)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

def plot_partials(partials, pos):
    for i in range(len(partials)):
        fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
        fig1.add_subplot(111)
        partial = partials[i]
        nx.draw_networkx_nodes(partial, pos, node_size=500, nodelist=[0, 1, 2], node_color='grey')
        nx.draw_networkx_edges(partial, pos, width=4, arrow_style='->', arrowsize=15)
        labels = {}
        for i in range(3):
            labels[i] = r"${}$".format(i+1)
        nx.draw_networkx_labels(partial, pos, labels, font_size=16)

def plot_dir_partials(dir_partials, pos):
    for i in range(len(dir_partials)):
        fig1 = plt.figure(figsize=(4, 3), tight_layout=True)
        fig1.add_subplot(111)
        partial = dir_partials[i]
        nx.draw_networkx_nodes(partial, pos, node_size=500, nodelist=[0, 1, 2], node_color='grey')
        nx.draw_networkx_edges(partial, pos, width=4, arrow_style='->', arrowsize=15)
        labels = {}
        for j in range(3):
            labels[j] = r"${}$".format(j+1)
        nx.draw_networkx_labels(partial, pos, labels, font_size=16)
        if save == True:
            fig1.savefig(path + "/{}/{}/directional_diagram_graph_{}_{}.png".format(model, date, i, run))
#===============================================================================
#== Graph ======================================================================
#===============================================================================

k12 = 2
k21 = 3
k23 = 5
k32 = 7
k13 = 11
k31 = 13

rates = [k12, k21, k23, k32, k13, k31]

G = nx.MultiDiGraph()
generate_edges(G, rates)
pos = generate_nodes()

weights = []
for (u, v, wt) in G.edges.data('weight'):
    weights.append(wt)

n_edges = G.number_of_edges()
n_nodes = G.number_of_nodes()

k12 == G.edges[0, 1, 0]['weight']
k21 == G.edges[1, 0, 0]['weight']
k23 == G.edges[1, 2, 0]['weight']
k32 == G.edges[2, 1, 0]['weight']
k13 == G.edges[0, 2, 0]['weight']
k31 == G.edges[2, 0, 0]['weight']

#===============================================================================
#= Generate Partial Diagrams ===================================================
#===============================================================================

def find_unique_edges(diagram):
    edges = list(diagram.edges)           # Get list of edges
    sorted_edges = np.sort(edges)      # Sort list of edges
    tuples = [(sorted_edges[i, 1], sorted_edges[i, 2]) for i in range(len(sorted_edges))]   # Make list of edges tuples
    return list(set(tuples))

def generate_partial_diagrams(G):
    unique_edges = find_unique_edges(G)     # Get list of unique edges
    N_partials = len(unique_edges) - 1         # Number of edges for each partial diagram
    diagrams = []
    for i in range(len(unique_edges)):
        diag = G.copy()
        diag.remove_edge(unique_edges[i][0], unique_edges[i][1], 0)
        diag.remove_edge(unique_edges[i][1], unique_edges[i][0], 0)
        diagrams.append(diag)
    return diagrams

#===============================================================================
#= Generate Directional Partial Diagrams =======================================
#===============================================================================

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

def generate_directional_edges(connections):
    values = []
    for i in connections.keys():
        for j in connections[i]:
            values.append((j, i, 0))
    return values

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
            products = np.array([1])
            for e in edge_list:
                products *= G.edges[e[0], e[1], e[2]]['weight']
            partial_multiplicities[i] = products[0]
        N_terms = np.int(len(directional_partials)/G.number_of_nodes())
        for j in range(N_terms):
            state_multiplicities[j] = partial_multiplicities[N_terms*j:N_terms*j+N_terms].sum(axis=0)
    state_probabilities = state_multiplicities/state_multiplicities.sum(axis=0)
    return state_probabilities

#===============================================================================
#= Run Method ==================================================================
#===============================================================================

partials = generate_partial_diagrams(G)
directional_partials = generate_directional_partial_diagrams(partials)
state_probs = calc_state_probabilities(G, directional_partials)
print(state_probs)
print(state_probs.sum(axis=0))

date = '02_13_2020'
run = 'test'
model = 'NHE'
pc = "laptop"     # 'home', 'laptop' or 'work'
plot = False
save = False

if pc == "home":
    path = "/c/Users/Nikolaus/phy495/antiporter-model/data"
elif pc == "laptop":
	path = "C:/Users/nikol/phy495/antiporter-model/data"
elif pc == "work":
    path = "/nfs/homes/nawtrey/Documents/PHY495/data"

if plot == True:
    plot_G(G, pos)
    plot_partials(partials, pos)
    plot_dir_partials(directional_partials, pos)
plt.show()

#===============================================================================
#===============================================================================
#===============================================================================
