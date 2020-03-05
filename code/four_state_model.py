# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/15/2020
# 4 State Kinetic Model

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#===============================================================================
#== Functions ==================================================================
#===============================================================================

def generate_node_positions(center=[0, 0], radius=10, N=4):
    """Generates positions for nodes
    """
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
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
                               (2, 3, rates[4]),
                               (3, 2, rates[5]),
                               (3, 0, rates[6]),
                               (0, 3, rates[7])]) # Alternatively, one could make an attribute 'k' and assign these to that

#===============================================================================
#== Graph ======================================================================
#===============================================================================

# k12 = 2
# k21 = 3
# k23 = 5
# k32 = 7
# k34 = 11
# k43 = 13
# k41 = 17
# k14 = 19
#
# rates = [k12, k21, k23, k32, k34, k43, k41, k14]
# G = nx.MultiDiGraph()
# generate_edges(G, rates)
# pos = generate_node_positions()

#===============================================================================
#== Run Method =================================================================
#===============================================================================

# import hill_biochemical_kinetic_diagram_analyzer as kda
# import plot_diagrams as pd
#
# partials = kda.generate_partial_diagrams(G)
# directional_partials = kda.generate_directional_partial_diagrams(partials)
# state_probs = kda.calc_state_probabilities(G, directional_partials)
# print(state_probs)
# print(state_probs.sum(axis=0))
#
# date = '02_20_2020'
# run = '4_state'
# pc = "home"     # 'home', 'laptop' or 'work'
# plot = True
# save = True     # To save, plot must also be True
#
# if pc == "home":
#     path = "C:/Users/Nikolaus/phy495/hill-biochemical-kinetic-diagram-analyzer/data/plots"
# elif pc == "laptop":
# 	path = "C:/Users/nikol/phy495/hill-biochemical-kinetic-diagram-analyzer/data/plots"
# elif pc == "work":
#     path = "/nfs/homes/nawtrey/Documents/PHY495/data"
#
# if plot == True:
#     pd.plot_input_diagram(G, pos, save=save, path=path, date=date, run=run)
#     pd.plot_partials(partials, pos, save=save, path=path, date=date, run=run)
#     pd.plot_directional_partials(directional_partials, pos, save=save, path=path, date=date, run=run)
#     plt.show()
