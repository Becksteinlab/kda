# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/13/2020
# 6 State Kinetic Model

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#===============================================================================
#== Functions ==================================================================
#===============================================================================

def pos_6(center=[0, 0], radius=10):
    """
    Generates positions of nodes for a hexagon in the xy-plane

    Parameters
    ----------
    center : list
        Defines the center of the hexagon in the xy-plane
    radius : int
        The radius of the hexagon, from center to node
    """
    N = 6                                                       # number of states/nodes
    angle = np.pi*np.array([1/2, 5/6, 7/6, 3/2, 11/6, 13/6])    # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                    # Empty 2D array of shape (6x2)
    for i in range(N):                                          # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}                                                    # empty dict for positions to go in
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos

def edges_6(G, rates):
    """
    Generates edges 6 state model
    """
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (2, 3, rates[4]),
                               (3, 2, rates[5]),
                               (3, 4, rates[6]),
                               (4, 3, rates[7]),
                               (4, 5, rates[8]),
                               (5, 4, rates[9]),
                               (5, 0, rates[10]),
                               (0, 5, rates[11])]) # Alternatively, one could make an attribute 'k' and assign these to that

#===============================================================================
#== Graph ======================================================================
#===============================================================================

# a12 = 2
# a21 = 3
# a23 = 5
# a32 = 7
# a34 = 11
# a43 = 13
# a45 = 17
# a54 = 19
# a56 = 23
# a65 = 29
# a61 = 31
# a16 = 37

# rates = np.array([a12, a21, a23, a32, a34, a43, a45, a54, a56, a65, a61, a16])
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
# run = '6_state'
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
