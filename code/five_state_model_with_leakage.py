# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/05/2020
# 5 State Kinetic Model With Leakage

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

def generate_node_positions(center=[0, 0], radius=10, N=5):
    """Generates positions for nodes
    """
    h = radius*np.sqrt(3)/2 # height of equilateral triangle
    pos = {0 : [0, h],
           1 : [-radius/2, 0],
           2 : [radius/2, 0],
           3 : [-radius/2, -radius],
           4 : [radius/2, -radius]}
    return pos

def generate_edges(G, rates):
    G.add_weighted_edges_from([(0, 1, rates[0]),
                               (1, 0, rates[1]),
                               (1, 2, rates[2]),
                               (2, 1, rates[3]),
                               (0, 2, rates[4]),
                               (2, 0, rates[5]),
                               (1, 3, rates[6]),
                               (3, 1, rates[7]),
                               (2, 4, rates[8]),
                               (4, 2, rates[9]),
                               (3, 4, rates[10]),
                               (4, 3, rates[11])]) # Alternatively, one could make an attribute 'k' and assign these to that

#===============================================================================
#== Graph ======================================================================
#===============================================================================

k12 = 2     # 01
k21 = 3     # 10
k23 = 5     # 12
k32 = 7     # 21
k13 = 11    # 02
k31 = 13    # 20
k24 = 17    # 13
k42 = 19    # 31
k35 = 31    # 24
k53 = 37    # 42
k45 = 23    # 34
k54 = 29    # 43

rates = [k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54]

G = nx.MultiDiGraph()
generate_edges(G, rates)
pos = generate_node_positions()

#===============================================================================
#== Run Method =================================================================
#===============================================================================
# 
# import hill_biochemical_kinetic_diagram_analyzer as kda
# import plot_diagrams as pd
#
# partials = kda.generate_partial_diagrams(G)
# directional_partials = kda.generate_directional_partial_diagrams(partials)
# state_mult, state_probs = kda.calc_state_probabilities(G, directional_partials)
# print(state_probs)
# print(state_probs.sum(axis=0))
#
# date = '02_20_2020'
# run = '4_state_with_leakage'
# pc = "home"     # 'home', 'laptop' or 'work'
# plot = True
# save = False     # To save, plot must also be True
#
# if pc == "home":
#     path = "C:/Users/Nikolaus/phy495/hill-biochemical-kinetic-diagram-analyzer/data/plots"
# elif pc == "laptop":
# 	path = "C:/Users/nikol/phy495/hill-biochemical-kinetic-diagram-analyzer/data/plots"
#
# if plot == True:
#     pd.plot_input_diagram(G, pos, save=save, path=path, date=date, run=run)
#     # pd.plot_partials(partials, pos, save=save, path=path, date=date, run=run)
#     # pd.plot_directional_partials(directional_partials, pos, save=save, path=path, date=date, run=run)
#     plt.show()
