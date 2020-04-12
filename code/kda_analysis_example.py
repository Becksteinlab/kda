# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Example analysis of 6 state model

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from model_generation import pos_6
import kinetic_diagram_analyzer as kda
import plotting

k12 = 2
k21 = 3
k23 = 5
k32 = 7
k34 = 9
k43 = 2
k45 = 8
k54 = 3
k56 = 11
k65 = 2
k61 = 10
k16 = 3

kvals = np.array([[  0, k12,   0,   0,   0, k16],
                  [k21,   0, k23,   0,   0,   0],
                  [  0, k32,   0, k34,   0,   0],
                  [  0,   0, k43,   0, k45,   0],
                  [  0,   0,   0, k54,   0, k56],
                  [k61,   0,   0,   0, k65,   0]])

knames = np.array([[  0, "k12",   0,   0,   0, "k16"],
                   ["k21",   0, "k23",   0,   0,   0],
                   [  0, "k32",   0, "k34",   0,   0],
                   [  0,   0, "k43",   0, "k45",   0],
                   [  0,   0,   0, "k54",   0, "k56"],
                   ["k61",   0,   0,   0, "k65",   0]])

rate_names = ["k12", "k21", "k23", "k32", "k34", "k43", "k45", "k54", "k56", "k65", "k61", "k16"]

G = nx.MultiDiGraph()
kda.generate_edges(G, knames, kvals, name_key='name', val_key='val')
pos = pos_6()

# Generate Diagrams and Functions
partials = kda.generate_partial_diagrams(G)
dir_partials = kda.generate_directional_partial_diagrams(partials)
state_probs = kda.calc_state_probabilities(G, dir_partials, key='val')
mult_funcs, norm_func = kda.calc_state_probabilities(G, dir_partials, key='name', output_strings=True)
sympy_funcs = kda.construct_sympy_funcs(mult_funcs, norm_func)
lambdify_funcs = kda.construct_lambdify_funcs(sympy_funcs, rate_names)

# Add data to G as graph and node attributes
kda.add_graph_attribute(G, kvals, 'k_val_array')
kda.add_graph_attribute(G, knames, 'k_name_array')
kda.add_graph_attribute(G, rate_names, 'k_name_list')
kda.add_graph_attribute(G, partials, 'partial_diagrams')
kda.add_graph_attribute(G, dir_partials, 'directional_partial_diagrams')
kda.add_node_attribute(G, mult_funcs, 'state_mult_func')
kda.add_node_attribute(G, G.number_of_nodes()*[norm_func], 'state_norm_func')
kda.add_node_attribute(G, state_probs, 'probability')
kda.add_node_attribute(G, sympy_funcs, 'sympy_func')
kda.add_node_attribute(G, lambdify_funcs, 'lambdify_func')

# Plot

path = 'C:/Users/nikol/phy495/kinetic-diagram-analyzer/data/plots/plot_dump'

# INSERT FANCY PLOTTING CODE HERE TO TEST IT
# def fancy_plot(G, pos, node_sizes, path):
#     node_sizes = state_probs*scale_factor


# plotting.plot_input_diagram(G, pos)
# fancy_plot(G, pos, state_probs*4e3)
# plotting.plot_partials(partials, pos, panel=True)
# plotting.plot_partials(dir_partials, pos, panel=True, font_size=10, panel_scale=1)
# plt.show()


#==
