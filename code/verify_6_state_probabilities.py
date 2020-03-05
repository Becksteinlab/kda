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

def generate_node_positions(center=[0, 0], radius=10, N=6):
    """Generates positions for nodes
    """
    angle = np.pi*np.array([1/2, 5/6, 7/6, 3/2, 11/6, 13/6])        # Angles start at 0 and go clockwise (like unit circle)
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
                               (3, 4, rates[6]),
                               (4, 3, rates[7]),
                               (4, 5, rates[8]),
                               (5, 4, rates[9]),
                               (5, 0, rates[10]),
                               (0, 5, rates[11])]) # Alternatively, one could make an attribute 'k' and assign these to that

#===============================================================================
#== Graph ======================================================================
#===============================================================================

k_on = 1e9                         # units:  /s
k_off = 1e6                    # units:  /s
k_conf = 5e6                   # rate of conformational change
A_conc = 1e-3                       # total [A], in M
B_conc = 1e-7                       # total [B], in M
A_in = A_conc
B_in = 1e-6
A_out = A_conc
B_out = B_conc

a12 = k_on*A_out    # a01
a21 = k_off         # a10
a23 = k_conf        # a12
a32 = k_conf        # a21
a34 = k_off         # a23
a43 = k_on*A_in     # a32
a45 = k_on*B_in     # a34
a54 = k_off         # a43
a56 = k_conf        # a45
a65 = k_conf        # a54
a61 = k_off         # a50
a16 = k_on*B_out    # a05

rates = np.array([a12, a21, a23, a32, a34, a43, a45, a54, a56, a65, a61, a16])
G = nx.MultiDiGraph()
generate_edges(G, rates)
pos = generate_node_positions()

#===============================================================================
#== Run Method =================================================================
#===============================================================================

import hill_biochemical_kinetic_diagram_analyzer as kda
import plot_diagrams as pd

partials = kda.generate_partial_diagrams(G)
directional_partials = kda.generate_directional_partial_diagrams(partials)
p_kda = kda.calc_state_probabilities(G, directional_partials)

#===============================================================================
#== Manual =====================================================================
#===============================================================================

P1 = a65*a54*a43*a32*a21 + a61*a54*a43*a32*a21 + a56*a61*a43*a32*a21 + a45*a56*a61*a32*a21 + a34*a45*a56*a61*a21 + a23*a34*a45*a56*a61
P2 = a12*a65*a54*a43*a32 + a61*a12*a54*a43*a32 + a65*a61*a12*a43*a32 + a45*a56*a61*a32*a12 + a34*a45*a56*a61*a12 + a16*a65*a54*a43*a32
P3 = a12*a23*a65*a54*a43 + a61*a12*a23*a54*a43 + a56*a61*a12*a23*a43 + a45*a56*a61*a12*a23 + a21*a16*a65*a54*a43 + a23*a16*a65*a54*a43
P4 = a65*a54*a12*a23*a34 + a54*a61*a12*a23*a34 + a56*a61*a12*a23*a34 + a32*a21*a16*a65*a54 + a21*a16*a65*a54*a34 + a16*a65*a54*a23*a34
P5 = a65*a12*a23*a34*a45 + a61*a12*a23*a34*a45 + a43*a32*a21*a16*a65 + a45*a32*a21*a16*a65 + a34*a45*a21*a16*a65 + a23*a34*a45*a16*a65
P6 = a12*a23*a34*a45*a56 + a54*a43*a32*a21*a16 + a56*a43*a32*a21*a16 + a45*a56*a32*a21*a16 + a34*a45*a56*a21*a16 + a16*a23*a34*a45*a56

Sigma = P1 + P2 + P3 + P4 + P5 + P6 # Normalization factor
p_manual = np.array([P1, P2, P3, P4, P5, P6])/Sigma

#===============================================================================
#== ODE Solver =================================================================
#===============================================================================

from functions import probability_integrator
k_on = 1e9                         # units:  /s
k_off = 1e6                    # units:  /s
k_conf = 5e6                   # rate of conformational change
A_conc = 1e-3                       # total [A], in M
B_conc = 1e-7                       # total [B], in M
A_in = A_conc
B_in = 1e-6
A_out = A_conc
B_out = B_conc
t_max = 2e-5
max_step = 1e-10
flux = False
# p1, p2, p3, p4, p5, p6
p = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
# [A_in], [A_out], [B_in], [B_out]
CAB = np.array([A_in, A_out, B_in, B_out])
print("Beginning simulation.")
results = probability_integrator([p, CAB], k_off, k_on, k_conf, t_max, max_step=max_step, AB_flux=flux)

p_ode = np.array([results.y[0], results.y[1] , results.y[2], results.y[3], results.y[4], results.y[5]])

#===============================================================================

for i in range(6):
    print("For state {}, probabilities (KDA, Manual, ODE) = ({}, {}, {})".format(i+1, p_kda[i], p_manual[i], p_ode[i][-1]))
