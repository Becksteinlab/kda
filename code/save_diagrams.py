# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/30/2020
# Save plots

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from model_generation import edges_6
from model_generation import pos_6
import kinetic_diagram_analyzer as kda
import plotting as pd

#===============================================================================
#== Graph ======================================================================
#===============================================================================

k12 = 2
k21 = 3
k23 = 5
k32 = 7
k34 = 11
k43 = 13
k45 = 17
k54 = 19
k56 = 23
k65 = 29
k61 = 31
k16 = 37

rates = np.array([k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16])
G = nx.MultiDiGraph()
edges_6(G, rates)
pos = pos_6()

#===============================================================================
#== Run Method =================================================================
#===============================================================================

def save_diagrams(G, date, run, path=None):
    partials = kda.generate_partial_diagrams(G)
    directional_partials = kda.generate_directional_partial_diagrams(partials)
    state_probs = kda.calc_state_probabilities(G, directional_partials)
    pd.plot_input_diagram(G, pos, save=True, path=path, date=date, run=run)
    pd.plot_partials(partials, pos, save=True, path=path, date=date, run=run)
    pd.plot_directional_partials(directional_partials, pos, save=True, path=path, date=date, run=run)
    plt.show()

date = '03_30_2020'
run = '6_state'
path = "C:/Users/nikol/phy495/hill-biochemical-kinetic-diagram-analyzer/data/plots"

save_diagrams(G, date, run, path)


#==
