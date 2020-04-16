# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Example analysis of 6 state model

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import kinetic_diagram_analyzer as kda
import model_generation as mg

# ODE Parameters
t_max = 5e0
path = "C:/Users/nikol/phy495/kinetic-diagram-analyzer/data/plots/plot_dump"

#===============================================================================
#== 3 State ====================================================================
#===============================================================================
k12 = 2
k21 = 3
k23 = 5
k32 = 7
k13 = 11
k31 = 13
k3 = np.array([[0, k12, k13],
              [k21, 0, k23],
              [k31, k32, 0]])
k3s = np.array([[0, "k12", "k13"],
                ["k21", 0, "k23"],
                ["k31", "k32", 0]])
rate_names3 = ["k12", "k21", "k23", "k32", "k13", "k31"]
G3 = nx.MultiDiGraph()
pos3 = mg.pos_3()
kda.generate_edges(G3, k3s, k3, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars3 = kda.generate_partial_diagrams(G3)
dir_pars3 = kda.generate_directional_partial_diagrams(pars3)
sp3_KDA = kda.calc_state_probabilities(G3, dir_pars3, key='val')

#== ODE ========================================================================
N3 = 3   # number of states in cycle
p_list = np.random.rand(N3)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p3 = p_list/sigma       # normalize probabilities
results3 = kda.solve_ODE(p3, k3, t_max)
probs3 = results3.y[:N3]
sp3_ODE = []
for i in probs3:
    sp3_ODE.append(i[-1])
sp3_ODE = np.array(sp3_ODE)

#===============================================================================
#== 4 State ====================================================================
#===============================================================================
b12 = 2
b21 = 3
b23 = 5
b32 = 7
b34 = 11
b43 = 13
b41 = 17
b14 = 19
k4 = np.array([[0, b12, 0, b14],
               [b21, 0, b23, 0],
               [0, b32, 0, b34],
               [b41, 0, b43, 0]])
k4s = np.array([[0, "k12", 0, "k14"],
                ["k21", 0, "k23", 0],
                [0, "k32", 0, "k34"],
                ["k41", 0, "k43", 0]])
rate_names4 = ["k12", "k21", "k23", "k32", "k34", "k43", "k41", "k14"]
G4 = nx.MultiDiGraph()
pos4 = mg.pos_4()
kda.generate_edges(G4, k4s, k4, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars4 = kda.generate_partial_diagrams(G4)
dir_pars4 = kda.generate_directional_partial_diagrams(pars4)
sp4_KDA = kda.calc_state_probabilities(G4, dir_pars4, key='val')

#== ODE ========================================================================
N4 = 4   # number of states in cycle
p_list = np.random.rand(N4)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p4 = p_list/sigma       # normalize probabilities
results4 = kda.solve_ODE(p4, k4, t_max)
probs4 = results4.y[:N4]
sp4_ODE = []
for i in probs4:
    sp4_ODE.append(i[-1])
sp4_ODE = np.array(sp4_ODE)

#===============================================================================
#== 4 State w/ Leakage =========================================================
#===============================================================================
b12 = 2
b21 = 3
b23 = 5
b32 = 7
b34 = 11
b43 = 13
b41 = 17
b14 = 19
b24 = 23
b42 = 29
k4wl = np.array([[0, b12, 0, b14],
                [b21, 0, b23, b24],
                [0, b32, 0, b34],
                [b41, b42, b43, 0]])
k4wls = np.array([[0, "k12", 0, "k14"],
                  ["k21", 0, "k23", "k24"],
                  [0, "k32", 0, "k34"],
                  ["k41", "k42", "k43", 0]])
rate_names4wl = ["k12", "k21", "k23", "k32", "k34", "k43", "k41", "k14", "k24", "k42"]
G4wl = nx.MultiDiGraph()
pos4wl = mg.pos_4wl()
kda.generate_edges(G4wl, k4wls, k4wl, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars4wl = kda.generate_partial_diagrams(G4wl)
dir_pars4wl = kda.generate_directional_partial_diagrams(pars4wl)
sp4wl_KDA = kda.calc_state_probabilities(G4wl, dir_pars4wl, key='val')

#== ODE ========================================================================
N4wl = 4   # number of states in cycle
p_list = np.random.rand(N4wl)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p4wl = p_list/sigma       # normalize probabilities
results4wl = kda.solve_ODE(p4wl, k4wl, t_max)
probs4wl = results4wl.y[:N4wl]
sp4wl_ODE = []
for i in probs4wl:
    sp4wl_ODE.append(i[-1])
sp4wl_ODE = np.array(sp4wl_ODE)

#===============================================================================
#== 5 State w/ Leakage =========================================================
#===============================================================================
c12 = 2
c21 = 3
c23 = 5
c32 = 7
c13 = 11
c31 = 13
c24 = 17
c42 = 19
c35 = 31
c53 = 37
c45 = 23
c54 = 29
k5wl = np.array([[  0, c12, c13,   0,   0],
                 [c21,   0, c23, c24,   0],
                 [c31, c32,   0,   0, c35],
                 [  0, c42,   0,   0, c45],
                 [  0,   0, c53, c54,   0]])
k5wls = np.array([[  0, "k12", "k13",   0,   0],
                  ["k21",   0, "k23", "k24",   0],
                  ["k31", "k32",   0,   0, "k35"],
                  [  0, "k42",   0,   0, "k45"],
                  [  0,   0, "k53", "k54",   0]])
rate_names5wl = ["k12", "k21", "k23", "k32", "k13", "k31", "k24", "k42", "k35", "k53", "k45", "k54"]
G5wl = nx.MultiDiGraph()
pos5wl = mg.pos_5wl()
kda.generate_edges(G5wl, k5wls, k5wl, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars5wl = kda.generate_partial_diagrams(G5wl)
dir_pars5wl = kda.generate_directional_partial_diagrams(pars5wl)
sp5wl_KDA = kda.calc_state_probabilities(G5wl, dir_pars5wl, key='val')

#== ODE ========================================================================
N5wl = 5   # number of states in cycle
p_list = np.random.rand(N5wl)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p5wl = p_list/sigma       # normalize probabilities
results5wl = kda.solve_ODE(p5wl, k5wl, t_max)
probs5wl = results5wl.y[:N5wl]
sp5wl_ODE = []
for i in probs5wl:
    sp5wl_ODE.append(i[-1])
sp5wl_ODE = np.array(sp5wl_ODE)

#===============================================================================
#== 6 State ====================================================================
#===============================================================================
a12 = 2
a21 = 3
a23 = 5
a32 = 7
a34 = 11
a43 = 13
a45 = 17
a54 = 19
a56 = 23
a65 = 29
a61 = 31
a16 = 37
k6 = np.array([[  0, a12,   0,   0,   0, a16],
               [a21,   0, a23,   0,   0,   0],
               [  0, a32,   0, a34,   0,   0],
               [  0,   0, a43,   0, a45,   0],
               [  0,   0,   0, a54,   0, a56],
               [a61,   0,   0,   0, a65,   0]])
k6s = np.array([[  0, "k12",   0,   0,   0, "k16"],
                ["k21",   0, "k23",   0,   0,   0],
                [  0, "k32",   0, "k34",   0,   0],
                [  0,   0, "k43",   0, "k45",   0],
                [  0,   0,   0, "k54",   0, "k56"],
                ["k61",   0,   0,   0, "k65",   0]])
rate_names6 = ["k12", "k21", "k23", "k32", "k34", "k43", "k45", "k54", "k56", "k65", "k61", "k16"]
G6 = nx.MultiDiGraph()
pos6 = mg.pos_6()
kda.generate_edges(G6, k6s, k6, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars6 = kda.generate_partial_diagrams(G6)
dir_pars6 = kda.generate_directional_partial_diagrams(pars6)
sp6_KDA = kda.calc_state_probabilities(G6, dir_pars6, key='val')

#== ODE ========================================================================
N6 = 6   # number of states in cycle
p_list = np.random.rand(N6)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p6 = p_list/sigma       # normalize probabilities
results6 = kda.solve_ODE(p6, k6, t_max)
probs6 = results6.y[:N6]
sp6_ODE = []
for i in probs6:
    sp6_ODE.append(i[-1])
sp6_ODE = np.array(sp6_ODE)

#===============================================================================
#== Plot =======================================================================
#===============================================================================
import plotting

# Save input diagrams
plotting.plot_input_diagram(G3, pos3, path=path, label='3_state')
plotting.plot_input_diagram(G4, pos4, path=path, label='4_state')
plotting.plot_input_diagram(G4wl, pos4wl, path=path, label='4wl_state')
plotting.plot_input_diagram(G5wl, pos5wl, path=path, label='5wl_state')
plotting.plot_input_diagram(G6, pos6, path=path, label='6_state')

# Save partial panels
plotting.plot_partials(pars3, pos3, panel=True, panel_scale=2, font_size=10, path=path, label='3_state_partial')
plotting.plot_partials(pars4, pos4, panel=True, panel_scale=2, font_size=10, path=path, label='4_state_partial')
plotting.plot_partials(pars4wl, pos4wl, panel=True, panel_scale=2, font_size=10, path=path, label='4wl_state_partial')
plotting.plot_partials(pars5wl, pos5wl, panel=True, panel_scale=2, font_size=10, path=path, label='5wl_state_partial')
plotting.plot_partials(pars6, pos6, panel=True, panel_scale=2, font_size=10, path=path, label='6_state_partial')

# Save directional partial panels
plotting.plot_partials(dir_pars3, pos3, panel=True, panel_scale=2, font_size=10, cbt=True, path=path, label='3_state_dir_partial')
plotting.plot_partials(dir_pars4, pos4, panel=True, panel_scale=2, font_size=10, cbt=True, path=path, label='4_state_dir_partial')
plotting.plot_partials(dir_pars4wl, pos4wl, panel=True, panel_scale=2, font_size=10, cbt=True, path=path, label='4wl_state_dir_partial')
plotting.plot_partials(dir_pars5wl, pos5wl, panel=True, panel_scale=2, font_size=10, cbt=True, path=path, label='5wl_state_dir_partial')
plotting.plot_partials(dir_pars6, pos6, panel=True, panel_scale=2, font_size=10, cbt=True, path=path, label='6_state_dir_partial')

# Save ODE probability plots
plotting.plot_ODE_probs(results3, path=path, label='3_state')
plotting.plot_ODE_probs(results4, path=path, label='4_state')
plotting.plot_ODE_probs(results4wl, path=path, label='4wl_state')
plotting.plot_ODE_probs(results5wl, path=path, label='5wl_state')
plotting.plot_ODE_probs(results6, path=path, label='6_state')

#===
