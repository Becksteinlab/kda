# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/19/2020

import numpy as np
import networkx as nx

import hill_biochemical_kinetic_diagram_analyzer as kda

from model_generation import edges_3 as ge3
from model_generation import edges_4 as ge4
from model_generation import edges_4wl as ge4wl
from model_generation import edges_5wl as ge5wl
from model_generation import edges_6 as ge6

#===============================================================================
#== 3 State ====================================================================
#===============================================================================

k12 = 2
k21 = 3
k23 = 5
k32 = 7
k13 = 11
k31 = 13
rates3 = [k12, k21, k23, k32, k13, k31]
rate_names3 = ["x12", "x21", "x23", "x32", "x13", "x31"]
G3 = nx.MultiDiGraph()
ge3(G3, rates3)
pars3 = kda.generate_partial_diagrams(G3)
dir_pars3 = kda.generate_directional_partial_diagrams(pars3)
sp3_probs = kda.calc_state_probabilities(G3, dir_pars3)

state_mults3, norm3 = kda.construct_string_funcs(G3, dir_pars3, rates3, rate_names3)
sympy_funcs3 = kda.construct_sympy_funcs(state_mults3, norm3)
state_prob_funcs3 = kda.construct_lambdify_funcs(sympy_funcs3, rate_names3)

three_state_probs = []
for i in range(G3.number_of_nodes()):
    three_state_probs.append(state_prob_funcs3[i](k12, k21, k23, k32, k13, k31))

#===============================================================================
#== 4 State ====================================================================
#===============================================================================
y12 = 2
y21 = 3
y23 = 5
y32 = 7
y34 = 11
y43 = 13
y41 = 17
y14 = 19
rates4 = [y12, y21, y23, y32, y34, y43, y41, y14]
rate_names4 = ["y12", "y21", "y23", "y32", "y34", "y43", "y41", "y14"]
G4 = nx.MultiDiGraph()
ge4(G4, rates4)
pars4 = kda.generate_partial_diagrams(G4)
dir_pars4 = kda.generate_directional_partial_diagrams(pars4)
sp4_probs = kda.calc_state_probabilities(G4, dir_pars4)


state_mults4, norm4 = kda.construct_string_funcs(G4, dir_pars4, rates4, rate_names4)
sympy_funcs4 = kda.construct_sympy_funcs(state_mults4, norm4)
state_prob_funcs4 = kda.construct_lambdify_funcs(sympy_funcs4, rate_names4)

four_state_probs = []
for i in range(G4.number_of_nodes()):
    four_state_probs.append(state_prob_funcs4[i](y12, y21, y23, y32, y34, y43, y41, y14))

#===============================================================================
#== 4 State w/ Leakage =========================================================
#===============================================================================
y12 = 2
y21 = 3
y23 = 5
y32 = 7
y34 = 11
y43 = 13
y41 = 17
y14 = 19
y24 = 23
y42 = 29
rates4wl = [y12, y21, y23, y32, y34, y43, y41, y14, y24, y42]
rate_names4wl = ["y12", "y21", "y23", "y32", "y34", "y43", "y41", "y14", "y24", "y42"]
G4wl = nx.MultiDiGraph()
ge4wl(G4wl, rates4wl)
pars4wl = kda.generate_partial_diagrams(G4wl)
dir_pars4wl = kda.generate_directional_partial_diagrams(pars4wl)
sp4wl_probs = kda.calc_state_probabilities(G4wl, dir_pars4wl)

state_mults4wl, norm4wl = kda.construct_string_funcs(G4wl, dir_pars4wl, rates4wl, rate_names4wl)
sympy_funcs4wl = kda.construct_sympy_funcs(state_mults4wl, norm4wl)
state_prob_funcs4wl = kda.construct_lambdify_funcs(sympy_funcs4wl, rate_names4wl)

four_state_wl_probs = []
for i in range(G4wl.number_of_nodes()):
    four_state_wl_probs.append(state_prob_funcs4wl[i](y12, y21, y23, y32, y34, y43, y41, y14, y24, y42))

#===============================================================================
#== 5 State w/ Leakage =========================================================
#===============================================================================
z12 = 2
z21 = 3
z23 = 5
z32 = 7
z13 = 11
z31 = 13
z24 = 17
z42 = 19
z35 = 31
z53 = 37
z45 = 23
z54 = 29
rates5wl = [z12, z21, z23, z32, z13, z31, z24, z42, z35, z53, z45, z54]
rate_names5wl = ["z12", "z21", "z23", "z32", "z13", "z31", "z24", "z42", "z35", "z53", "z45", "z54"]
G5wl = nx.MultiDiGraph()
ge5wl(G5wl, rates5wl)
pars5wl = kda.generate_partial_diagrams(G5wl)
dir_pars5wl = kda.generate_directional_partial_diagrams(pars5wl)
sp5wl_probs = kda.calc_state_probabilities(G5wl, dir_pars5wl)

state_mults5wl, norm5wl = kda.construct_string_funcs(G5wl, dir_pars5wl, rates5wl, rate_names5wl)
sympy_funcs5wl = kda.construct_sympy_funcs(state_mults5wl, norm5wl)
state_prob_funcs5wl = kda.construct_lambdify_funcs(sympy_funcs5wl, rate_names5wl)

five_state_wl_probs = []
for i in range(G5wl.number_of_nodes()):
    five_state_wl_probs.append(state_prob_funcs5wl[i](z12, z21, z23, z32, z13, z31, z24, z42, z35, z53, z45, z54))

#===============================================================================
#== 6 State ====================================================================
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
rates6 = [k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16]
rate_names6 = ["x12", "x21", "x23", "x32", "x34", "x43", "x45", "x54", "x56", "x65", "x61", "x16"]
G6 = nx.MultiDiGraph()
ge6(G6, rates6)
pars6 = kda.generate_partial_diagrams(G6)
dir_pars6 = kda.generate_directional_partial_diagrams(pars6)
sp6_probs = kda.calc_state_probabilities(G6, dir_pars6)

state_mults6, norm6 = kda.construct_string_funcs(G6, dir_pars6, rates6, rate_names6)
sympy_funcs6 = kda.construct_sympy_funcs(state_mults6, norm6)
state_prob_funcs6 = kda.construct_lambdify_funcs(sympy_funcs6, rate_names6)

six_state_probs = []
for i in range(G6.number_of_nodes()):
    six_state_probs.append(state_prob_funcs6[i](k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16))

#===============================================================================
#== Verify SymPy Probabilities =================================================
#===============================================================================
print("KDA Probabilities - Lambdify Function Probabilities")
print("3 State:", sp3_probs - three_state_probs)
print("4 State:", sp4_probs - four_state_probs)
print("4wl State:", sp4wl_probs - four_state_wl_probs)
print("5wl State:", sp5wl_probs - five_state_wl_probs)
print("6 State:", sp6_probs - six_state_probs)



#===
