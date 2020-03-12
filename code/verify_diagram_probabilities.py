# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/13/2020
# 6 State Kinetic Model

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy

import hill_biochemical_kinetic_diagram_analyzer as kda

from three_state_model import generate_edges as ge3
from four_state_model import generate_edges as ge4
from four_state_model_with_leakage import generate_edges as ge4wl
from five_state_model_with_leakage import generate_edges as ge5wl
from six_state_model import generate_edges as ge6

#===============================================================================
#== 3 State ====================================================================
#===============================================================================
k12 = 2
k21 = 3
k23 = 5
k32 = 7
k13 = 11
k31 = 13
G3 = nx.MultiDiGraph()
ge3(G3, [k12, k21, k23, k32, k13, k31])

#== State probabilitites =======================================================
partials3 = kda.generate_partial_diagrams(G3)
directional_partials3 = kda.generate_directional_partial_diagrams(partials3)
sp3_mult, sp3_diag = kda.calc_state_probabilities(G3, directional_partials3)
#== Manual =====================================================================
P1 = k23*k31 + k32*k21 + k31*k21
P2 = k13*k32 + k32*k12 + k31*k12
P3 = k13*k23 + k12*k23 + k21*k13
Sigma = P1 + P2 + P3 # Normalization factor
sp3_manual = np.array([P1, P2, P3])/Sigma


# from sympy import *
# k12, k21, k23, k32, k13, k31 = symbols('k12 k21 k23 k32 k13 k31')
# init_printing(use_unicode=True)
# simplify((k23*k31 + k32*k21 + k31*k21)/(k23*k31 + k32*k21 + k31*k21 + k13*k32 + k32*k12 + k31*k12 + k13*k23 + k12*k23 + k21*k13))
# print(latex((k23*k31 + k32*k21 + k31*k21)/(k23*k31 + k32*k21 + k31*k21 + k13*k32 + k32*k12 + k31*k12 + k13*k23 + k12*k23 + k21*k13)))

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
G4 = nx.MultiDiGraph()
ge4(G4, [b12, b21, b23, b32, b34, b43, b41, b14])

#== State probabilitites =======================================================
partials4 = kda.generate_partial_diagrams(G4)
directional_partials4 = kda.generate_directional_partial_diagrams(partials4)
sp4_mult, sp4_diag = kda.calc_state_probabilities(G4, directional_partials4)
#== Manual =====================================================================
P1 = b43*b32*b21 + b23*b34*b41 + b21*b34*b41 + b41*b32*b21
P2 = b12*b43*b32 + b14*b43*b32 + b34*b41*b12 + b32*b41*b12
P3 = b43*b12*b23 + b23*b14*b43 + b21*b14*b43 + b41*b12*b23
P4 = b12*b23*b34 + b14*b23*b34 + b34*b21*b14 + b32*b21*b14
Sigma = P1 + P2 + P3 + P4 # Normalization factor
sp4_manual = np.array([P1, P2, P3, P4])/Sigma

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
G4wl = nx.MultiDiGraph()
ge4wl(G4wl, [b12, b21, b23, b32, b34, b43, b41, b14, b24, b42])

#== State probabilitites =======================================================
partials4wl = kda.generate_partial_diagrams(G4wl)
directional_partials4wl = kda.generate_directional_partial_diagrams(partials4wl)
sp4wl_mult, sp4wl_diag = kda.calc_state_probabilities(G4wl, directional_partials4wl)
#== Manual =====================================================================
P1 = b43*b32*b21 + b23*b34*b41 + b21*b34*b41 + b41*b32*b21 + b32*b42*b21 + b24*b34*b41 + b34*b42*b21 + b32*b24*b41
P2 = b12*b43*b32 + b14*b43*b32 + b34*b41*b12 + b32*b41*b12 + b32*b42*b12 + b34*b14*b42 + b12*b34*b42 + b32*b14*b42
P3 = b43*b12*b23 + b23*b14*b43 + b21*b14*b43 + b41*b12*b23 + b12*b42*b23 + b14*b24*b43 + b12*b24*b43 + b14*b42*b23
P4 = b12*b23*b34 + b14*b23*b34 + b34*b21*b14 + b32*b21*b14 + b32*b12*b24 + b14*b24*b34 + b34*b12*b24 + b14*b32*b24
Sigma = P1 + P2 + P3 + P4 # Normalization factor
sp4wl_manual = np.array([P1, P2, P3, P4])/Sigma

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
G5wl = nx.MultiDiGraph()
ge5wl(G5wl, [c12, c21, c23, c32, c13, c31, c24, c42, c35, c53, c45, c54])

#== State probabilitites =======================================================
partials5wl = kda.generate_partial_diagrams(G5wl)
directional_partials5wl = kda.generate_directional_partial_diagrams(partials5wl)
sp5wl_mult, sp5wl_diag = kda.calc_state_probabilities(G5wl, directional_partials5wl)
#== Manual =====================================================================
P1 = c35*c54*c42*c21 + c24*c45*c53*c31 + c21*c45*c53*c31 + c42*c21*c53*c31 + c54*c42*c21*c31 + c54*c42*c32*c21 + c45*c53*c23*c31 + c53*c32*c42*c21 + c42*c23*c53*c31 + c54*c42*c23*c31 + c45*c53*c32*c21
P2 = c12*c35*c54*c42 + c13*c35*c54*c42 + c45*c53*c31*c12 + c42*c53*c31*c12 + c31*c12*c54*c42 + c54*c42*c32*c12 + c45*c53*c13*c32 + c53*c32*c42*c12 + c53*c13*c32*c42 + c54*c42*c13*c32 + c45*c53*c32*c12
P3 = c12*c24*c45*c53 + c13*c24*c45*c53 + c21*c13*c45*c53 + c42*c21*c13*c53 + c54*c42*c21*c13 + c54*c42*c12*c23 + c45*c53*c23*c13 + c42*c12*c23*c53 + c42*c23*c13*c53 + c54*c42*c23*c13 + c45*c53*c12*c23
P4 = c12*c24*c35*c54 + c24*c13*c35*c54 + c21*c13*c35*c54 + c53*c31*c12*c24 + c54*c31*c12*c24 + c12*c32*c24*c54 + c13*c23*c35*c54 + c53*c32*c12*c24 + c13*c53*c32*c24 + c13*c32*c24*c54 + c12*c23*c35*c54
P5 = c35*c12*c24*c45 + c13*c35*c24*c45 + c45*c21*c13*c35 + c42*c21*c13*c35 + c31*c12*c24*c45 + c12*c32*c24*c45 + c13*c23*c35*c45 + c12*c42*c23*c35 + c42*c23*c13*c35 + c13*c32*c24*c45 + c12*c23*c35*c45
Sigma = P1 + P2 + P3 + P4 + P5 # Normalization factor
sp5wl_manual = np.array([P1, P2, P3, P4, P5])/Sigma

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
G6 = nx.MultiDiGraph()
ge6(G6, [a12, a21, a23, a32, a34, a43, a45, a54, a56, a65, a61, a16])

#== State probabilitites =======================================================
partials6 = kda.generate_partial_diagrams(G6)
directional_partials6 = kda.generate_directional_partial_diagrams(partials6)
sp6_mult, sp6_diag = kda.calc_state_probabilities(G6, directional_partials6)

#== Manual =====================================================================
P1 = a65*a54*a43*a32*a21 + a61*a54*a43*a32*a21 + a56*a61*a43*a32*a21 + a45*a56*a61*a32*a21 + a34*a45*a56*a61*a21 + a23*a34*a45*a56*a61
P2 = a12*a65*a54*a43*a32 + a61*a12*a54*a43*a32 + a56*a61*a12*a43*a32 + a45*a56*a61*a32*a12 + a34*a45*a56*a61*a12 + a16*a65*a54*a43*a32
P3 = a12*a23*a65*a54*a43 + a61*a12*a23*a54*a43 + a56*a61*a12*a23*a43 + a45*a56*a61*a12*a23 + a21*a16*a65*a54*a43 + a23*a16*a65*a54*a43
P4 = a65*a54*a12*a23*a34 + a54*a61*a12*a23*a34 + a56*a61*a12*a23*a34 + a32*a21*a16*a65*a54 + a21*a16*a65*a54*a34 + a16*a65*a54*a23*a34
P5 = a65*a12*a23*a34*a45 + a61*a12*a23*a34*a45 + a43*a32*a21*a16*a65 + a45*a32*a21*a16*a65 + a34*a45*a21*a16*a65 + a23*a34*a45*a16*a65
P6 = a12*a23*a34*a45*a56 + a54*a43*a32*a21*a16 + a56*a43*a32*a21*a16 + a45*a56*a32*a21*a16 + a34*a45*a56*a21*a16 + a16*a23*a34*a45*a56
Sigma = P1 + P2 + P3 + P4 + P5 + P6 # Normalization factor
sp6_manual = np.array([P1, P2, P3, P4, P5, P6])/Sigma

#===============================================================================
#== Verification ===============================================================
#===============================================================================
print("3 state model probabilities, manual - diag: {}".format(sp3_manual - sp3_diag))
print("4 state model probabilities, manual - diag: {}".format(sp4_manual - sp4_diag))
print("4 state model with leakage probabilities, manual - diag: {}".format(sp4wl_manual - sp4wl_diag))
print("5 state model with leakage probabilities, manual - diag: {}".format(sp5wl_manual - sp5wl_diag))
print("6 state model probabilities, manual - diag: {}".format(sp6_manual - sp6_diag))

# Check multiplicities
# Check probabilitites
# Check that probabilities are normalized to 1

#===============================================================================
#== ODE Solver =================================================================
#===============================================================================

# from functions import probability_integrator
# k_on = 1e9                         # units:  /s
# k_off = 1e6                    # units:  /s
# k_conf = 5e6                   # rate of conformational change
# A_conc = 1e-3                       # total [A], in M
# B_conc = 1e-7                       # total [B], in M
# A_in = A_conc
# B_in = 1e-6
# A_out = A_conc
# B_out = B_conc
# t_max = 2e-5
# max_step = 1e-10
# flux = False
# # p1, p2, p3, p4, p5, p6
# p = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
# # [A_in], [A_out], [B_in], [B_out]
# CAB = np.array([A_in, A_out, B_in, B_out])
# print("Beginning simulation.")
# results = probability_integrator([p, CAB], k_off, k_on, k_conf, t_max, max_step=max_step, AB_flux=flux)
#
# p_ode = np.array([results.y[0], results.y[1] , results.y[2], results.y[3], results.y[4], results.y[5]])
#
# #===============================================================================
#
# for i in range(6):
#     print("For state {}, probabilities (KDA, Manual, ODE) = ({}, {}, {})".format(i+1, p_kda[i], p_manual[i], p_ode[i][-1]))
