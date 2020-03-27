# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 02/13/2020

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import hill_biochemical_kinetic_diagram_analyzer as kda

from three_state_model import generate_edges as ge3
from four_state_model import generate_edges as ge4
from four_state_model_with_leakage import generate_edges as ge4wl
from five_state_model_with_leakage import generate_edges as ge5wl
from six_state_model import generate_edges as ge6
from ODE_integrator import integrate_prob_ODE as integrate
from ODE_integrator import plot_ODE_probs

t_max = 5e0
max_step = t_max/1e3
plot = False

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
sp3_diag, sp3_mult = kda.calc_state_probabilities(G3, directional_partials3, state_mults=True)
#== Manual =====================================================================
P1 = k23*k31 + k32*k21 + k31*k21
P2 = k13*k32 + k32*k12 + k31*k12
P3 = k13*k23 + k12*k23 + k21*k13
Sigma = P1 + P2 + P3 # Normalization factor
sp3_manual = np.array([P1, P2, P3])/Sigma
#== ODE ========================================================================
N3 = 3   # number of states in cycle
p_list = np.random.rand(N3)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p3 = p_list/sigma       # normalize probabilities
k3 = np.array([[0, k12, k13],
              [k21, 0, k23],
              [k31, k32, 0]])
results3 = integrate(p3, k3, t_max, max_step)
probs3 = results3.y[:N3]
sp3_ODE = []
for i in probs3:
    sp3_ODE.append(i[-1])

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
sp4_diag, sp4_mult = kda.calc_state_probabilities(G4, directional_partials4, state_mults=True)
#== Manual =====================================================================
P1 = b43*b32*b21 + b23*b34*b41 + b21*b34*b41 + b41*b32*b21
P2 = b12*b43*b32 + b14*b43*b32 + b34*b41*b12 + b32*b41*b12
P3 = b43*b12*b23 + b23*b14*b43 + b21*b14*b43 + b41*b12*b23
P4 = b12*b23*b34 + b14*b23*b34 + b34*b21*b14 + b32*b21*b14
Sigma = P1 + P2 + P3 + P4 # Normalization factor
sp4_manual = np.array([P1, P2, P3, P4])/Sigma
#== ODE ========================================================================
N4 = 4   # number of states in cycle
p_list = np.random.rand(N4)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p4 = p_list/sigma       # normalize probabilities
k4 = np.array([[0, b12, 0, b14],
               [b21, 0, b23, 0],
               [0, b32, 0, b34],
               [b41, 0, b43, 0]])
results4 = integrate(p4, k4, t_max, max_step)
probs4 = results4.y[:N4]
sp4_ODE = []
for i in probs4:
    sp4_ODE.append(i[-1])

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
sp4wl_diag, sp4wl_mult = kda.calc_state_probabilities(G4wl, directional_partials4wl, state_mults=True)
#== Manual =====================================================================
P1 = b43*b32*b21 + b23*b34*b41 + b21*b34*b41 + b41*b32*b21 + b32*b42*b21 + b24*b34*b41 + b34*b42*b21 + b32*b24*b41
P2 = b12*b43*b32 + b14*b43*b32 + b34*b41*b12 + b32*b41*b12 + b32*b42*b12 + b34*b14*b42 + b12*b34*b42 + b32*b14*b42
P3 = b43*b12*b23 + b23*b14*b43 + b21*b14*b43 + b41*b12*b23 + b12*b42*b23 + b14*b24*b43 + b12*b24*b43 + b14*b42*b23
P4 = b12*b23*b34 + b14*b23*b34 + b34*b21*b14 + b32*b21*b14 + b32*b12*b24 + b14*b24*b34 + b34*b12*b24 + b14*b32*b24
Sigma = P1 + P2 + P3 + P4 # Normalization factor
sp4wl_manual = np.array([P1, P2, P3, P4])/Sigma
#== ODE ========================================================================
N4wl = 4   # number of states in cycle
p_list = np.random.rand(N4wl)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p4wl = p_list/sigma       # normalize probabilities
k4wl = np.array([[0, b12, 0, b14],
                [b21, 0, b23, b24],
                [0, b32, 0, b34],
                [b41, b42, b43, 0]])
results4wl = integrate(p4wl, k4wl, t_max, max_step)
probs4wl = results4wl.y[:N4wl]
sp4wl_ODE = []
for i in probs4wl:
    sp4wl_ODE.append(i[-1])

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
sp5wl_diag, sp5wl_mult = kda.calc_state_probabilities(G5wl, directional_partials5wl, state_mults=True)
#== Manual =====================================================================
P1 = c35*c54*c42*c21 + c24*c45*c53*c31 + c21*c45*c53*c31 + c42*c21*c53*c31 + c54*c42*c21*c31 + c54*c42*c32*c21 + c45*c53*c23*c31 + c53*c32*c42*c21 + c42*c23*c53*c31 + c54*c42*c23*c31 + c45*c53*c32*c21
P2 = c12*c35*c54*c42 + c13*c35*c54*c42 + c45*c53*c31*c12 + c42*c53*c31*c12 + c31*c12*c54*c42 + c54*c42*c32*c12 + c45*c53*c13*c32 + c53*c32*c42*c12 + c53*c13*c32*c42 + c54*c42*c13*c32 + c45*c53*c32*c12
P3 = c12*c24*c45*c53 + c13*c24*c45*c53 + c21*c13*c45*c53 + c42*c21*c13*c53 + c54*c42*c21*c13 + c54*c42*c12*c23 + c45*c53*c23*c13 + c42*c12*c23*c53 + c42*c23*c13*c53 + c54*c42*c23*c13 + c45*c53*c12*c23
P4 = c12*c24*c35*c54 + c24*c13*c35*c54 + c21*c13*c35*c54 + c53*c31*c12*c24 + c54*c31*c12*c24 + c12*c32*c24*c54 + c13*c23*c35*c54 + c53*c32*c12*c24 + c13*c53*c32*c24 + c13*c32*c24*c54 + c12*c23*c35*c54
P5 = c35*c12*c24*c45 + c13*c35*c24*c45 + c45*c21*c13*c35 + c42*c21*c13*c35 + c31*c12*c24*c45 + c12*c32*c24*c45 + c13*c23*c35*c45 + c12*c42*c23*c35 + c42*c23*c13*c35 + c13*c32*c24*c45 + c12*c23*c35*c45
Sigma = P1 + P2 + P3 + P4 + P5 # Normalization factor
sp5wl_manual = np.array([P1, P2, P3, P4, P5])/Sigma
#== ODE ========================================================================
N5wl = 5   # number of states in cycle
p_list = np.random.rand(N5wl)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p5wl = p_list/sigma       # normalize probabilities
k5wl = np.array([[  0, c12, c13,   0,   0],
                 [c21,   0, c23, c24,   0],
                 [c31, c32,   0,   0, c35],
                 [  0, c42,   0,   0, c45],
                 [  0,   0, c53, c54,   0]])
results5wl = integrate(p5wl, k5wl, t_max, max_step)
probs5wl = results5wl.y[:N5wl]
sp5wl_ODE = []
for i in probs5wl:
    sp5wl_ODE.append(i[-1])

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
sp6_diag, sp6_mult = kda.calc_state_probabilities(G6, directional_partials6, state_mults=True)

#== Manual =====================================================================
P1 = a65*a54*a43*a32*a21 + a61*a54*a43*a32*a21 + a56*a61*a43*a32*a21 + a45*a56*a61*a32*a21 + a34*a45*a56*a61*a21 + a23*a34*a45*a56*a61
P2 = a12*a65*a54*a43*a32 + a61*a12*a54*a43*a32 + a56*a61*a12*a43*a32 + a45*a56*a61*a32*a12 + a34*a45*a56*a61*a12 + a16*a65*a54*a43*a32
P3 = a12*a23*a65*a54*a43 + a61*a12*a23*a54*a43 + a56*a61*a12*a23*a43 + a45*a56*a61*a12*a23 + a21*a16*a65*a54*a43 + a23*a16*a65*a54*a43
P4 = a65*a54*a12*a23*a34 + a54*a61*a12*a23*a34 + a56*a61*a12*a23*a34 + a32*a21*a16*a65*a54 + a21*a16*a65*a54*a34 + a16*a65*a54*a23*a34
P5 = a65*a12*a23*a34*a45 + a61*a12*a23*a34*a45 + a43*a32*a21*a16*a65 + a45*a32*a21*a16*a65 + a34*a45*a21*a16*a65 + a23*a34*a45*a16*a65
P6 = a12*a23*a34*a45*a56 + a54*a43*a32*a21*a16 + a56*a43*a32*a21*a16 + a45*a56*a32*a21*a16 + a34*a45*a56*a21*a16 + a16*a23*a34*a45*a56
Sigma = P1 + P2 + P3 + P4 + P5 + P6 # Normalization factor
sp6_manual = np.array([P1, P2, P3, P4, P5, P6])/Sigma
#== ODE ========================================================================
N6 = 6   # number of states in cycle
p_list = np.random.rand(N6)  # generate random probabilities
sigma = p_list.sum(axis=0)  # normalization factor
p6 = p_list/sigma       # normalize probabilities
k6 = np.array([[  0, a12,   0,   0,   0, a16],
               [a21,   0, a23,   0,   0,   0],
               [  0, a32,   0, a34,   0,   0],
               [  0,   0, a43,   0, a45,   0],
               [  0,   0,   0, a54,   0, a56],
               [a61,   0,   0,   0, a65,   0]])
results6 = integrate(p6, k6, t_max, max_step)
probs6 = results6.y[:N6]
sp6_ODE = []
for i in probs6:
    sp6_ODE.append(i[-1])

#===============================================================================
#== Verification ===============================================================
#===============================================================================

print("======== Three State ========")
for i in range(N3):
    print("State {}: manual = {}, diag = {}, ODE = {}".format(i+1, sp3_manual[i], sp3_diag[i], sp3_ODE[i]))
print("======== Four State ========")
for i in range(N4):
    print("State {}: manual = {}, diag = {}, ODE = {}".format(i+1, sp4_manual[i], sp4_diag[i], sp4_ODE[i]))
print("======== Four State with Leakage ========")
for i in range(N4wl):
    print("State {}: manual = {}, diag = {}, ODE = {}".format(i+1, sp4wl_manual[i], sp4wl_diag[i], sp4wl_ODE[i]))
print("======== Five State with Leakage ========")
for i in range(N5wl):
    print("State {}: manual = {}, diag = {}, ODE = {}".format(i+1, sp5wl_manual[i], sp5wl_diag[i], sp5wl_ODE[i]))
print("======== Six State ========")
for i in range(N6):
    print("State {}: manual = {}, diag = {}, ODE = {}".format(i+1, sp6_manual[i], sp6_diag[i], sp6_ODE[i]))

# print(sp3_manual - sp3_ODE)
# print(sp4_manual - sp4_ODE)
# print(sp4wl_manual - sp4wl_ODE)
# print(sp5wl_manual - sp5wl_ODE)
# print(sp6_manual - sp6_ODE)

if plot == True:
    plot_ODE_probs(results3)
    plot_ODE_probs(results4)
    plot_ODE_probs(results4wl)
    plot_ODE_probs(results5wl)
    plot_ODE_probs(results6)
# check probabilities are normalized to 1





#===
