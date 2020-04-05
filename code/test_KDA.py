# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/30/2020
# Biochemical Kinetic Diagram Analyzer Testing

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import kinetic_diagram_analyzer as kda
from plotting import plot_ODE_probs

# ODE Parameters
t_max = 5e0
plot = False
path = "C:/Users/nikol/phy495/hill-biochemical-kinetic-diagram-analyzer/data/html"

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
kda.generate_edges(G3, k3s, k3, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars3 = kda.generate_partial_diagrams(G3)
dir_pars3 = kda.generate_directional_partial_diagrams(pars3)
sp3_KDA = kda.calc_state_probabilities(G3, dir_pars3, key='val')
#== State Func probabilitites ==================================================
state_mults3, norm3 = kda.calc_state_probabilities(G3, dir_pars3, key='name', output_strings=True)
sympy_funcs3 = kda.construct_sympy_funcs(state_mults3, norm3)
state_prob_funcs3 = kda.construct_lambdify_funcs(sympy_funcs3, rate_names3)
sp3_SymPy = []
for i in range(G3.number_of_nodes()):
    sp3_SymPy.append(state_prob_funcs3[i](k12, k21, k23, k32, k13, k31))
sp3_SymPy = np.array(sp3_SymPy)
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
kda.generate_edges(G4, k4s, k4, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars4 = kda.generate_partial_diagrams(G4)
dir_pars4 = kda.generate_directional_partial_diagrams(pars4)
sp4_KDA = kda.calc_state_probabilities(G4, dir_pars4, key='val')
#== State Func probabilitites ==================================================
state_mults4, norm4 = kda.calc_state_probabilities(G4, dir_pars4, key='name', output_strings=True)
sympy_funcs4 = kda.construct_sympy_funcs(state_mults4, norm4)
state_prob_funcs4 = kda.construct_lambdify_funcs(sympy_funcs4, rate_names4)
sp4_SymPy = []
for i in range(G4.number_of_nodes()):
    sp4_SymPy.append(state_prob_funcs4[i](b12, b21, b23, b32, b34, b43, b41, b14))
sp4_SymPy = np.array(sp4_SymPy)
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
kda.generate_edges(G4wl, k4wls, k4wl, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars4wl = kda.generate_partial_diagrams(G4wl)
dir_pars4wl = kda.generate_directional_partial_diagrams(pars4wl)
sp4wl_KDA = kda.calc_state_probabilities(G4wl, dir_pars4wl, key='val')
#== State Func probabilitites ==================================================
state_mults4wl, norm4wl = kda.calc_state_probabilities(G4wl, dir_pars4wl, key='name', output_strings=True)
sympy_funcs4wl = kda.construct_sympy_funcs(state_mults4wl, norm4wl)
state_prob_funcs4wl = kda.construct_lambdify_funcs(sympy_funcs4wl, rate_names4wl)
sp4wl_SymPy = []
for i in range(G4wl.number_of_nodes()):
    sp4wl_SymPy.append(state_prob_funcs4wl[i](b12, b21, b23, b32, b34, b43, b41, b14, b24, b42))
sp4wl_SymPy = np.array(sp4wl_SymPy)
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
kda.generate_edges(G5wl, k5wls, k5wl, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars5wl = kda.generate_partial_diagrams(G5wl)
dir_pars5wl = kda.generate_directional_partial_diagrams(pars5wl)
sp5wl_KDA = kda.calc_state_probabilities(G5wl, dir_pars5wl, key='val')
#== State Func probabilitites ==================================================
state_mults5wl, norm5wl = kda.calc_state_probabilities(G5wl, dir_pars5wl, key='name', output_strings=True)
sympy_funcs5wl = kda.construct_sympy_funcs(state_mults5wl, norm5wl)
state_prob_funcs5wl = kda.construct_lambdify_funcs(sympy_funcs5wl, rate_names5wl)
sp5wl_SymPy = []
for i in range(G5wl.number_of_nodes()):
    sp5wl_SymPy.append(state_prob_funcs5wl[i](c12, c21, c23, c32, c13, c31, c24, c42, c35, c53, c45, c54))
sp5wl_SymPy = np.array(sp5wl_SymPy)
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
kda.generate_edges(G6, k6s, k6, name_key='name', val_key='val')
#== State probabilitites =======================================================
pars6 = kda.generate_partial_diagrams(G6)
dir_pars6 = kda.generate_directional_partial_diagrams(pars6)
sp6_KDA = kda.calc_state_probabilities(G6, dir_pars6, key='val')
#== State Func probabilitites ==================================================
state_mults6, norm6 = kda.calc_state_probabilities(G6, dir_pars6, key='name', output_strings=True)
sympy_funcs6 = kda.construct_sympy_funcs(state_mults6, norm6)
state_prob_funcs6 = kda.construct_lambdify_funcs(sympy_funcs6, rate_names6)
sp6_SymPy = []
for i in range(G6.number_of_nodes()):
    sp6_SymPy.append(state_prob_funcs6[i](a12, a21, a23, a32, a34, a43, a45, a54, a56, a65, a61, a16))
sp6_SymPy = np.array(sp6_SymPy)
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
results6 = kda.solve_ODE(p6, k6, t_max)
probs6 = results6.y[:N6]
sp6_ODE = []
for i in probs6:
    sp6_ODE.append(i[-1])
sp6_ODE = np.array(sp6_ODE)


#===============================================================================
#== Verification ===============================================================
#===============================================================================

if plot == True:
    plot_ODE_probs(results3)
    plot_ODE_probs(results4)
    plot_ODE_probs(results4wl)
    plot_ODE_probs(results5wl)
    plot_ODE_probs(results6)

def rel_error(theoretical, experimental):
    return np.abs(theoretical - experimental)/theoretical

def get_error_array(manual, other):
    N_methods = len(other)
    N_states = len(other[0])
    error_array = np.zeros((N_methods, N_states))
    for m in range(N_methods):
        for s in range(N_states):
            error_array[m, s] = rel_error(manual[s], other[m, s])
    return error_array

def get_diff_array(theoretical, experimental):
    N_methods = len(experimental)
    N_states = len(experimental[0])
    diff_array = np.zeros((N_methods, N_states))
    for m in range(N_methods):
        for s in range(N_states):
            diff_array[m, s] = theoretical[s] - experimental[m, s]
    return diff_array

def generate_table(state_probs, name, path=path):
    N = len(state_probs)
    cols = ["Theoretical", "KDA", "SymPy", "ODE", "KDA Diff", "SymPy Diff", "ODE Diff", "KDA Error", "SymPy Error", "ODE Error"]
    rows = ["State {}".format(i+1) for i in range(N)]
    df = pd.DataFrame(data=state_probs, index=rows, columns=cols)
    df.to_html(path + '/{}.html'.format(name))
    print(df)

# Make arrays of all probability values for all 3 methods
sp3_probs = np.array([sp3_KDA, sp3_SymPy, sp3_ODE])
sp4_probs = np.array([sp4_KDA, sp4_SymPy, sp4_ODE])
sp4wl_probs = np.array([sp4wl_KDA, sp4wl_SymPy, sp4wl_ODE])
sp5wl_probs = np.array([sp5wl_KDA, sp5wl_SymPy, sp5wl_ODE])
sp6_probs = np.array([sp6_KDA, sp6_SymPy, sp6_ODE])

# Make arrays of all probability differences for all 3 methods
sp3_diffs = get_diff_array(sp3_manual, sp3_probs)
sp4_diffs = get_diff_array(sp4_manual, sp4_probs)
sp4wl_diffs = get_diff_array(sp4wl_manual, sp4wl_probs)
sp5wl_diffs = get_diff_array(sp5wl_manual, sp5wl_probs)
sp6_diffs = get_diff_array(sp6_manual, sp6_probs)

# Make arrays of all relative error values for all 3 methods
sp3_error = get_error_array(sp3_manual, sp3_probs)
sp4_error = get_error_array(sp4_manual, sp4_probs)
sp4wl_error = get_error_array(sp4wl_manual, sp4wl_probs)
sp5wl_error = get_error_array(sp5wl_manual, sp5wl_probs)
sp6_error = get_error_array(sp6_manual, sp6_probs)

# Stack arrays
sp3 = np.vstack((sp3_manual, sp3_probs, sp3_diffs, sp3_error)).T
sp4 = np.vstack((sp4_manual, sp4_probs, sp4_diffs, sp4_error)).T
sp4wl = np.vstack((sp4wl_manual, sp4wl_probs, sp4wl_diffs, sp4wl_error)).T
sp5wl = np.vstack((sp5wl_manual, sp5wl_probs, sp5wl_diffs, sp5wl_error)).T
sp6 = np.vstack((sp6_manual, sp6_probs, sp6_diffs, sp6_error)).T

# Generate dataframes, print, and output to html
generate_table(sp3, name="sp3")
generate_table(sp4, name="sp4")
generate_table(sp4wl, name="sp4wl")
generate_table(sp5wl, name="sp5wl")
generate_table(sp6, name="sp6")




#===


def test_three_state():
    return NotImplementedError

def test_four_state():
    return NotImplementedError

def test_four_state_with_leakage():
    return NotImplementedError

def test_five_state_with_leakage():
    return NotImplementedError

def test_six_state():
    return NotImplementedError
