# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
# 03/19/2020

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy
from sympy import *
from sympy.utilities.lambdify import lambdify

import hill_biochemical_kinetic_diagram_analyzer as kda

from three_state_model import generate_edges as ge3
from four_state_model import generate_edges as ge4
from four_state_model_with_leakage import generate_edges as ge4wl
from five_state_model_with_leakage import generate_edges as ge5wl
from six_state_model import generate_edges as ge6

#===============================================================================
#== 3 State ====================================================================
#===============================================================================

# k12 = 2
# k21 = 3
# k23 = 5
# k32 = 7
# k13 = 11
# k31 = 13
# rates3 = [k12, k21, k23, k32, k13, k31]
# rate_names3 = ["x12", "x21", "x23", "x32", "x13", "x31"]
# G3 = nx.MultiDiGraph()
# ge3(G3, rates3)
# pars3 = kda.generate_partial_diagrams(G3)
# dir_pars3 = kda.generate_directional_partial_diagrams(pars3)
# var_dict3 = kda.generate_rate_dict(rates3, rate_names3)
# state_mult_funcs3, norm_func3 = kda.construct_analytic_functions(G3, dir_pars3, var_dict3)
#
# print("===== 3 state model =====")
# for i, func in enumerate(state_mult_funcs3):
#     print(i+1, func)
# print("Normalization Factor:", norm_func3)

# #===============================================================================
# #== 4 State ====================================================================
# #===============================================================================
# y12 = 2
# y21 = 3
# y23 = 5
# y32 = 7
# y34 = 11
# y43 = 13
# y41 = 17
# y14 = 19
# rates4 = [y12, y21, y23, y32, y34, y43, y41, y14]
# rate_names4 = ["y12", "y21", "y23", "y32", "y34", "y43", "y41", "y14"]
# G4 = nx.MultiDiGraph()
# ge4(G4, rates4)
# pars4 = kda.generate_partial_diagrams(G4)
# dir_pars4 = kda.generate_directional_partial_diagrams(pars4)
# var_dict4 = kda.generate_rate_dict(rates4, rate_names4)
# state_mult_funcs4, norm_func4 = kda.construct_analytic_functions(G4, dir_pars4, var_dict4)
#
# print("===== 4 state model =====")
# for i, func in enumerate(state_mult_funcs4):
#     print(i+1, func)
# print("Normalization Factor:", norm_func4)
#
# #===============================================================================
# #== 4 State w/ Leakage =========================================================
# #===============================================================================
# y12 = 2
# y21 = 3
# y23 = 5
# y32 = 7
# y34 = 11
# y43 = 13
# y41 = 17
# y14 = 19
# y24 = 23
# y42 = 29
# rates4wl = [y12, y21, y23, y32, y34, y43, y41, y14, y24, y42]
# rate_names4wl = ["y12", "y21", "y23", "y32", "y34", "y43", "y41", "y14", "y24", "y42"]
# G4wl = nx.MultiDiGraph()
# ge4wl(G4wl, rates4wl)
# pars4wl = kda.generate_partial_diagrams(G4wl)
# dir_pars4wl = kda.generate_directional_partial_diagrams(pars4wl)
# var_dict4wl = kda.generate_rate_dict(rates4wl, rate_names4wl)
# state_mult_funcs4wl, norm_func4wl = kda.construct_analytic_functions(G4wl, dir_pars4wl, var_dict4wl)
#
# print("===== 4 state model w/ Leakage =====")
# for i, func in enumerate(state_mult_funcs4wl):
#     print(i+1, func)
# print("Normalization Factor:", norm_func4wl)
#
# #===============================================================================
# #== 5 State w/ Leakage =========================================================
# #===============================================================================
# z12 = 2
# z21 = 3
# z23 = 5
# z32 = 7
# z13 = 11
# z31 = 13
# z24 = 17
# z42 = 19
# z35 = 31
# z53 = 37
# z45 = 23
# z54 = 29
# rates5wl = [z12, z21, z23, z32, z13, z31, z24, z42, z35, z53, z45, z54]
# rate_names5wl = ["z12", "z21", "z23", "z32", "z13", "z31", "z24", "z42", "z35", "z53", "z45", "z54"]
# G5wl = nx.MultiDiGraph()
# ge5wl(G5wl, rates5wl)
# pars5wl = kda.generate_partial_diagrams(G5wl)
# dir_pars5wl = kda.generate_directional_partial_diagrams(pars5wl)
# var_dict5wl = kda.generate_rate_dict(rates5wl, rate_names5wl)
# state_mult_funcs5wl, norm_func5wl = kda.construct_analytic_functions(G5wl, dir_pars5wl, var_dict5wl)
#
# print("===== 5 state model w/ Leakage =====")
# for i, func in enumerate(state_mult_funcs5wl):
#     print(i+1, func)
# print("Normalization Factor:", norm_func5wl)
#
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
var_dict6 = kda.generate_rate_dict(rates6, rate_names6)
state_mult_funcs6, norm_func6 = kda.construct_analytic_functions(G6, dir_pars6, var_dict6)

print("===== 6 state model w/ Leakage =====")
for i, func in enumerate(state_mult_funcs6):
    print(i+1, func)
print("Normalization Factor:", norm_func6)

#===============================================================================
#== Generate Probability Functions =============================================
#===============================================================================

state_1_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[0], norm_func6, vars=True)
state_2_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[1], norm_func6, vars=True)
state_3_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[2], norm_func6, vars=True)
state_4_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[3], norm_func6, vars=True)
state_5_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[4], norm_func6, vars=True)
state_6_prob_func6_sym, names6 = kda.gen_analytic_prob_func(rate_names6, state_mult_funcs6[5], norm_func6, vars=True)

state_1_prob_func6 = lambdify(rate_names6, state_1_prob_func6_sym)
state_2_prob_func6 = lambdify(rate_names6, state_2_prob_func6_sym)
state_3_prob_func6 = lambdify(rate_names6, state_3_prob_func6_sym)
state_4_prob_func6 = lambdify(rate_names6, state_4_prob_func6_sym)
state_5_prob_func6 = lambdify(rate_names6, state_5_prob_func6_sym)
state_6_prob_func6 = lambdify(rate_names6, state_6_prob_func6_sym)

p1_6 = state_1_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
p2_6 = state_2_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
p3_6 = state_3_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
p4_6 = state_4_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
p5_6 = state_5_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
p6_6 = state_6_prob_func6(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

six_state_probs = [p1_6, p2_6, p3_6, p4_6, p5_6, p6_6]


#===============================================================================
#== Verify SymPy Multiplicities ================================================
#===============================================================================

var_dict6s = kda.generate_rate_dict(rate_names6, rates6)
mult_funcs6, state_mults6 = kda.calc_sympy_state_mult(state_mult_funcs6, rate_names6, var_dict6s)
sp6_diag, sp6_mult = kda.calc_state_probabilities(G6, dir_pars6, state_mults=True)
print(sp6_mult - state_mults6)
print(sp6_diag - six_state_probs)
# Simplify 6 state model like we did for ODE solver
# x, y = symbols('x y')
# x12, x21, x23, x32, x34, x43, x45, x54, x56, x65, x61, x16 = names6
# new_prob_func6 = prob_func6.subs({x23: y, x32: y, x56: y, x65: y, x61: x, x21: x, x61: x, x43: x, x61: x, x45: x})
# print(new_prob_func6)

# def gen_analytic_prob_func(rate_names, state_func, norm_func):
#     var_names = " ".join(rate_names)
#     var_names = symbols(var_names)
#     prob_func = sympy.parsing.sympy_parser.parse_expr(state_func)/sympy.parsing.sympy_parser.parse_expr(norm_func)
#     return prob_func

# def generate_analytic_prob_func(rate_names, state_func, norm_func):
#
#     def prob_func(*rate_names):
#         return NotImplementedError
#
#     return prob_func




#===
