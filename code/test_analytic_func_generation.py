import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy
from sympy import *

import hill_biochemical_kinetic_diagram_analyzer as kda

from three_state_model import generate_edges as ge3
from four_state_model import generate_edges as ge4
from four_state_model_with_leakage import generate_edges as ge4wl
from five_state_model_with_leakage import generate_edges as ge5wl
from six_state_model import generate_edges as ge6

#===============================================================================
#== Functions ==================================================================
#===============================================================================

def generate_variable_dictionary(rates, rate_names):
    """
    Generates dictionary where rate constant values are keys and rate constant
    names are the values.
    """
    var_dict = dict.fromkeys(rates, {})
    for i in range(len(rates)):
        var_dict[rates[i]] = rate_names[i]
    return var_dict

def construct_analytic_functions(G, dir_parts, var_dict):
    """
    This function will input a list of all directional partial diagrams and
    output a list of analytic functions for the steady-state probability of
    each state in the original diagram.

    Parameters
    ----------
    G : networkx diagram object
    dir_parts : list of networkx diagram objects
        List of all directional partial diagrams for the given diagram "G"
    var_dict : dict
        Dictionary where the rate constant values are the keys and the rate
        constant names are the values

    Returns
    -------
    state_mults : list
        List of length 'N', where N is the number of states, that contains the
        analytic multiplicity function for each state
    norm : str
        Sum of all state multiplicity functions, the normalization factor to
        calculate the state probabilities
    """
    state_mults = []    # create empty list to fill with summed terms
    for s in range(G.number_of_nodes()):    # iterate over number of states, "s"
        part_mults = []    # generate zero array of length # of directional partial diagrams
        for i in range(len(dir_parts)):      # iterate over the directional partial diagrams
            edge_list = list(dir_parts[i].edges)     # get a list of all edges for partial directional diagram i
            products = []          # generate an empty list to store individual products of terms
            for edge in edge_list:
                products.append(var_dict[G.edges[edge[0], edge[1], edge[2]]['weight']])
            part_mults.append(products)
    N_terms = int(len(part_mults)/G.number_of_nodes()) # number of terms per state
    term_list = []
    for vars in part_mults:
        term_list.append("*".join(vars))
    for j in range(G.number_of_nodes()):
        state_mults.append("+".join(term_list[N_terms*j:N_terms*j+N_terms]))
    norm = "+".join(state_mults)
    return state_mults, norm

def output_sympy_state_prob_function(rate_names, state_func, norm_func, latex=None):
    joined_names = " ".join(rate_names)
    joined_names = symbols(joined_names)
    init_printing(use_unicode=True)
    if latex == True:
        return latex(simplify(sympy.parsing.sympy_parser.parse_expr(state_func)/sympy.parsing.sympy_parser.parse_expr(norm_func)))
    else:
        return simplify(sympy.parsing.sympy_parser.parse_expr(state_func)/sympy.parsing.sympy_parser.parse_expr(norm_func))

#===============================================================================
#== 3 State ====================================================================
#===============================================================================

# from sympy import *
# k12, k21, k23, k32, k13, k31 = symbols('k12 k21 k23 k32 k13 k31')
k12 = 2
k21 = 3
k23 = 5
k32 = 7
k13 = 11
k31 = 13
rates3 = [k12, k21, k23, k32, k13, k31]
rate_names3 = ["k12", "k21", "k23", "k32", "k13", "k31"]
G3 = nx.MultiDiGraph()
ge3(G3, rates3)
pars3 = kda.generate_partial_diagrams(G3)
dir_pars3 = kda.generate_directional_partial_diagrams(pars3)
var_dict3 = generate_variable_dictionary(rates3, rate_names3)
state_mult_funcs3, norm_func3 = construct_analytic_functions(G3, dir_pars3, var_dict3)

print("===== 3 state model =====")
for i, func in enumerate(state_mult_funcs3):
    print(i+1, func)
print("Normalization Factor:", norm_func3)

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
rates4 = [b12, b21, b23, b32, b34, b43, b41, b14]
rate_names4 = ["b12", "b21", "b23", "b32", "b34", "b43", "b41", "b14"]
G4 = nx.MultiDiGraph()
ge4(G4, rates4)
pars4 = kda.generate_partial_diagrams(G4)
dir_pars4 = kda.generate_directional_partial_diagrams(pars4)
var_dict4 = generate_variable_dictionary(rates4, rate_names4)
state_mult_funcs4, norm_func4 = construct_analytic_functions(G4, dir_pars4, var_dict4)

print("===== 4 state model =====")
for i, func in enumerate(state_mult_funcs4):
    print(i+1, func)
print("Normalization Factor:", norm_func4)

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
rates4wl = [b12, b21, b23, b32, b34, b43, b41, b14, b24, b42]
rate_names4wl = ["b12", "b21", "b23", "b32", "b34", "b43", "b41", "b14", "b24", "b42"]
G4wl = nx.MultiDiGraph()
ge4wl(G4wl, rates4wl)
pars4wl = kda.generate_partial_diagrams(G4wl)
dir_pars4wl = kda.generate_directional_partial_diagrams(pars4wl)
var_dict4wl = generate_variable_dictionary(rates4wl, rate_names4wl)
state_mult_funcs4wl, norm_func4wl = construct_analytic_functions(G4wl, dir_pars4wl, var_dict4wl)

print("===== 4 state model w/ Leakage =====")
for i, func in enumerate(state_mult_funcs4wl):
    print(i+1, func)
print("Normalization Factor:", norm_func4wl)

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
rates5wl = [c12, c21, c23, c32, c13, c31, c24, c42, c35, c53, c45, c54]
rate_names5wl = ["c12", "c21", "c23", "c32", "c13", "c31", "c24", "c42", "c35", "c53", "c45", "c54"]
G5wl = nx.MultiDiGraph()
ge5wl(G5wl, rates5wl)
pars5wl = kda.generate_partial_diagrams(G5wl)
dir_pars5wl = kda.generate_directional_partial_diagrams(pars5wl)
var_dict5wl = generate_variable_dictionary(rates5wl, rate_names5wl)
state_mult_funcs5wl, norm_func5wl = construct_analytic_functions(G5wl, dir_pars5wl, var_dict5wl)

print("===== 5 state model w/ Leakage =====")
for i, func in enumerate(state_mult_funcs5wl):
    print(i+1, func)
print("Normalization Factor:", norm_func5wl)

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
rates6 = [a12, a21, a23, a32, a34, a43, a45, a54, a56, a65, a61, a16]
rate_names6 = ["a12", "a21", "a23", "a32", "a34", "a43", "a45", "a54", "a56", "a65", "a61", "a16"]
G6 = nx.MultiDiGraph()
ge6(G6, rates6)
pars6 = kda.generate_partial_diagrams(G6)
dir_pars6 = kda.generate_directional_partial_diagrams(pars6)
var_dict6 = generate_variable_dictionary(rates6, rate_names6)
state_mult_funcs6, norm_func6 = construct_analytic_functions(G6, dir_pars6, var_dict6)

print("===== 6 state model w/ Leakage =====")
for i, func in enumerate(state_mult_funcs6):
    print(i+1, func)
print("Normalization Factor:", norm_func6)

#== LaTeX ======================================================================

a = output_sympy_state_prob_function(rate_names5wl, state_mult_funcs5wl[0], norm_func5wl)
print(latex(a))
