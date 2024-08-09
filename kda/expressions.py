# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Algebraic Expressions
=========================================================================
This file contains a host of functions aimed at creating SymPy/Python
functions from algebraic expressions in string form.

.. autofunction:: construct_sympy_prob_funcs
.. autofunction:: construct_sympy_net_cycle_flux_func
.. autofunction:: construct_lambda_funcs

"""


from sympy import lambdify, Mul
from sympy.parsing.sympy_parser import parse_expr


def construct_sympy_prob_funcs(state_mult_funcs):
    """
    Constructs analytic state probability SymPy functions

    Parameters
    ----------
    state_mult_funcs : list of str
        List of length 'N', where N is the number of states, that contains the
        analytic multiplicity function for each state
    norm_func : str
        Sum of all state multiplicity functions, the normalization factor to
        calculate the state probabilities

    Returns
    -------
    sympy_funcs : list
        List of analytic state probability SymPy functions.
    """
    # convert the state multiplicity strings into sympy expressions
    parsed_mult_funcs = [parse_expr(e) for e in state_mult_funcs]
    # create the normalization expression
    parsed_sigma = sum(parsed_mult_funcs)
    # normalize the multiplicity expressions
    prob_funcs = [e/parsed_sigma for e in parsed_mult_funcs]
    return prob_funcs



def construct_sympy_net_cycle_flux_func(pi_diff_str, sigma_K_str, sigma_str):
    """
    Creates the analytic net cycle flux SymPy function for a given cycle.

    Parameters
    ----------
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    sigma_K_str : str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle in string form.
    sigma_str : str
        Sum of rate products of all directional diagrams for input
        diagram G, in string form.

    Returns
    -------
    net_cycle_flux_func : SymPy object
        Analytic net cycle flux SymPy function
    """
    if sigma_K_str == 1:
        net_cycle_flux_func = parse_expr(pi_diff_str) / parse_expr(sigma_str)
        return net_cycle_flux_func
    else:
        net_cycle_flux_func = (
            parse_expr(pi_diff_str) * parse_expr(sigma_K_str)
        ) / parse_expr(sigma_str)
        return net_cycle_flux_func


def construct_lambda_funcs(sympy_funcs, rate_names):
    """
    Constructs Python lambda functions from SymPy functions.

    Parameters
    ----------
    sympy_funcs : list of SymPy functions
        List of SymPy functions.
    rate_names : list
        List of strings, where each element is the name of the variables for
        the input probability functions, ["x12", "x21", "x23", ...].

    Returns
    -------
    state_prob_funcs : list
        List of lambdified analytic state probability functions.
    """
    if isinstance(sympy_funcs, Mul) == True:
        return lambdify(rate_names, sympy_funcs, "numpy")
    elif isinstance(sympy_funcs, list) == True:
        state_prob_funcs = (
            []
        )  # create empty list to fill with state probability functions
        for func in sympy_funcs:
            state_prob_funcs.append(
                lambdify(rate_names, func, "numpy")
            )  # convert into "lambdified" functions that work with NumPy arrays
        return state_prob_funcs
