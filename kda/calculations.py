# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: State Probability & Flux Calculations
=========================================================================
This file contains a host of functions aimed at calculating quantities of
interest from biochemical kinetic diagrams, using the methods of T.L. Hill.

.. autofunction:: calc_state_probs
.. autofunction:: calc_net_cycle_flux
.. autofunction:: calc_sigma
.. autofunction:: calc_sigma_K
.. autofunction:: calc_pi_difference
.. autofunction:: calc_thermo_force
.. autofunction:: calc_state_probs_from_diags
"""

import numpy as np
import networkx as nx
from sympy import parse_expr, logcombine

from kda import graphs, diagrams, expressions


def _get_ordered_cycle(G, cycle):
    """
    Takes in arbitrary list of nodes and returns list of nodes in correct order.
    Can be used in conjunction with `diagrams._construct_cycle_edges` to
    generate list of edge tuples for an arbitrary input cycle. Assumes input
    cycle only exists once in the input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    ordered_cycles : list of int, list of lists of int
        Ordered list of nodes for the input cycle, or if several cycles are
        found, a list of lists of nodes for the input cycle.
    """
    ordered_cycles = []
    for cyc in graphs.find_all_unique_cycles(G):
        if sorted(cyc) == sorted(cycle):
            ordered_cycles.append(cyc)
    if ordered_cycles == []:
        print("No cycles found for nodes {}.".format(cycle))
    elif len(ordered_cycles) > 1:
        return ordered_cycles
    else:
        return ordered_cycles[0]


def calc_state_probs_from_diags(G, dir_partials, key, output_strings=False):
    """
    Calculates state probabilities and generates analytic function strings from
    input diagram and directional partial diagrams. If directional partial
    diagrams are already generated, this offers faster calculation than
    `calc_state_probs`.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic state multplicity and normalization functions.

    Returns
    -------
    state_probabilities : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    state_mults : list
        List of analytic state multiplicity functions in string form.
    norm : str
        Analytic state multiplicity function normalization function in
        string form. This is the sum of all multiplicty functions.
    """
    N = G.number_of_nodes()  # Number of nodes/states
    partial_mults = []
    edges = list(G.edges)
    if output_strings == False:
        if isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        for i in range(
            len(dir_partials)
        ):  # iterate over the directional partial diagrams
            edge_list = list(
                dir_partials[i].edges
            )  # get a list of all edges for partial directional diagram i
            products = 1  # assign initial value of 1
            for (
                e
            ) in (
                edge_list
            ):  # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]][
                    key
                ]  # multiply the rate of each edge in edge_list
            partial_mults.append(products)
        N_terms = np.int(
            len(dir_partials) / N
        )  # calculate the number of terms to be summed for given state, s
        state_mults = []
        partial_mults = np.array(partial_mults)
        for s in range(N):  # iterate over number of states, "s"
            state_mults.append(
                partial_mults[N_terms * s : N_terms * s + N_terms].sum(axis=0)
            )
        state_mults = np.array(state_mults)
        state_probs = state_mults / state_mults.sum(axis=0)
        if any(elem < 0 for elem in state_probs) == True:
            raise ValueError(
                "Calculated negative state probabilities, overflow or underflow occurred."
            )
        return state_probs
    elif output_strings == True:
        if not isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        for i in range(
            len(dir_partials)
        ):  # iterate over the directional partial diagrams
            edge_list = list(
                dir_partials[i].edges
            )  # get a list of all edges for partial directional diagram i
            products = []
            for e in edge_list:
                products.append(
                    G.edges[e[0], e[1], e[2]][key]
                )  # append rate constant names from dir_par to list
            partial_mults.append(products)
        N_terms = np.int(
            len(dir_partials) / N
        )  # calculate the number of terms to be summed for given state, s
        state_mults = []
        term_list = []  # create empty list to put products of rate constants (terms) in
        for k in partial_mults:
            term_list.append(
                "*".join(k)
            )  # join rate constants for each dir_par by delimeter "*"
        for s in range(N):  # iterate over number of states, "s"
            state_mults.append(
                "+".join(term_list[N_terms * s : N_terms * s + N_terms])
            )  # join appropriate terms for each state by delimeter "+"
        norm = "+".join(state_mults)  # sum all terms to get normalization factor
        return state_mults, norm


def calc_sigma(G, dir_partials, key, output_strings=False):
    """
    Calculates sigma, the normalization factor for calculating state
    probabilities and cycle fluxes for a given diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    dir_partials : list
        List of all directional partial diagrams for the input diagram G.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the normalization factor
        using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic cycle flux function.
    Returns
    -------
    sigma : float
        Normalization factor for state probabilities.
    sigma_str : str
        Sum of rate products of all directional partial diagrams for input
        diagram G, in string form.
    """
    N = G.number_of_nodes()  # Number of nodes/states
    partial_mults = []
    edges = list(G.edges)
    if output_strings == False:
        if isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        for i in range(
            len(dir_partials)
        ):  # iterate over the directional partial diagrams
            edge_list = list(
                dir_partials[i].edges
            )  # get a list of all edges for partial directional diagram i
            products = 1  # assign initial value of 1
            for (
                e
            ) in (
                edge_list
            ):  # iterate over the edges in the given directional partial diagram i
                products *= G.edges[e[0], e[1], e[2]][
                    key
                ]  # multiply the rate of each edge in edge_list
            partial_mults.append(products)
        sigma = np.array(partial_mults).sum(axis=0)
        return sigma
    elif output_strings == True:
        if not isinstance(G.edges[edges[0][0], edges[0][1], edges[0][2]][key], str):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        for i in range(
            len(dir_partials)
        ):  # iterate over the directional partial diagrams
            edge_list = list(
                dir_partials[i].edges
            )  # get a list of all edges for partial directional diagram i
            products = []
            for e in edge_list:
                products.append(
                    G.edges[e[0], e[1], e[2]][key]
                )  # append rate constant names from dir_par to list
            partial_mults.append(products)
        state_mults = []
        term_list = []  # create empty list to put products of rate constants (terms) in
        for k in partial_mults:
            term_list.append(
                "*".join(k)
            )  # join rate constants for each dir_par by delimeter "*"
        sigma_str = "+".join(term_list)  # sum all terms to get normalization factor
        return sigma_str


def calc_sigma_K(G, cycle, flux_diags, key, output_strings=False):
    """
    Calculates sigma_K, the sum of all directional flux diagrams.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter but should not contain all nodes.
    flux_diags : list
        List of relevant directional flux diagrams for input cycle.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the sum of all
        directional flux diagrams using numbers. If True, this will assume the
        input 'key' will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    sigma_K : float
        Sum of rate products of directional flux diagram edges pointing to
        input cycle.
    sigma_K_str : str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle in string form.
    """
    if isinstance(flux_diags, list) == False:
        print(
            "No flux diagrams detected for cycle {}. Sigma K value is 1.".format(cycle)
        )
        return 1
    else:
        ordered_cycle = _get_ordered_cycle(G, cycle)
        cycle_edges = diagrams._construct_cycle_edges(ordered_cycle)
        if output_strings == False:
            if isinstance(
                G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key],
                str,
            ):
                raise TypeError(
                    "To enter variable strings set parameter output_strings=True."
                )
            rate_products = []
            for diagram in flux_diags:
                diag = diagram.copy()
                for edge in cycle_edges:
                    diag.remove_edge(edge[0], edge[1], edge[2])
                    diag.remove_edge(edge[1], edge[0], edge[2])
                vals = 1
                for edge in diag.edges:
                    vals *= G.edges[edge[0], edge[1], edge[2]][key]
                rate_products.append(vals)
            sigma_K = np.array(rate_products).sum(axis=0)
            return sigma_K
        elif output_strings == True:
            if not isinstance(
                G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key],
                str,
            ):
                raise TypeError(
                    "To enter variable values set parameter output_strings=False."
                )
            rate_products = []
            for diagram in flux_diags:
                diag = diagram.copy()
                for edge in cycle_edges:
                    diag.remove_edge(edge[0], edge[1], edge[2])
                    diag.remove_edge(edge[1], edge[0], edge[2])
                rates = []
                for edge in diag.edges:
                    rates.append(G.edges[edge[0], edge[1], edge[2]][key])
                rate_products.append("*".join(rates))
            sigma_K_str = "+".join(rate_products)
            return sigma_K_str


def calc_pi_difference(G, cycle, order, key, output_strings=False):
    """
    Calculates the difference of the forward and reverse rate products for a
    given cycle, where forward rates are defined as counter clockwise.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2, where the integers must be nodes in the
        input cycle. This pair of nodes is used to determine which direction is
        CCW.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the difference using
        numbers. If True, this will assume the input 'key' will return strings
        of variable names to join into the analytic function.

    Returns
    -------
    pi_diff : float
        Difference of product of counter clockwise cycle rates and clockwise
        cycle rates.
    pi_diff_str : str
        String of difference of product of counter clockwise cycle rates and
        clockwise cycle rates.
    """
    cycle_count = 0
    for cyc in graphs.find_all_unique_cycles(G):
        if sorted(cycle) == sorted(cyc):
            cycle_count += 1
    if cycle_count > 1:  # for all-node cycles
        CCW_cycle = graphs.get_ccw_cycle(cycle, order)
        cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    elif cycle_count == 1:  # for all other cycles
        ordered_cycle = _get_ordered_cycle(G, cycle)
        CCW_cycle = graphs.get_ccw_cycle(ordered_cycle, order)
        cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    else:
        raise CycleError("Cycle {} could not be found in G.".format(cycle))
    if output_strings == False:
        if isinstance(
            G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str
        ):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        pi_difference = ccw_rates - cw_rates
        return pi_difference
    elif output_strings == True:
        if not isinstance(
            G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str
        ):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        pi_difference = "-".join(["*".join(ccw_rates), "*".join(cw_rates)])
        return pi_difference


def calc_thermo_force(G, cycle, order, key, output_strings=False):
    """
    Calculates the thermodynamic driving force for a given cycle in diagram G.
    The driving force is calculated as the natural log of the ratio of the
    forward rate product and the reverse rate product in the cycle, where the
    forward direction is defined as counter clockwise. The value returned should
    be multiplied by 'kT' to obtain the actual thermodynamic force, in units of
    energy.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2, where the integers must be nodes in the
        input cycle. This pair of nodes is used to determine which direction is
        CCW.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the thermodynamic force
        using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    thermo_force : float
        The calculated thermodynamic force for the input cycle. This value is
        unitless and should be multiplied by 'kT'.
    parsed_thermo_force_str : SymPy function
        The thermodynamic force equation in SymPy function form. Should be
        multiplied by 'kT' to get actual thermodynamic force.
    """
    cycle_count = 0
    for cyc in graphs.find_all_unique_cycles(G):
        if sorted(cycle) == sorted(cyc):
            cycle_count += 1
    if cycle_count > 1:  # for all-node cycles
        CCW_cycle = graphs.get_ccw_cycle(cycle, order)
        cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    elif cycle_count == 1:  # for all other cycles
        ordered_cycle = _get_ordered_cycle(G, cycle)
        CCW_cycle = graphs.get_ccw_cycle(ordered_cycle, order)
        cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    else:
        raise CycleError("Cycle {} could not be found in G.".format(cycle))
    if output_strings == False:
        if isinstance(
            G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str
        ):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        thermo_force = np.log(ccw_rates / cw_rates)
        return thermo_force
    elif output_strings == True:
        if not isinstance(
            G.edges[cycle_edges[0][0], cycle_edges[0][1], cycle_edges[0][2]][key], str
        ):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        thermo_force_str = (
            "ln(" + "*".join(ccw_rates) + ") - ln(" + "*".join(cw_rates) + ")"
        )
        parsed_thermo_force_str = logcombine(parse_expr(thermo_force_str), force=True)
        return parsed_thermo_force_str


def calc_state_probs(G, key, output_strings=False):
    """
    Calculates state probabilities directly.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the state
        probabilities using numbers. If True, this will assume the input
        'key' will return strings of variable names to join into the
        analytic state multplicity and normalization function.

    Returns
    -------
    state_probs : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    state_probs_sympy : SymPy object
        List of analytic SymPy state probability functions.
    """
    dir_pars = diagrams.generate_directional_partial_diagrams(G)
    if output_strings == False:
        state_probs = calc_state_probs_from_diags(
            G, dir_pars, key, output_strings=output_strings
        )
        return state_probs
    if output_strings == True:
        state_mults, norm = calc_state_probs_from_diags(
            G, dir_pars, key, output_strings=output_strings
        )
        state_probs_sympy = expressions.construct_sympy_prob_funcs(state_mults, norm)
        return state_probs_sympy


def calc_net_cycle_flux(G, cycle, order, key, output_strings=False):
    """
    Calculates net cycle flux for a given cycle in diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram.
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    key : str
        Definition of key in NetworkX diagram edges, used to call edge rate
        values or names. This needs to match the key used for the rate
        constants names or values in the input diagram G.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is False, which tells the function to calculate the cycle flux using
        numbers. If True, this will assume the input 'key' will return strings
        of variable names to join into the analytic cycle flux function.

    Returns
    -------
    net_cycle_flux : float
        Net cycle flux for input cycle.
    net_cycle_flux_func : SymPy object
        Analytic net cycle flux SymPy function.
    """
    dir_pars = diagrams.generate_directional_partial_diagrams(G)
    flux_diags = diagrams.generate_flux_diagrams(G, cycle)
    if output_strings == False:
        pi_diff = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K = calc_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma = calc_sigma(G, dir_pars, key, output_strings=output_strings)
        net_cycle_flux = pi_diff * sigma_K / sigma
        return net_cycle_flux
    if output_strings == True:
        pi_diff_str = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K_str = calc_sigma_K(
            G, cycle, flux_diags, key, output_strings=output_strings
        )
        sigma_str = calc_sigma(G, dir_pars, key, output_strings=output_strings)
        sympy_net_cycle_flux_func = expressions.construct_sympy_net_cycle_flux_func(
            pi_diff_str, sigma_K_str, sigma_str
        )
        return sympy_net_cycle_flux_func
