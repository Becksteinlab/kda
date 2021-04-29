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
.. autofunction:: calc_state_probs_from_diags
.. autofunction:: calc_sigma
.. autofunction:: calc_sigma_K
.. autofunction:: calc_pi_difference
.. autofunction:: calc_net_cycle_flux
.. autofunction:: calc_net_cycle_flux_from_diags
.. autofunction:: calc_thermo_force
"""

import math
import numpy as np
import networkx as nx
from sympy import parse_expr, logcombine

from kda import graph_utils, diagrams, expressions
from kda.exceptions import CycleError


def _get_ordered_cycle(G, input_cycle):
    """
    Takes in arbitrary list of nodes and returns list of nodes in correct order.
    Can be used in conjunction with `diagrams._construct_cycle_edges` to
    generate list of edge tuples for an arbitrary input cycle. Assumes input
    cycle only exists once in the input diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    input_cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.

    Returns
    -------
    ordered_cycles : list of int
        Ordered list of nodes for the input cycle
    """
    possible_nodes = G.nodes()
    for i in input_cycle:
        if i not in possible_nodes:
            raise CycleError(f"Input cycle contains nodes not within input diagram.")

    # get the unique cycles
    unique_cycles = graph_utils.find_all_unique_cycles(G)
    # filter out any cycles that don't contain the correct number of nodes
    filtered_cycles = [c for c in unique_cycles if len(c) == len(input_cycle)]

    if len(filtered_cycles) == 1:
        # if only 1 cycle is left after filtering, this is the ordered cycle
        return filtered_cycles[0]

    ordered_cycles = []
    for cycle in filtered_cycles:
        if sorted(input_cycle) == sorted(cycle):
            ordered_cycles.append(cycle)

    if ordered_cycles == []:
        raise CycleError(f"No cycles found for cycle: {input_cycle}")

    if len(ordered_cycles) == 1:
        return ordered_cycles[0]

    elif len(ordered_cycles) > 1:
        # if multiple cycles are found, we must have multiple cycles that
        # contain the nodes in the input cycle
        # for this case, we just need to check that the input cycle is
        # identical to one of the ordered cycles found
        if input_cycle in ordered_cycles:
            # if the input cycle matches one of the ordered cycles
            # then we can simply return the input cycle since it is
            # already ordered
            return input_cycle
        else:
            # if the input cycle doesn't match any of the ordered cycles,
            # raise an error
            raise CycleError(
                f"Distinct ordered cycle could not be determined. Input diagram"
                f" has multiple unique cycles that contain all nodes in the"
                f" input cycle ({input_cycle}). To fix ambiguity, please input a"
                f" cycle with the nodes in the correct orientation. Select"
                f" one of the following possibilities: {ordered_cycles}"
            )


def calc_sigma(G, dirpar_edges, key, output_strings=False):
    """
    Calculates sigma, the normalization factor for calculating state
    probabilities and cycle fluxes for a given diagram G.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    dirpar_edges : list
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
    # Number of nodes/states
    n_states = G.number_of_nodes()
    n_dirpars = dirpar_edges.shape[0]
    edge_value = G.edges[list(G.edges)[0]][key]

    if not output_strings:
        if isinstance(edge_value, str):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        dirpar_rate_products = np.ones(n_dirpars, dtype=float)
        # iterate over the directional partial diagrams
        for i, edge_list in enumerate(dirpar_edges):
            # iterate over the edges in the given directional partial diagram i
            for edge in edge_list:
                # multiply the rate of each edge in edge_list
                dirpar_rate_products[i] *= G.edges[edge][key]
        sigma = math.fsum(dirpar_rate_products)
        return sigma
    elif output_strings:
        if not isinstance(edge_value, str):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        dirpar_rate_products = np.empty(shape=(n_dirpars,), dtype=object)
        # iterate over the directional partial diagrams
        for i, edge_list in enumerate(dirpar_edges):
            rate_product_vals = []
            for edge in edge_list:
                # append rate constant names from dir_par to list
                rate_product_vals.append(G.edges[edge][key])
            dirpar_rate_products[i] = "*".join(rate_product_vals)
        # sum all terms to get normalization factor
        sigma_str = "+".join(dirpar_rate_products)
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
        # check that the input cycle is in the correct order
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
            sigma_K = math.fsum(rate_products)
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
    # check that the input cycle is in the correct order
    ordered_cycle = _get_ordered_cycle(G, cycle)
    CCW_cycle = graph_utils.get_ccw_cycle(ordered_cycle, order)
    cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
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
    # check that the input cycle is in the correct order
    ordered_cycle = _get_ordered_cycle(G, cycle)
    CCW_cycle = graph_utils.get_ccw_cycle(ordered_cycle, order)
    cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
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
    dirpar_edges = diagrams.generate_directional_partial_diagrams(G, return_edges=True)
    if output_strings == False:
        state_probs = calc_state_probs_from_diags(
            G, dirpar_edges, key, output_strings=output_strings
        )
        return state_probs
    if output_strings == True:
        state_mults, norm = calc_state_probs_from_diags(
            G, dirpar_edges, key, output_strings=output_strings
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
    dirpar_edges = diagrams.generate_directional_partial_diagrams(G, return_edges=True)
    flux_diags = diagrams.generate_flux_diagrams(G, cycle)
    if output_strings == False:
        pi_diff = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K = calc_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma = calc_sigma(G, dirpar_edges, key, output_strings=output_strings)
        net_cycle_flux = pi_diff * sigma_K / sigma
        return net_cycle_flux
    if output_strings == True:
        pi_diff_str = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K_str = calc_sigma_K(
            G, cycle, flux_diags, key, output_strings=output_strings
        )
        sigma_str = calc_sigma(G, dirpar_edges, key, output_strings=output_strings)
        sympy_net_cycle_flux_func = expressions.construct_sympy_net_cycle_flux_func(
            pi_diff_str, sigma_K_str, sigma_str
        )
        return sympy_net_cycle_flux_func


def calc_state_probs_from_diags(G, dirpar_edges, key, output_strings=False):
    """
    Calculates state probabilities and generates analytic function strings from
    input diagram and directional partial diagrams. If directional partial
    diagrams are already generated, this offers faster calculation than
    `calc_state_probs`.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dirpar_edges : array
        Array of all directional partial diagram edges (made from 2-tuples).
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
    # get the number of nodes/states
    n_states = G.number_of_nodes()
    # get the number of directional partial diagrams
    n_dirpars = dirpar_edges.shape[0]
    # get the number of partial diagrams
    n_partials = int(n_dirpars / n_states)

    edge_value = G.edges[list(G.edges)[0]][key]
    if not output_strings:
        if isinstance(edge_value, str):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        # create array of ones for storing rate products
        dirpar_rate_products = np.ones(n_dirpars, dtype=float)
        # iterate over the directional partial diagrams
        for i, edge_list in enumerate(dirpar_edges):
            # iterate over the edges in the given directional partial diagram i
            for edge in edge_list:
                # multiply the rate of each edge
                dirpar_rate_products[i] *= G.edges[edge][key]

        state_mults = dirpar_rate_products.reshape(n_states, n_partials).sum(axis=1)
        state_probs = state_mults / math.fsum(dirpar_rate_products)
        if any(elem < 0 for elem in state_probs):
            raise ValueError(
                "Calculated negative state probabilities, overflow or underflow occurred."
            )
        return state_probs
    elif output_strings:
        if not isinstance(edge_value, str):
            raise TypeError(
                "To enter variable values set parameter output_strings=False."
            )
        dirpar_rate_products = np.empty(shape=(n_dirpars,), dtype=object)
        for i, edge_list in enumerate(dirpar_edges):
            rate_product_vals = []
            for edge in edge_list:
                rate_product_vals.append(G.edges[edge][key])
            dirpar_rate_products[i] = "*".join(rate_product_vals)

        state_mults = np.empty(shape=(n_states,), dtype=object)
        dirpar_rate_products = dirpar_rate_products.reshape(n_states, n_partials)
        for i, arr in enumerate(dirpar_rate_products):
            state_mults[i] = "+".join(arr)
        # sum all terms to get normalization factor
        norm = "+".join(state_mults)
        return state_mults, norm


def calc_net_cycle_flux_from_diags(G, dirpars, cycle, order, key, output_strings=False):
    """
    Calculates net cycle flux and generates analytic function strings from
    input diagram and directional partial diagrams. If directional partial
    diagrams are already generated, this offers faster calculation than
    `calc_net_cycle_flux`.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram.
    dirpars : list
        List of all directional partial diagrams for a given set of partial
        diagrams.
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
    flux_diags = diagrams.generate_flux_diagrams(G, cycle)
    if output_strings == False:
        pi_diff = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K = calc_sigma_K(G, cycle, flux_diags, key, output_strings=output_strings)
        sigma = calc_sigma(G, dirpars, key, output_strings=output_strings)
        net_cycle_flux = pi_diff * sigma_K / sigma
        return net_cycle_flux
    if output_strings == True:
        pi_diff_str = calc_pi_difference(
            G, cycle, order, key, output_strings=output_strings
        )
        sigma_K_str = calc_sigma_K(
            G, cycle, flux_diags, key, output_strings=output_strings
        )
        sigma_str = calc_sigma(G, dirpars, key, output_strings=output_strings)
        sympy_net_cycle_flux_func = expressions.construct_sympy_net_cycle_flux_func(
            pi_diff_str, sigma_K_str, sigma_str
        )
        return sympy_net_cycle_flux_func
