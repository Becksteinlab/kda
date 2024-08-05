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
from sympy import logcombine
from sympy.parsing.sympy_parser import parse_expr

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
        if list(input_cycle) in ordered_cycles:
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


def calc_sigma(G, dirpar_edges, key="name", output_strings=True):
    """
    Generates the normalization factor expression for state
    probabilities and cycle fluxes, which is the sum of directional
    diagrams for the kinetic diagram `G` [Hill1989]_.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram
    dirpar_edges : list
        List of all directional diagrams for the input diagram G.
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
        Sum of rate products of all directional diagrams for input
        diagram G, in string form.

    Notes
    -----
    The expression generated here is important for normalizing
    both state probabilities and net cycle fluxes.
    State probabilities are defined [Hill1989]_,

    .. math::

        p_i = \Omega_{i} / \Sigma,

    where :math:`\Omega_{i}` is the state multiplicity for state
    :math:`i` (the sum of directional diagrams for state :math:`i`)
    and :math:`\Sigma` is the sum of all directional diagrams.

    Additionally :math:`\Sigma` is used when calculating the net
    cycle flux for some cycle :math:`k` [Hill1989]_,

    .. math::

        J_{k} = (\Pi_{+} - \Pi_{-}) \Sigma_{k} / \Sigma,

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.

    References
    ----------
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
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
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dirpar_edges):
            # iterate over the edges in the given directional diagram i
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
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dirpar_edges):
            rate_product_vals = []
            for edge in edge_list:
                # append rate constant names from dir_par to list
                rate_product_vals.append(G.edges[edge][key])
            dirpar_rate_products[i] = "*".join(rate_product_vals)
        # sum all terms to get normalization factor
        sigma_str = "+".join(dirpar_rate_products)
        return sigma_str


def calc_sigma_K(G, cycle, flux_diags, key="name", output_strings=True):
    """
    Generates the expression for the path-based componenet of the
    sum of flux diagrams for some `cycle` in kinetic diagram `G`.
    The sum of flux diagrams is used in calculating net
    cycle fluxes [Hill1989]_.

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

    Notes
    -----
    The expression generated here is important for generating
    the net cycle flux expressions. The net cycle flux for some
    cycle :math:`k` is [Hill1989]_,

    .. math::

        J_{k} = (\Pi_{+} - \Pi_{-}) \Sigma_{k} / \Sigma,

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Sigma_{k}` is the path-based component of the flux diagram
    sum. For cycles with no flux diagrams, :math:`\Sigma_{k} = 1`.

    References
    ----------
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
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


def calc_pi_difference(G, cycle, order, key="name", output_strings=True):
    """
    Generates the expression for the cycle-based componenet of the
    sum of flux diagrams for some `cycle` in kinetic diagram `G`.
    The sum of flux diagrams is used in calculating net
    cycle fluxes [Hill1989]_.

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

    Notes
    -----
    The expression generated here is important for generating
    the net cycle flux expressions. The net cycle flux for some
    cycle :math:`k` is [Hill1989]_,

    .. math::

        J_{k} = (\Pi_{+} - \Pi_{-}) \Sigma_{k} / \Sigma,

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Pi_{+} - \Pi_{-}` is the cycle-based component of the flux
    diagram sum, where :math:`\Pi_{+}` and :math:`\Pi_{-}` are the
    forward and reverse rate-products along cycle :math:`k` and
    the forward (i.e. positive) direction is counter-clockwise (CCW).

    References
    ----------
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
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


def calc_thermo_force(G, cycle, order, key="name", output_strings=True):
    """
    Generates the expression for the thermodynamic driving force
    for some `cycle` in the kinetic diagram `G`.

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

    Notes
    -----
    The expression generated here is used to calculate the thermodynamic
    driving force. The thermodynamic driving force for some cycle
    :math:`k` is [Hill1989]_,

    .. math::

        \chi_{k} = kT \ln \left( \frac{\Pi_{+}}{\Pi_{-}} \right),

    where :math:`\Pi_{+}` and :math:`\Pi_{-}` are the forward
    and reverse rate-products along cycle :math:`k` and the forward
    (i.e. positive) direction is counter-clockwise (CCW). The
    returned expression does not include :math:`kT`. At equilibrium
    the thermodynamic driving force for any cycle is zero
    (i.e. :math:`\chi_{k} = 0`).

    References
    ----------
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
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


def calc_state_probs(G, key="name", output_strings=True):
    """
    Generates the state probability expressions using the diagram method
    developed by King and Altman [King1956]_ and Hill [Hill1989]_.

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

    Notes
    -----
    The algebraic expressions for the state probabilities (at steady-state)
    are created directly from the directional diagrams. Subsets of the
    directional diagrams are summed based on a common target state. For some
    state :math:`i` the sum of directional diagrams (i.e., rate-products)
    with target state :math:`i` yields the unnormalized state probability
    expression [Hill1989]_,

    .. math::

        \Omega_{i} = \sum \text{directional diagrams for state }i.

    The state probabilities are

    .. math::

        p_{i} = \frac{\Omega_i}{\Sigma},

    where :math:`\Sigma` is the sum of all directional diagrams (i.e. all
    :math:`\Omega_i`s) for the kinetic diagram.

    References
    ----------
    .. [King1956] E. L. King, C. Altman (1956). "A Schematic Method of Deriving
        the Rate Laws for Enzyme-Catalyzed Reactions." The Journal of Physical
        Chemistry 1956, 60, 1375–1378.
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
    """
    dirpar_edges = diagrams.generate_directional_diagrams(G, return_edges=True)
    state_probs = calc_state_probs_from_diags(
        G, dirpar_edges=dirpar_edges, key=key, output_strings=output_strings,
    )
    if output_strings:
        state_probs = expressions.construct_sympy_prob_funcs(state_mult_funcs=state_probs)
    return state_probs


def calc_net_cycle_flux(G, cycle, order, key="name", output_strings=True):
    """
    Generates the expression for the net cycle flux for some `cycle`
    in kinetic diagram `G`.

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

    Notes
    -----
    The net cycle flux for some cycle :math:`k` is [Hill1989]_,

    .. math::

        J_{k} = (\Pi_{+} - \Pi_{-}) \Sigma_{k} / \Sigma,

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Pi_{+}` and :math:`\Pi_{-}` are the forward and reverse
    rate-products along cycle :math:`k` where the forward
    (i.e. positive) direction is counter-clockwise (CCW).

    References
    ----------
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
    """
    dirpar_edges = diagrams.generate_directional_diagrams(G, return_edges=True)
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


def calc_state_probs_from_diags(G, dirpar_edges, key="name", output_strings=True):
    """
    Generates the state probability expressions using the diagram method
    developed by King and Altman [King1956]_ and Hill [Hill1989]_. If
    directional diagram edges are already generated this offers better
    performance than :func:`calc_state_probs`.

    Parameters
    ----------
    G : NetworkX MultiDiGraph
        Input diagram
    dirpar_edges : array
        Array of all directional diagram edges (made from 2-tuples).
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

    References
    ----------
    .. [King1956] E. L. King, C. Altman (1956). "A Schematic Method of Deriving
        the Rate Laws for Enzyme-Catalyzed Reactions." The Journal of Physical
        Chemistry 1956, 60, 1375–1378.
    .. [Hill1989] T. L. Hill (1989). "Free Energy Transduction
        and Biochemical Cycle Kinetics." Springer-Verlag.
    """
    # get the number of nodes/states
    n_states = G.number_of_nodes()
    # get the number of directional diagrams
    n_dirpars = dirpar_edges.shape[0]
    # get the number of partial diagrams
    n_partials = int(n_dirpars / n_states)
    # retrieve the rate matrix from G
    Kij = graph_utils.retrieve_rate_matrix(G)

    edge_value = G.edges[list(G.edges)[0]][key]
    if not output_strings:
        if isinstance(edge_value, str):
            raise TypeError(
                "To enter variable strings set parameter output_strings=True."
            )
        # create array of ones for storing rate products
        dirpar_rate_products = np.ones(n_dirpars, dtype=float)
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dirpar_edges):
            # for each edge list, retrieve an array of the ith and jth indices,
            # retrieve the values associated with each (i, j) pair, and
            # calculate the product of those values
            Ki = edge_list[:, 0]
            Kj = edge_list[:, 1]
            dirpar_rate_products[i] = np.prod(Kij[Ki, Kj])

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
        return state_mults


def calc_net_cycle_flux_from_diags(
    G, dirpar_edges, cycle, order, key="name", output_strings=True
):
    """
    Generates the expression for the net cycle flux for some `cycle`
    in kinetic diagram `G`. If directional diagram edges are already
    generated this offers better performance than :func:`calc_net_cycle_flux`.

    Parameters
    ----------
    G : NetworkX MultiDiGraph Object
        Input diagram.
    dirpar_edges : array
        Array of all directional diagram edges (made from 2-tuples).
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
