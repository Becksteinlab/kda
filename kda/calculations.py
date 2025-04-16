# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
State Probability & Flux Calculations
=====================================
The :mod:`~kda.calculations` module contains code to calculate steady-state
probabilities and fluxes from a user-defined kinetic diagram.

.. autofunction:: calc_state_probs
.. autofunction:: calc_cycle_flux
.. autofunction:: calc_sigma
.. autofunction:: calc_sigma_K
.. autofunction:: calc_pi_difference
.. autofunction:: calc_thermo_force

References
==========
.. footbibliography::

"""
import warnings
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
    Can be used in conjunction with ``diagrams._construct_cycle_edges`` to
    generate list of edge tuples for an arbitrary input cycle. Assumes input
    cycle only exists once in the input diagram G.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
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
    r"""
    Generates the normalization factor expression for state
    probabilities and cycle fluxes, which is the sum of directional
    diagrams for the kinetic diagram `G` :footcite:`hill_free_1989`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    dirpar_edges : ndarray
        Array of all directional diagram edges (made from 2-tuples)
        for the input diagram ``G``. Created using
        :meth:`~kda.diagrams.generate_directional_diagrams`
        with ``return_edges=True``.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the normalization
        factor using numbers. If ``True``, this will assume the input
        ``'key'`` will return strings of variable names to join into the
        analytic cycle flux function.

    Returns
    -------
    sigma : float or str
        Sum of rate products of all directional diagrams for the input
        diagram ``G`` as a float (``output_strings=False``) or a string
        (``output_strings=True``).

    Notes
    -----
    The expression generated here is important for normalizing
    both state probabilities and net cycle fluxes.
    State probabilities are defined :footcite:`hill_free_1989`,

    .. math::

        p_i = \Omega_{i} / \Sigma,

    where :math:`\Omega_{i}` is the state multiplicity for state
    :math:`i` (the sum of directional diagrams for state :math:`i`)
    and :math:`\Sigma` is the sum of all directional diagrams.

    Additionally :math:`\Sigma` is used when calculating the net
    cycle flux for some cycle :math:`k` :footcite:`hill_free_1989`,

    .. math::

        J_{k} = \frac{(\Pi_{+} - \Pi_{-}) \Sigma_{k}}{\Sigma},

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.

    """
    edge_is_str = isinstance(G.edges[list(G.edges)[0]][key], str)
    if output_strings != edge_is_str:
        msg = f"""Inputs `key={key}` and `output_strings={output_strings}`
            do not match. If symbolic outputs are requested the input `key`
            should retrieve edge data from `G` that corresponds to symbolic
            variable names for all edges."""
        raise TypeError(msg)

    n_dir_diagrams = dirpar_edges.shape[0]
    if output_strings:
        rate_products = np.empty(shape=(n_dir_diagrams,), dtype=object)
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dirpar_edges):
            rates = [G.edges[edge][key] for edge in edge_list]
            rate_products[i] = "*".join(rates)
        # sum all terms to get normalization factor
        sigma = "+".join(rate_products)
    else:
        rate_products = np.ones(n_dir_diagrams, dtype=float)
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dirpar_edges):
            # iterate over the edges in the given directional diagram i
            for edge in edge_list:
                # multiply the rate of each edge in edge_list
                rate_products[i] *= G.edges[edge][key]
        sigma = math.fsum(rate_products)
    return sigma


def calc_sigma_K(G, cycle, flux_diags, key="name", output_strings=True):
    r"""
    Generates the expression for the path-based componenet of the
    sum of flux diagrams for some ``cycle`` in kinetic diagram ``G``.
    The sum of flux diagrams is used in calculating net
    cycle fluxes :footcite:`hill_free_1989`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter but should not contain all nodes.
    flux_diags : list
        List of relevant directional flux diagrams for input cycle.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
        Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the sum of all
        directional flux diagrams using numbers. If ``True``, this will assume
        the input ``'key'`` will return strings of variable names to join into
        the analytic function.

    Returns
    -------
    sigma_K : float or str
        Sum of rate products of directional flux diagram edges pointing to
        input cycle as a float (``output_strings=False``) or as a string
        (``output_strings=True``).

    Notes
    -----
    The expression generated here is important for generating
    the net cycle flux expressions. The net cycle flux for some
    cycle :math:`k` is :footcite:`hill_free_1989`,

    .. math::

        J_{k} = \frac{(\Pi_{+} - \Pi_{-}) \Sigma_{k}}{\Sigma},

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Sigma_{k}` is the path-based component of the flux diagram
    sum. For cycles with no flux diagrams, :math:`\Sigma_{k} = 1`.

    """
    if not isinstance(flux_diags, list):
        print(f"No flux diagrams detected for cycle {cycle}. Sigma K value is 1.")
        return 1
    edge_is_str = isinstance(G.edges[list(G.edges)[0]][key], str)
    if output_strings != edge_is_str:
        msg = f"""Inputs `key={key}` and `output_strings={output_strings}`
            do not match. If symbolic outputs are requested the input `key`
            should retrieve edge data from `G` that corresponds to symbolic
            variable names for all edges."""
        raise TypeError(msg)

    # check that the input cycle is in the correct order
    ordered_cycle = _get_ordered_cycle(G, cycle)
    cycle_edges = diagrams._construct_cycle_edges(ordered_cycle)
    if output_strings:
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
        sigma_K = "+".join(rate_products)
    else:
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


def calc_pi_difference(G, cycle, order, key="name",
        output_strings=True, net=True):
    r"""
    Generates the expression for the cycle-based componenet of the
    sum of flux diagrams for some ``cycle`` in kinetic diagram ``G``.
    The sum of flux diagrams is used in calculating net
    cycle fluxes :footcite:`hill_free_1989`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2 (e.g. ``[0, 1]``), where the integers are
        nodes in ``cycle``. The pair of nodes should be ordered such that
        a counter-clockwise path is followed.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
        Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the difference
        using numbers. If ``True``, this will assume the input ``'key'``
        will return strings of variable names to join into the analytic
        function.
    net : bool (optional)
        Used to determine whether to return the _forward_ cycle product
        (i.e., ``net=False``) or the _difference_ of the forward and reverse
        cycle products (i.e., ``net=True``). Default is ``True``.

    Returns
    -------
    pi_difference : float or str
        Difference of the counter-clockwise and clockwise cycle
        rate-products as a float (``output_strings=False``)
        or a string (``output_strings=True``).

    Notes
    -----
    The expression generated here is important for generating
    the net cycle flux expressions. The net cycle flux for some
    cycle :math:`k` is :footcite:`hill_free_1989`,

    .. math::

        J_{k} = \frac{(\Pi_{+} - \Pi_{-}) \Sigma_{k}}{\Sigma},

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Pi_{+} - \Pi_{-}` is the cycle-based component of the flux
    diagram sum, where :math:`\Pi_{+}` and :math:`\Pi_{-}` are the
    forward and reverse rate-products along cycle :math:`k` and
    the forward (i.e. positive) direction is counter-clockwise (CCW).

    """
    edge_is_str = isinstance(G.edges[list(G.edges)[0]][key], str)
    if output_strings != edge_is_str:
        msg = f"""Inputs `key={key}` and `output_strings={output_strings}`
            do not match. If symbolic outputs are requested the input `key`
            should retrieve edge data from `G` that corresponds to symbolic
            variable names for all edges."""
        raise TypeError(msg)

    # check that the input cycle is in the correct order
    ordered_cycle = _get_ordered_cycle(G, cycle)
    CCW_cycle = graph_utils.get_ccw_cycle(ordered_cycle, order)
    cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    if output_strings:
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        if net:
            pi_difference = "-".join(["*".join(ccw_rates), "*".join(cw_rates)])
        else:
            pi_difference = "*".join(ccw_rates)
    else:
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        if net:
            pi_difference = ccw_rates - cw_rates
        else:
            pi_difference = ccw_rates
    return pi_difference


def calc_thermo_force(G, cycle, order, key="name", output_strings=True):
    r"""
    Generates the expression for the thermodynamic driving force
    for some ``cycle`` in the kinetic diagram ``G``.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter unless your cycle contains all nodes.
    order : list of int
        List of integers of length 2 (e.g. ``[0, 1]``), where the integers are
        nodes in ``cycle``. The pair of nodes should be ordered such that
        a counter-clockwise path is followed.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
        Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the thermodynamic
        force using numbers. If ``True``, this will assume the input
        ``'key'`` will return strings of variable names to join into the
        analytic function.

    Returns
    -------
    thermo_force : float or ``SymPy`` expression
        The thermodynamic force for the input ``cycle`` returned
        as a float (``output_strings=False``) or a ``SymPy`` expression
        (``output_strings=True``). The returned value is unitless and
        should be multiplied by ``kT`` to calculate the actual
        thermodynamic force.

    Notes
    -----
    The expression generated here is used to calculate the thermodynamic
    driving force. The thermodynamic driving force for some cycle
    :math:`k` is :footcite:`hill_free_1989`,

    .. math::

        \chi_{k} = kT \ln \left( \frac{\Pi_{+}}{\Pi_{-}} \right),

    where :math:`\Pi_{+}` and :math:`\Pi_{-}` are the forward
    and reverse rate-products along cycle :math:`k` and the forward
    (i.e. positive) direction is counter-clockwise (CCW). The
    returned expression does not include :math:`kT`. At equilibrium
    the thermodynamic driving force for any cycle is zero
    (i.e. :math:`\chi_{k} = 0`).

    """
    edge_is_str = isinstance(G.edges[list(G.edges)[0]][key], str)
    if output_strings != edge_is_str:
        msg = f"""Inputs `key={key}` and `output_strings={output_strings}`
            do not match. If symbolic outputs are requested the input `key`
            should retrieve edge data from `G` that corresponds to symbolic
            variable names for all edges."""
        raise TypeError(msg)

    # check that the input cycle is in the correct order
    ordered_cycle = _get_ordered_cycle(G, cycle)
    CCW_cycle = graph_utils.get_ccw_cycle(ordered_cycle, order)
    cycle_edges = diagrams._construct_cycle_edges(CCW_cycle)
    if output_strings:
        ccw_rates = []
        cw_rates = []
        for edge in cycle_edges:
            ccw_rates.append(G.edges[edge[0], edge[1], edge[2]][key])
            cw_rates.append(G.edges[edge[1], edge[0], edge[2]][key])
        thermo_force = (
            "ln(" + "*".join(ccw_rates) + ") - ln(" + "*".join(cw_rates) + ")"
        )
        thermo_force = logcombine(parse_expr(thermo_force), force=True)
    else:
        ccw_rates = 1
        cw_rates = 1
        for edge in cycle_edges:
            ccw_rates *= G.edges[edge[0], edge[1], edge[2]][key]
            cw_rates *= G.edges[edge[1], edge[0], edge[2]][key]
        thermo_force = np.log(ccw_rates / cw_rates)
    return thermo_force


def calc_state_probs(G, key="name", output_strings=True, dir_edges=None):
    r"""Generates the state probability expressions using the diagram
    method developed by King and Altman :footcite:`king_schematic_1956` and
    Hill :footcite:`hill_studies_1966`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the state
        probabilities using numbers. If ``True``, this will assume the input
        ``'key'`` will return strings of variable names to join into the
        analytic state multplicity and normalization function.
    dir_edges : ndarray (optional)
        Array of all directional diagram edges (made from 2-tuples)
        for the input diagram ``G``. Created using
        :meth:`~kda.diagrams.generate_directional_diagrams`
        with ``return_edges=True``.

    Returns
    -------
    state_probs : ndarray or list of ``SymPy`` expressions
        Array of state probabilities for ``N`` states
        of the form ``[p1, p2, p3, ..., pN]``
        (``output_strings=False``), or a
        list of symbolic state probability expressions
        in the same order (``output_strings=True``).

    Notes
    -----
    The algebraic expressions for the state probabilities (at steady-state)
    are created directly from the directional diagrams. Subsets of the
    directional diagrams are summed based on a common target state. For some
    state :math:`i` the sum of directional diagrams (i.e., rate-products)
    with target state :math:`i` yields the unnormalized state probability
    expression :footcite:`hill_free_1989`,

    .. math::

        \Omega_{i} = \sum \text{directional diagrams for state } i.

    The state probabilities are

    .. math::

        p_{i} = \frac{\Omega_{i}}{\Sigma},

    where :math:`\Sigma` is the sum of all directional diagrams (i.e.
    all :math:`\Omega_i` s) for the kinetic diagram.

    """
    edge_is_str = isinstance(G.edges[list(G.edges)[0]][key], str)
    if output_strings != edge_is_str:
        msg = f"""Inputs `key={key}` and `output_strings={output_strings}`
            do not match. If symbolic outputs are requested the input `key`
            should retrieve edge data from `G` that corresponds to symbolic
            variable names for all edges."""
        raise TypeError(msg)

    if dir_edges is None:
        # generate the directional diagram edges
        dir_edges = diagrams.generate_directional_diagrams(G=G, return_edges=True)
    # get the number of nodes/states
    n_states = G.number_of_nodes()
    # get the number of directional diagrams
    n_dir_diagrams = dir_edges.shape[0]
    # get the number of partial diagrams
    n_partials = int(n_dir_diagrams / n_states)
    if output_strings:
        rate_products = np.empty(shape=(n_dir_diagrams,), dtype=object)
        for i, edge_list in enumerate(dir_edges):
            rates = [G.edges[edge][key] for edge in edge_list]
            rate_products[i] = "*".join(rates)
        rate_products = rate_products.reshape(n_states, n_partials)
        state_mults = np.empty(shape=(n_states,), dtype=object)
        for i, arr in enumerate(rate_products):
            state_mults[i] = "+".join(arr)
        state_probs = expressions.construct_sympy_prob_funcs(state_mult_funcs=state_mults)
    else:
        # retrieve the rate matrix from G
        Kij = graph_utils.retrieve_rate_matrix(G)
        # create array of ones for storing rate products
        rate_products = np.ones(n_dir_diagrams, dtype=float)
        # iterate over the directional diagrams
        for i, edge_list in enumerate(dir_edges):
            # for each edge list, retrieve an array of the ith and jth indices,
            # retrieve the values associated with each (i, j) pair, and
            # calculate the product of those values
            Ki = edge_list[:, 0]
            Kj = edge_list[:, 1]
            rate_products[i] = np.prod(Kij[Ki, Kj])
        state_mults = rate_products.reshape(n_states, n_partials).sum(axis=1)
        state_probs = state_mults / math.fsum(rate_products)
        if any(elem < 0 for elem in state_probs):
            msg = """Calculated negative state probabilities,
                overflow or underflow occurred."""
            raise ValueError(msg)
    return state_probs


def calc_cycle_flux(G, cycle, order, key="name",
        output_strings=True, dir_edges=None, net=True):
    r"""Generates the expression for the one-way or net cycle
    flux for a ``cycle`` in the kinetic diagram ``G``.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    order : list of int
        List of integers of length 2 (e.g. ``[0, 1]``), where the integers are
        nodes in ``cycle``. The pair of nodes should be ordered such that
        a counter-clockwise path is followed.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
        Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the cycle flux
        using numbers. If ``True``, this will assume the input ``'key'``
        will return strings of variable names to join into the analytic
        cycle flux function.
    dir_edges : ndarray (optional)
        Array of all directional diagram edges (made from 2-tuples)
        for the input diagram ``G``. Given as an option for performance reasons
        (when calculating net cycle fluxes for multiple cycles it is best to
        generate the directional diagram edges up front and provide them).
        Created using :meth:`~kda.diagrams.generate_directional_diagrams`
        with ``return_edges=True``.
    net : bool (optional)
        Used to determine whether to return the _one-way_ or _net_ cycle flux.
        Default is ``True`` (i.e., to generate the _net_ cycle flux).

    Returns
    -------
    cycle_flux : float or ``SymPy`` expression
        The one-way or net cycle flux for the input ``cycle``.

    Notes
    -----
    The net cycle flux for some cycle :math:`k` is :footcite:`hill_free_1989`,

    .. math::

        J_{k} = \frac{(\Pi_{+} - \Pi_{-}) \Sigma_{k}}{\Sigma},

    where :math:`(\Pi_{+} - \Pi_{-}) \Sigma_{k}` is the sum of all
    flux diagrams for cycle :math:`k` and :math:`\Sigma` is the sum
    of all directional diagrams for the kinetic diagram.
    :math:`\Pi_{+}` and :math:`\Pi_{-}` are the forward and reverse
    rate-products along cycle :math:`k` where the forward
    (i.e. positive) direction is counter-clockwise (CCW).

    """
    if dir_edges is None:
        # generate the directional diagram edges
        dir_edges = diagrams.generate_directional_diagrams(G=G, return_edges=True)
    # generate the flux diagrams
    flux_diags = diagrams.generate_flux_diagrams(G=G, cycle=cycle)
    # construct the expressions for (Pi+ - Pi-), sigma, and sigma_k
    # from the directional diagram edges
    pi_diff = calc_pi_difference(
        G=G, cycle=cycle, order=order, key=key,
        output_strings=output_strings, net=net)
    sigma_K = calc_sigma_K(
        G=G, cycle=cycle, flux_diags=flux_diags,
        key=key, output_strings=output_strings)
    sigma = calc_sigma(
        G=G, dirpar_edges=dir_edges, key=key, output_strings=output_strings)
    if output_strings:
        cycle_flux = expressions.construct_sympy_net_cycle_flux_func(
            pi_diff_str=pi_diff, sigma_K_str=sigma_K, sigma_str=sigma)
    else:
        cycle_flux = pi_diff * sigma_K / sigma
    return cycle_flux


def calc_state_probs_from_diags(G, dirpar_edges, key="name", output_strings=True):
    """Generates the state probability expressions using the diagram
    method developed by King and Altman :footcite:`king_schematic_1956` and
    Hill :footcite:`hill_studies_1966`. If directional diagram edges are already
    generated this offers better performance than
    :meth:`~kda.calculations.calc_state_probs`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    dirpar_edges : array
        Array of all directional diagram edges (made from 2-tuples)
        for the input diagram ``G``.  Created using
        :meth:`~kda.diagrams.generate_directional_diagrams`
        with ``return_edges=True``.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the state
        probabilities using numbers. If ``True``, this will assume the input
        ``'key'`` will return strings of variable names to join into the
        analytic state multplicity and normalization functions.

    Returns
    -------
    state_probs : ndarray or list of ``SymPy`` expressions
        Array of state probabilities for ``N`` states
        of the form ``[p1, p2, p3, ..., pN]``
        (``output_strings=False``), or a
        list of symbolic state probability expressions
        in the same order (``output_strings=True``).

    """
    msg = """`kda.calculations.calc_state_probs_from_diags` will be deprecated.
        Use `kda.calculations.calc_state_probs` with parameter `dir_edges`."""
    warnings.warn(msg, DeprecationWarning)
    state_probs = calc_state_probs(
        G=G, dir_edges=dirpar_edges, key=key, output_strings=output_strings,
    )
    if output_strings:
        state_probs = expressions.construct_sympy_prob_funcs(state_mult_funcs=state_probs)
    return state_probs


def calc_net_cycle_flux_from_diags(
    G, dirpar_edges, cycle, order, key="name", output_strings=True
):
    """Generates the expression for the net cycle flux for some ``cycle``
    in kinetic diagram ``G``. If directional diagram edges are already
    generated this offers better performance than
    :meth:`~kda.calculations.calc_cycle_flux`.

    Parameters
    ----------
    G : ``NetworkX.MultiDiGraph``
        A kinetic diagram
    dirpar_edges : ndarray
        Array of all directional diagram edges (made from 2-tuples)
        for the input diagram ``G``. Created using
        :meth:`~kda.diagrams.generate_directional_diagrams`
        with ``return_edges=True``.
    cycle : list of int
        List of node indices for cycle of interest, index zero. Order of node
        indices does not matter.
    order : list of int
        List of integers of length 2 (e.g. ``[0, 1]``), where the integers are
        nodes in ``cycle``. The pair of nodes should be ordered such that
        a counter-clockwise path is followed.
    key : str
        Attribute key used to retrieve edge data from ``G.edges``. The default
        ``NetworkX`` edge key is ``"weight"``, however the ``kda`` edge keys
        are ``"name"`` (for rate constant names, e.g. ``"k12"``) and ``"val"``
        (for the rate constant values, e.g. ``100``). Default is ``"name"``.
    output_strings : bool (optional)
        Used to denote whether values or strings will be combined. Default
        is ``False``, which tells the function to calculate the cycle flux
        using numbers. If ``True``, this will assume the input ``'key'``
        will return strings of variable names to join into the analytic
        cycle flux function.

    Returns
    -------
    net_cycle_flux : float or ``SymPy`` expression
        Net cycle flux for the input ``cycle``.

    """
    msg = """`kda.calculations.calc_net_cycle_flux_from_diags` will be deprecated.
        Use `kda.calculations.calc_cycle_flux` with parameter `dir_edges`."""
    warnings.warn(msg, DeprecationWarning)
    return calc_cycle_flux(
        G=G,
        cycle=cycle,
        order=order,
        key=key,
        output_strings=output_strings,
        dir_edges=dirpar_edges,
        net=True,
    )
