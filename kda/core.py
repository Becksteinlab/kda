# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
KDA Core Objects
======================================================================
The :mod:`~kda.core` module contains code to build the core
:class:`~kda.core.KineticModel` object which contains all system information
for a kinetic model.

.. autoclass:: KineticModel
   :members:

References
==========
.. footbibliography::
"""

import networkx as nx
from sympy import symbols, Mul

from kda import graph_utils, diagrams, calculations


class KineticModel(object):
	"""The KDA KineticModel contains all the information describing the system.

	Includes methods to construct the kinetic diagram, generate the
	intermediate graphs, and build the algebraic expressions for
	steady-state probabilities and fluxes.

	Attributes
	==========
	cycles : list of lists of int
		All cycles in the kinetic diagram. This attribute becomes
		available after running the
		:meth:`~kda.core.KineticModel.build_cycles` method.
	partial_diagrams : array of Networkx Graphs
		The set of partial diagrams (i.e. spanning trees) for the
		kinetic diagram. This attribute becomes available after
		running the :meth:`~kda.core.KineticModel.build_partial_diagrams`
		method.
	directional_diagrams : array of Networkx MultiDiGraphs
		The set of directional diagrams for the kinetic diagram.
		This attribute becomes available after running the
		:meth:`~kda.core.KineticModel.build_directional_diagrams` method.
	flux_diagrams : list of lists of Networkx MultiDiGraphs
		The set of flux diagrams for each cycle in the kinetic
		diagram. This attribute becomes available after running
		the :meth:`~kda.core.KineticModel.build_flux_diagrams` method.
	probabilities : array of floats or list of SymPy expressions
		The steady-state probabilities for all states in the kinetic
		diagram. Probabilities are either an array of numeric values
		or the algebraic expressions. This attribute becomes available
		after running the
		:meth:`~kda.core.KineticModel.build_state_probabilities` method.

	"""

	def __init__(self, K=None, G=None):
		"""
		Parameters
		==========
		K : ndarray (optional)
			``NxN`` array where ``N`` is the number of nodes in the
			diagram ``G``. Adjacency matrix for ``G`` where each element
			``kij`` is the edge weight (i.e. transition rate constant).
			For example, for a 2-state model with ``k12=3`` and ``k21=4``,
			``K=[[0, 3], [4, 0]]``. Default is ``None``.
		G : NetworkX MultiDiGraph (optional)
			Input diagram. Default is ``None``.

		Raises
		======
		RuntimeError
			If both ``K`` and ``G`` are ``None``.
		"""
		if G is None or K is None:
			if K is not None:
				# if only K is input create the diagram
				G = nx.MultiDiGraph()
				graph_utils.generate_edges(G=G, vals=K)
			elif G is not None:
				# if only G is input create the kinetic rate matrix
				K = graph_utils.retrieve_rate_matrix(G)
			else:
				msg = "To create a `KineticModel`, K or G must be input."
				raise RuntimeError(msg)

		self.K = K
		self.G = G
		# assign future attributes
		self.cycles = None
		self.partial_diagrams = None
		self.directional_diagrams = None
		self.flux_diagrams = None
		self.probabilities = None


	def build_cycles(self):
		"""Builds all cycles from the kinetic diagram using
		:meth:`~kda.graph_utils.find_all_unique_cycles()`.
		"""
		self.cycles = graph_utils.find_all_unique_cycles(self.G)


	def build_partial_diagrams(self):
		"""Builds the partial diagrams for the kinetic diagram using
		:meth:`~kda.diagrams.generate_partial_diagrams()`.
		"""
		self.partial_diagrams = diagrams.generate_partial_diagrams(
			self.G, return_edges=False)


	def build_directional_diagrams(self):
		"""Builds the directional diagrams for the kinetic diagram using
		:meth:`~kda.diagrams.generate_directional_diagrams()`.
		"""
		self.directional_diagrams = diagrams.generate_directional_diagrams(
			self.G, return_edges=False)


	def build_flux_diagrams(self):
		"""Builds the flux diagrams for the kinetic diagram using
		:meth:`~kda.diagrams.generate_all_flux_diagrams()`.
		"""
		self.flux_diagrams = diagrams.generate_all_flux_diagrams(self.G)


	def get_partial_diagram_count(self):
		"""
		Returns the number of partial diagrams that will
		be created from the kinetic diagram. If partial diagrams
		have already been generated with
		:meth:`~kda.core.KineticModel.build_partial_diagrams()`
		the count will simply be returned. Otherwise
		:meth:`~kda.diagrams.enumerate_partial_diagrams()` is used.

		Returns
		=======
		The integer number of partial diagrams.
		"""
		if self.partial_diagrams is not None:
			return len(self.partial_diagrams)
		else:
			return diagrams.enumerate_partial_diagrams(self.G)


	def get_directional_diagram_count(self):
		"""
		Returns the number of directional diagrams that will
		be created from the kinetic diagram. If directional diagrams
		have already been generated with
		:meth:`~kda.core.KineticModel.build_directional_diagrams()`
		the count will simply be returned. Otherwise
		:meth:`~kda.core.KineticModel.get_partial_diagram_count()`
		is used (there are ``N`` directional diagrams per partial
		diagram for a kinetic diagram with ``N`` states).

		Returns
		=======
		The integer number of directional diagrams.
		"""
		if self.directional_diagrams is not None:
			return len(self.directional_diagrams)
		else:
			partial_count = self.get_partial_diagram_count()
			return self.G.number_of_nodes() * partial_count


	def get_flux_diagrams(self, cycle):
		"""
		Retrieves the flux diagrams for a specific cycle using
		:meth:`~kda.diagrams.generate_flux_diagrams()`.

		Parameters
		==========
		cycle : list of int
			List of node indices for cycle of interest, index zero.
			Order of node indices does not matter.

		Returns
		=======
		The flux diagrams associated with the input cycle.
		"""
		# TODO: see if we can check if flux diagrams have been
		# generated using `build_flux_diagrams`, then retrieve
		# the correct diagrams based on `cycle`
		return diagrams.generate_flux_diagrams(self.G, cycle)


	def build_state_probabilities(self, symbolic=True):
		"""
		Builds the state probabilities for the kinetic diagram using
		:meth:`~kda.calculations.calc_state_probs()`. Probabilities
		can be stored as raw values or symbolic algebraic expressions.

		Parameters
		==========
		symbolic : bool (optional)
			Used to determine whether raw values or symbolic
			expressions will be stored. Default is ``True``.
		"""
		# TODO: may be able to leverage `calc_state_probs_from_diags()`
		# here, but it would require the user has already generated the
		# directional diagrams as edges, which is probably atypical
		# NOTE: currently hacking in the `key` parameters here
		# assuming the edge attributes follow the name/val convention.
		# eventually all calculation functions should be sophisticated
		# enough where only `symbolic=True` is sufficient
		if symbolic:
			key = "name"
		else:
			key = "val"
		self.probabilities = calculations.calc_state_probs(
			self.G, key=key, output_strings=symbolic)


	def get_transition_flux(self, state_i, state_j, net=True, symbolic=True):
		r"""
		Generates the expressions for the one-way or net
		transition fluxes between two states.

		Parameters
		==========
		state_i : integer
			The index (index 1) of the initial state.
		state_j : integer
			The index (index 1) of the final state.
		net : bool (optional)
			Used to determine whether one-way transition fluxes
			or net transition fluxes will be returned. Default
			is ``True``.
		symbolic : bool (optional)
			Used to determine whether raw values or symbolic
			expressions will be returned. Default is ``True``.

		Returns
		=======
		The transition flux (one-way or net) from state ``i`` to state ``j``.

		Raises
		======
		ValueError
			If the input states are the same (i.e. ``i==j``).
		TypeError
			If the stored state probability type (symbolic or numeric)
			is a differnt type than the requested transition
			flux type.

		Notes
		-----
		The expressions generated here quantify the one-way or
		net probability flows between two states. The net transition flux
		between two states is defined :footcite:`hill_free_1989`,

		.. math::

			J_{ij} = j_{ij} - j_{ji},

		where :math:`j_{ij} = k_{ij} p_{i}` and :math:`j_{ji} = k_{ji} p_{j}`
		are the one-way transition fluxes. For the one-way fluxes,
		:math:`k_{ij}` is the kinetic rate from state :math:`i`
		to state :math:`j` and :math:`p_{i}` is the state probability
		for state :math:`i`.

		"""
		if state_i == state_j:
			msg = "Input indices must be unique (i.e. i != j)."
			raise ValueError(msg)

		if self.probabilities is None:
			print(
				f"No probabilities found, generating state"
				f" probabilities with symbolic={symbolic}")
			self.build_state_probabilities(symbolic=symbolic)
		else:
			# check if stored probability data type matches the
			# requested transition flux type (numeric vs symbolic)
			is_symbolic = isinstance(self.probabilities[0], Mul)
			if symbolic != is_symbolic:
				msg = (
					f"`KineticModel.probabilities` are the incorrect"
					f" type for the requested transition flux type."
					f" Regenerate probabilities with `symbolic={symbolic}`"
					f" before continuing."
				)
				raise TypeError(msg)

		if symbolic:
			# NOTE: symbols are index 1, probabilities are index 0
			j_ij = symbols(f"k{state_i}{state_j}") * self.probabilities[state_i-1]
			if net:
				j_ji = symbols(f"k{state_j}{state_i}") * self.probabilities[state_j-1]
				return (j_ij - j_ji).cancel()
			else:
				return j_ij.cancel()

		else:
			# numerical case
			j_ij = self.K[state_i-1][state_j-1] * self.probabilities[state_i-1]
			if net:
				j_ji = self.K[state_j-1][state_i-1] * self.probabilities[state_j-1]
				return j_ij - j_ji
			else:
				return j_ij

