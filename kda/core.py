# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Core
=========================================================================
The :class:`~kda.core.KineticModel` class contains all the information
for a kinetic model.
"""

import networkx as nx
from sympy import symbols, Mul

from kda import graph_utils, diagrams, calculations


class KineticModel(object):
	"""
	Core `kda` object that constructs the kinetic diagram,
	generates the intermediate graphs, and builds the algebraic
	expressions for steady-state probabilities and fluxes.

	Attributes
	==========
	cycles : list of lists of int
		All cycles in the kinetic diagram. This attribute becomes
		available after running the `build_cycles` method.
    partial_diagrams : array of Networkx Graphs
        The set of partial diagrams (i.e. spanning trees) for the
        kinetic diagram. This attribute becomes available after
        running the `build_partial_diagrams` method.
    directional_diagrams : array of Networkx MultiDiGraphs
        The set of directional diagrams for the kinetic diagram.
        This attribute becomes available after running the
        `build_directional_diagrams` method.
	flux_diagrams : list of lists of Networkx MultiDiGraphs
        The set of flux diagrams for each cycle in the kinetic
        diagram. This attribute becomes available after running
        the `build_flux_diagrams` method.
	probabilities : array of floats or list of SymPy expressions
		The steady-state probabilities for all states in the kinetic
		diagram. Probabilities are either an array of numeric values
		or the algebraic expressions. This attribute becomes available
		after running the `build_state_probabilities` method.
	"""

	def __init__(self, K=None, G=None):
		"""
		Parameters
		==========
	    K : ndarray (optional)
	        'NxN' array where 'N' is the number of nodes in the diagram `G`.
	        Adjacency matrix for `G` where each element kij is the edge weight
	        (i.e. transition rate constant). For example, for a 2-state model
	        with `k12=3` and `k21=4`, `K=[[0, 3], [4, 0]]`. Default is `None`.
	    G : NetworkX MultiDiGraph (optional)
	        Input diagram. Default is `None`.

		Raises
	    ======
	    RuntimeError
	    	If both `K` and `G` are `None`.
		"""
		if G is None or K is None:
			if K is not None:
				# if only K is input create the diagram
				G = nx.MultiDiGraph()
				graph_utils.generate_edges(G=G, K=K)
			elif G is not None:
				# if only G is input create the kinetic rate matrix
				K = graph_utils.retrieve_rate_matrix(G)
			else:
				msg = "To create a `KineticModel`, K or G must be input."
				raise RuntimeError(msg)

		self.K = K
		self.G = G

		# assign None for future attributes
		self.cycles = None
		self.partial_diagrams = None
		self.directional_diagrams = None
		self.flux_diagrams = None
		self.probabilities = None


	def build_cycles(self):
		"""Builds all cycles from the kinetic diagram."""
		self.cycles = graph_utils.find_all_unique_cycles(self.G)


	def build_partial_diagrams(self):
		"""Builds the partial diagrams for the kinetic diagram."""
		self.partial_diagrams = diagrams.generate_partial_diagrams(self.G, return_edges=False)


	def build_directional_diagrams(self):
		"""Builds the directional diagrams for the kinetic diagram."""
		self.directional_diagrams = diagrams.generate_directional_diagrams(self.G, return_edges=False)


	def build_flux_diagrams(self):
		"""Builds the flux diagrams for the kinetic diagram."""
		self.flux_diagrams = diagrams.generate_all_flux_diagrams(self.G)


	def get_partial_diagram_count(self):
		"""
		Retrieves the number of partial diagrams that will
		be created from the kinetic diagram.

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
		Retrieves the number of directional diagrams that will
		be created from the kinetic diagram.

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
		Retrieves the flux diagrams for a specific cycle.

		Parameters
		==========
	    cycle : list of int
	        List of node indices for cycle of interest, index zero.
	        Order of node indices does not matter.

		Returns
		=======
		The flux diagrams associated with the input cycle.
		"""
		return diagrams.generate_flux_diagrams(self.G, cycle)


	def build_state_probabilities(self, symbolic=True):
		"""
		Builds the state probabilities for the kinetic diagram. Probabilities
		can be stored as raw values or symbolic algebraic expressions.

		Parameters
		==========
	    symbolic : bool (optional)
	        Used to determine whether raw values or symbolic
	        expressions will be stored. Default is True.
		"""
		# TODO: may be able to leverage `calc_state_probs_from_diags()`
		# here, but it would require the user has already generated the
		# directional diagrams as edges, which is probably atypical
		self.probabilities = calculations.calc_state_probs(
			self.G, output_strings=symbolic)


	def transition_flux(self, i, j, net=True, symbolic=True):
		"""
		Creates the one-way or net transition fluxes.

		Parameters
		==========
		i : integer
			The index of the initial state for calculating
			transition fluxes.
		j : integer
			The index of the final state for calculating
			transition fluxes.
		net : bool (optional)
			Used to determine whether one-way transition fluxes
			or net transition fluxes will be returned. Default
			is True.
	    symbolic : bool (optional)
	        Used to determine whether raw values or symbolic
	        expressions will be returned. Default is True.

		Returns
		=======
		The transition flux (either one-way or net) from
		state `i` to state `j`.

	    Raises
	    ======
	    ValueError
	    	If the input states are the same (i.e. `i==j`).
	    TypeError
	    	If the stored state probability type (symbolic or numeric)
	    	is a differnt type than the requested transition
	    	flux type.
		"""
		if i == j:
			msg = "Input indices must be unique (i.e. i != j)."
			raise ValueError(msg)

		if self.probabilities is None:
			print(f"No probabilities found, generating state probabilities with symbolic={symbolic}")
			self.build_state_probabilities(symbolic=symbolic)
		else:
			# check if stored probability data type matches the
			# requested transition flux type (numeric vs symbolic)
			is_symbolic = isinstance(self.probabilities[0], Mul)
			if symbolic != is_symbolic:
				msg = (
					f"`KineticModel.probabilities` are the incorrect type for the"
					f" requested transition flux type. Regenerate probabilities"
					f" with `symbolic={symbolic}` before continuing."
				)
				raise TypeError(msg)

		if symbolic:
			# symbolic case
			# One-way transition flux: j_ij = k_ij * p_i
			j_ij = symbols(f"k{i}{j}") * self.probabilities[i-1]
			if not net:
				return j_ij.cancel()
			else:
				# Net transition flux: J_ij = j_ij - j_ji
				j_ji = symbols(f"k{j}{i}") * self.probabilities[j-1]
				return (j_ij - j_ji).cancel()
		else:
			# numerical case
			# One-way transition flux: j_ij = k_ij * p_i
			j_ij = self.K[i-1][j-1] * self.probabilities[i-1]
			if not net:
				return j_ij
			else:
				# Net transition flux: J_ij = j_ij - j_ji
				j_ji = self.K[j-1][i-1] * self.probabilities[j-1]
				return j_ij - j_ji
