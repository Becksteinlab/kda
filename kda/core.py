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
	Kinetic Model
	"""

	def __init__(self, K=None, G=None):
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
		self.cycles = graph_utils.find_all_unique_cycles(self.G)


	def build_partial_diagrams(self):
		self.partial_diagrams = diagrams.generate_partial_diagrams(self.G, return_edges=False)


	def build_directional_diagrams(self):
		self.directional_diagrams = diagrams.generate_directional_diagrams(self.G, return_edges=False)


	def build_flux_diagrams(self):
		self.flux_diagrams = diagrams.generate_all_flux_diagrams(self.G)


	def get_partial_diagram_count(self):
		if self.partial_diagrams is not None:
			return len(self.partial_diagrams)
		else:
			return diagrams.enumerate_partial_diagrams(self.G)


	def get_directional_diagram_count(self):
		if self.directional_diagrams is not None:
			return len(self.directional_diagrams)
		else:
			partial_count = self.get_partial_diagram_count()
			return self.G.number_of_nodes() * partial_count


	def get_flux_diagrams(self, cycle):
		return diagrams.generate_flux_diagrams(self.G, cycle)


	def build_state_probabilities(self, symbolic=True):
		# TODO: may be able to leverage `calc_state_probs_from_diags()`
		# here, but it would require the user has already generated the
		# directional diagrams as edges, which is probably atypical
		self.probabilities = calculations.calc_state_probs(self.G, output_strings=symbolic)


	def transition_flux(self, i, j, net=True, symbolic=True):
		"""
		One-way transition flux: j_ij = k_ij * p_i
		Net transition flux: J_ij = j_ij - j_ji
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
			j_ij = symbols(f"k{i}{j}") * self.probabilities[i-1]
			if not net:
				return j_ij.cancel()
			else:
				j_ji = symbols(f"k{j}{i}") * self.probabilities[j-1]
				return (j_ij - j_ji).cancel()
		else:
			# numerical case
			j_ij = self.K[i-1][j-1] * self.probabilities[i-1]
			if not net:
				return j_ij
			else:
				j_ji = self.K[j-1][i-1] * self.probabilities[j-1]
				return j_ij - j_ji
