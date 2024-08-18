# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import networkx as nx

import kda
from kda import graph_utils, calculations, diagrams


def build_graph(graph):
    if graph == "3-state":
        # 3-state model, simplest test model
        K = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
    elif graph == "Hill-5-state":
        # 5-state model with leakage, used extensively in T.L. Hill's
        # "Free Energy Transduction and Biochemical Cycle Kinetics"
        K = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ])

    elif graph == "Hill-8-state":
        # 8-state model with leakage used for flux diagram example in
        # T.L. Hill's "Free Energy Transduction and Biochemical Cycle Kinetics"
        K = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ])
    elif graph == "EmrE-8-state":
        # EmrE 8-state model
        K = np.array([
            [0, 1, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0],
        ])
    G = nx.MultiDiGraph()
    graph_utils.generate_edges(G=G, vals=K)
    return G


class StateProbs:
    """
    A benchmark to test the time and space complexity of the
    `calculations.calc_state_probs()` function.
    """
    param_names = ["graph", "output_strings"]
    params = [
        [
            "3-state",
            "Hill-5-state",
            "Hill-8-state",
            "EmrE-8-state",
        ],
        [
            True,
            False,
        ]
    ]

    def setup(self, graph, output_strings):
        # build the kinetic diagram and store it for use
        # in the time and memory tests
        if output_strings:
            key = "name"
        else:
            key = "val"
        self.key = key
        self.G = build_graph(graph=graph)

    def time_calc_state_probs(self, graph, output_strings):
        # benchmark state probability calculation algorithm
        # for various models we commonly use for testing
        calculations.calc_state_probs(
            G=self.G,
            key=self.key,
            output_strings=output_strings,
        )

    def peakmem_calc_state_probs(self, graph, output_strings):
        calculations.calc_state_probs(
            G=self.G,
            key=self.key,
            output_strings=output_strings,
        )


class Sigma:
    """
    A benchmark to test the time and space complexity of the
    `calculations.calc_sigma()` function.
    """
    param_names = ["graph", "output_strings"]
    params = [
        [
            "3-state",
            "Hill-5-state",
            "Hill-8-state",
            "EmrE-8-state",
        ],
        [
            True,
            False,
        ]
    ]

    def setup(self, graph, output_strings):
        # build the kinetic diagram and directional diagram edges
        # and store them for use in the time and memory benchmarks
        if output_strings:
            key = "name"
        else:
            key = "val"
        self.key = key
        self.G = build_graph(graph=graph)
        self.dir_edges = diagrams.generate_directional_diagrams(
            G=self.G, return_edges=True)

    def time_calc_sigma(self, graph, output_strings):
        # benchmark sigma calculation algorithm
        # for various models we commonly use for testing
        calculations.calc_sigma(
            G=self.G,
            dirpar_edges=self.dir_edges,
            key=self.key,
            output_strings=output_strings,
        )

    def peakmem_calc_sigma(self, graph, output_strings):
        calculations.calc_sigma(
            G=self.G,
            dirpar_edges=self.dir_edges,
            key=self.key,
            output_strings=output_strings,
        )


class CycleFlux:
    """
    A benchmark to test the time and space complexity of the
    `calculations.calc_net_cycle_flux()` function.
    """
    param_names = ["graph", "output_strings"]
    params = [
        [
            "3-state",
            "Hill-5-state",
            "Hill-8-state",
        ],
        [
            True,
            False,
        ]
    ]

    def setup(self, graph, output_strings):
        if output_strings:
            key = "name"
        else:
            key = "val"
        self.key = key
        G = build_graph(graph=graph)
        model = kda.KineticModel(G=G)
        model.build_cycles()
        self.G = model.G
        self.cycles = model.cycles
        self.dir_edges = diagrams.generate_directional_diagrams(
            G=self.G, return_edges=True)

    def time_calc_net_cycle_flux(self, graph, output_strings):
        # benchmark cycle flux calculation algorithm
        # for various models we commonly use for testing
        for cycle in self.cycles:
            calculations.calc_net_cycle_flux(
                G=self.G,
                cycle=cycle,
                order=cycle[:2],
                key=self.key,
                output_strings=output_strings,
                dir_edges=self.dir_edges,
            )

    def peakmem_calc_net_cycle_flux(self, graph, output_strings):
        for cycle in self.cycles:
            calculations.calc_net_cycle_flux(
                G=self.G,
                cycle=cycle,
                order=cycle[:2],
                key=self.key,
                output_strings=output_strings,
                dir_edges=self.dir_edges,
            )
