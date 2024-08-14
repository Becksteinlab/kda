# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import networkx as nx

from kda import graph_utils, diagrams


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


class PartialDiagrams:
    """
    A benchmark to test the time and space complexity of the
    `diagrams.generate_partial_diagrams()` function.
    """
    param_names = ["graph", "return_edges"]
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

    def setup(self, graph, return_edges):
        # build the kinetic diagram and store it for use
        # in the time and memory tests
        self.G = build_graph(graph=graph)

    def time_generate_partial_diagrams(self, graph, return_edges):
        # benchmark partial diagram generation algorithm
        # for various models we commonly use for testing
        diagrams.generate_partial_diagrams(
            G=self.G,
            return_edges=return_edges,
        )

    def peakmem_generate_partial_diagrams(self, graph, return_edges):
        diagrams.generate_partial_diagrams(
            G=self.G,
            return_edges=return_edges,
        )


class DirectionalDiagrams:
    """
    A benchmark to test the time and space complexity of the
    `diagrams.generate_directional_diagrams()` function.
    """
    param_names = ["graph", "return_edges"]
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

    def setup(self, graph, return_edges):
        # build the kinetic diagram and store it for use
        # in the time and memory tests
        self.G = build_graph(graph=graph)

    def time_generate_directional_diagrams(self, graph, return_edges):
        # benchmark directional diagram generation algorithm
        # for various models we commonly use for testing
        diagrams.generate_directional_diagrams(
            G=self.G,
            return_edges=return_edges,
        )

    def peakmem_generate_directional_diagrams(self, graph, return_edges):
        diagrams.generate_directional_diagrams(
            G=self.G,
            return_edges=return_edges,
        )


class FluxDiagrams:
    """
    A benchmark to test the time and space complexity of the
    `diagrams.generate_all_flux_diagrams()` function.
    """
    param_names = ["graph"]
    params = [
        [
            "3-state",
            "Hill-5-state",
            "Hill-8-state",
            "EmrE-8-state",
        ],
    ]

    def setup(self, graph):
        # build the kinetic diagram and store it for use
        # in the time and memory tests
        self.G = build_graph(graph=graph)

    def time_generate_flux_diagrams(self, graph):
        # benchmark flux diagram generation algorithm
        # for various models we commonly use for testing
        diagrams.generate_all_flux_diagrams(G=self.G)

    def peakmem_generate_flux_diagrams(self, graph):
        diagrams.generate_all_flux_diagrams(G=self.G)