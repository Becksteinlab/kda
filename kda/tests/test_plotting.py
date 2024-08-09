# Nikolaus Awtrey
# Beckstein Lab
# Department of Physics
# Arizona State University
#
# Kinetic Diagram Analysis Plotting Tests

import os
import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy.testing import assert_equal

from kda import plotting, graph_utils, diagrams


@pytest.mark.parametrize(
    "params, use_pos, cycles",
    [
        ({}, False, [0, 1, 3]),
        ({}, False, [0, 3, 2, 1]),
        ({}, True, [0, 1, 3]),
        ({"panel": False, "cbt": True}, True, [0, 1, 3]),
        ({"panel": False, "cbt": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        ({"panel": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        ({"panel": True, "cbt": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        (
            {"panel": True, "cbt": True, "curved_arrows": True},
            False,
            [
                [0, 1, 3],
                [0, 3, 2, 1],
                [0, 1, 3],
                [0, 3, 2, 1],
                [0, 1, 3],
                [0, 3, 2, 1],
                [0, 1, 3],
            ],
        ),
    ],
)
def test_draw_cycles(tmpdir, G4wl, pos_4wl, params, use_pos, cycles):
    if use_pos:
        pos = pos_4wl
    else:
        pos = None
    with tmpdir.as_cwd():
        path = os.getcwd()
        plotting.draw_cycles(G4wl, cycles, pos=pos, path=path, label="test", **params)


@pytest.mark.parametrize(
    "use_pos, use_flux_diags, double_flux_diags, params",
    [
        (False, False, False, {}),
        (True, False, False, {}),
        (False, True, False, {"panel": False}),
        (True, True, False, {"cbt": True}),
        (True, True, False, {"cbt": True, "panel": True}),
        (True, True, False, {"panel": True, "rows": 2}),
        (True, True, False, {"panel": True, "cols": 2}),
        (True, True, True, {"panel": True, "curved_arrows": True}),
    ],
)
def test_draw_diagrams(
    tmpdir,
    G4wl,
    pos_4wl,
    flux_diagrams_4wl,
    use_pos,
    use_flux_diags,
    double_flux_diags,
    params,
):
    if use_pos:
        pos = pos_4wl
    else:
        pos = None

    if use_flux_diags:
        diagrams = flux_diagrams_4wl
    else:
        diagrams = G4wl

    if double_flux_diags:
        diagrams *= 2
        diagrams = diagrams[:-1]

    with tmpdir.as_cwd():
        path = os.getcwd()
        plotting.draw_diagrams(diagrams, pos=pos, path=path, label="test", **params)


@pytest.mark.parametrize("params", [{}, {"bbox_coords": (0, 1)}])
def test_draw_ode_results(tmpdir, results_4wl, params):
    with tmpdir.as_cwd():
        path = os.getcwd()
        plotting.draw_ode_results(results_4wl, path=path, label="test", **params)


def test_plot_panel(tmpdir, flux_diagrams_4wl):
    # test to flush the code path for the position
    # generation for `plotting._plot_panel()`
    with tmpdir.as_cwd():
        path = os.getcwd()
        plotting._plot_panel(diagrams=flux_diagrams_4wl, pos=None)
        plt.close()


@pytest.mark.parametrize("input_mat", [
    # 5-state with leakage model from kda-examples
    np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ]
    ),
    # 8-state with leakage model from kda-examples
    np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ],
    ),
])
def test_color_by_target(input_mat):
    # test to cover the "color by target" functionality in the plotting
    # code. Double-checks that the flux diagrams are being generated
    # with the correct "is_target" attributes and checks the outputs
    # from `_get_node_colors()` for agreement.

    # initialize an empty graph object
    G = nx.MultiDiGraph()
    # populate the edge data
    graph_utils.generate_edges(G, input_mat)
    # collect the cycles and number of nodes from the diagram
    all_cycles = graph_utils.find_all_unique_cycles(G)
    n_nodes = G.number_of_nodes()

    for cycle in all_cycles:
        if len(cycle) == n_nodes:
            # skip any all-node cycles
            continue
        flux_diagrams = diagrams.generate_flux_diagrams(G, cycle)
        for diagram in flux_diagrams:
            # first we want to verify that the "is_target"
            # attribute is being assigned appropriately
            target_dict = nx.get_node_attributes(diagram, "is_target")
            nodes = np.asarray(list(target_dict.keys()))
            mask = np.asarray(list(target_dict.values()))
            actual_cbt_nodes = np.sort(nodes[mask])
            expected_cbt_nodes = np.sort(cycle)
            assert_equal(actual_cbt_nodes, expected_cbt_nodes)
            # if that's okay, let's check to see if the node colors
            # that are generated correspond to the node list when
            # using `_get_node_colors()`
            actual_node_colors = plotting._get_node_colors(cbt=True, obj=diagram)

            expected_node_colors = []
            for n in diagram.nodes:
                if n in cycle:
                    expected_node_colors.append("#FF8080")
                else:
                    expected_node_colors.append("0.8")
            assert_equal(actual_node_colors, expected_node_colors)
