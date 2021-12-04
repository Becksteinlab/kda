# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Kinetic Diagram Analyzer Testing

import os
import pytest
import numpy as np
import networkx as nx

from kda import plotting, graph_utils, diagrams, ode


@pytest.fixture(scope="module")
def G4wl():
    k4wl = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]
    )
    G4wl = nx.MultiDiGraph()
    graph_utils.generate_edges(G4wl, k4wl)
    return G4wl


@pytest.fixture(scope="function")
def flux_diagrams_4wl(G4wl):
    cycles = graph_utils.find_all_unique_cycles(G4wl)

    flux_diagrams = []
    for cycle in cycles:
        if not cycle is None:
            flux_diagrams = diagrams.generate_flux_diagrams(G4wl, cycle)
            if not flux_diagrams is None:
                flux_diagrams.extend(flux_diagrams)
    return flux_diagrams


@pytest.fixture(scope="module")
def pos_4wl():
    pos = {
        0: [1, 1],
        1: [-1, 1],
        2: [-1, -1],
        3: [1, -1],
    }
    return pos


@pytest.fixture(scope="module")
def results_4wl():
    k4wl = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]
    )
    G4wl = nx.MultiDiGraph()
    graph_utils.generate_edges(G4wl, k4wl)
    p4wl = np.array([1, 0, 0, 0])
    results4wl = ode.ode_solver(
        p4wl, k4wl, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
    )
    return results4wl


@pytest.mark.parametrize(
    "params, use_pos, cycles",
    [
        ({}, False, [0, 1, 3]),
        ({}, False, [0, 3, 2, 1]),
        ({}, True, [0, 1, 3]),
        ({"cbt": True}, True, [0, 1, 3]),
        ({"cbt": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        ({"panel": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        ({"panel": True, "cbt": True}, False, [[0, 1, 3], [0, 3, 2, 1]]),
        (
            {"panel": True, "cbt": True},
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
        (False, True, False, {}),
        (True, True, False, {"cbt": True}),
        (True, True, False, {"cbt": True, "panel": True}),
        (True, True, False, {"panel": True, "rows": 2}),
        (True, True, False, {"panel": True, "cols": 2}),
        (True, True, True, {"panel": True}),
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
