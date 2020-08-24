# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Kinetic Diagram Analyzer Testing

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import kda
import kda.plotting


@pytest.fixture(scope='module')
def G4wl():
    k12, k21, k23, k32, k34, k43, k41, k14, k24, k42 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    k4wl = np.array([[0, k12, 0, k14],
                    [k21, 0, k23, k24],
                    [0, k32, 0, k34],
                    [k41, k42, k43, 0]])
    k4wls = np.array([[0, "k12", 0, "k14"],
                      ["k21", 0, "k23", "k24"],
                      [0, "k32", 0, "k34"],
                      ["k41", "k42", "k43", 0]])
    G4wl = nx.MultiDiGraph()
    kda.generate_edges(G4wl, k4wl, k4wls, name_key='name', val_key='val')
    return G4wl


@pytest.fixture(scope='module')
def pos_4wl(center=[0, 0], radius=10):
    N = 4
    angle = np.pi*np.array([1/4, 3/4, 5/4, 7/4])        # Angles start at 0 and go clockwise (like unit circle)
    array = np.zeros((N, 2))                                        # Empty 2D array of shape (6x2)
    for i in range(N):                                              # Creates hexagon of atoms in the xy-plane
        array[i, 0] = np.cos(angle[i])
        array[i, 1] = np.sin(angle[i])
    pos = {}
    for i in range(N):
        pos[i] = array[i]*radius + center
    return pos


@pytest.fixture(scope='module')
def results_4wl():
    k12, k21, k23, k32, k34, k43, k41, k14, k24, k42 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    k4wl = np.array([[0, k12, 0, k14],
                    [k21, 0, k23, k24],
                    [0, k32, 0, k34],
                    [k41, k42, k43, 0]])
    k4wls = np.array([[0, "k12", 0, "k14"],
                      ["k21", 0, "k23", "k24"],
                      [0, "k32", 0, "k34"],
                      ["k41", "k42", "k43", 0]])
    rate_names4wl = ["k12", "k21", "k23", "k32", "k34", "k43", "k41", "k14", "k24", "k42"]
    G4wl = nx.MultiDiGraph()
    kda.generate_edges(G4wl, k4wl, k4wls, name_key='name', val_key='val')
    p4wl = np.array([1, 1, 1, 1])/4
    results4wl = kda.solve_ODE(p4wl, k4wl, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    return results4wl


def test_plot_diagram(G4wl, pos_4wl):
    kda.plotting.draw_diagrams(G4wl)
    kda.plotting.draw_diagrams(G4wl, pos=pos_4wl)


def test_plot_cycle(G4wl, pos_4wl):
    kda.plotting.draw_cycles(G4wl, [0, 1, 3])
    kda.plotting.draw_cycles(G4wl, [0, 3, 2, 1])
    kda.plotting.draw_cycles(G4wl, [0, 1, 3], pos=pos_4wl)
    kda.plotting.draw_cycles(G4wl, [0, 1, 3], pos=pos_4wl, cbt=True)


def test_plot_diagrams(G4wl, pos_4wl):
    flux_diags = kda.generate_flux_diagrams(G4wl, [0, 1, 3])
    kda.plotting.draw_diagrams(flux_diags)
    kda.plotting.draw_diagrams(flux_diags, pos=pos_4wl, cbt=True)
    kda.plotting.draw_diagrams(flux_diags, pos=pos_4wl, panel=True, cbt=True)


def test_plot_ODE_probs(results_4wl):
    kda.plotting.draw_ODE_results(results_4wl)
