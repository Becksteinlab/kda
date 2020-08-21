# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Kinetic Diagram Analyzer Testing

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import networkx as nx
import kda


@pytest.fixture(scope='function')
def SP3(k12, k21, k23, k32, k13, k31):
    P1 = k23*k31 + k32*k21 + k31*k21
    P2 = k13*k32 + k32*k12 + k31*k12
    P3 = k13*k23 + k12*k23 + k21*k13
    Sigma = P1 + P2 + P3 # Normalization factor
    return np.array([P1, P2, P3])/Sigma


@pytest.fixture(scope='function')
def SP4(k12, k21, k23, k32, k34, k43, k41, k14):
    P1 = k43*k32*k21 + k23*k34*k41 + k21*k34*k41 + k41*k32*k21
    P2 = k12*k43*k32 + k14*k43*k32 + k34*k41*k12 + k32*k41*k12
    P3 = k43*k12*k23 + k23*k14*k43 + k21*k14*k43 + k41*k12*k23
    P4 = k12*k23*k34 + k14*k23*k34 + k34*k21*k14 + k32*k21*k14
    Sigma = P1 + P2 + P3 + P4 # Normalization factor
    return np.array([P1, P2, P3, P4])/Sigma


@pytest.fixture(scope='function')
def SP4WL(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
    P1 = k43*k32*k21 + k23*k34*k41 + k21*k34*k41 + k41*k32*k21 + k32*k42*k21 + k24*k34*k41 + k34*k42*k21 + k32*k24*k41
    P2 = k12*k43*k32 + k14*k43*k32 + k34*k41*k12 + k32*k41*k12 + k32*k42*k12 + k34*k14*k42 + k12*k34*k42 + k32*k14*k42
    P3 = k43*k12*k23 + k23*k14*k43 + k21*k14*k43 + k41*k12*k23 + k12*k42*k23 + k14*k24*k43 + k12*k24*k43 + k14*k42*k23
    P4 = k12*k23*k34 + k14*k23*k34 + k34*k21*k14 + k32*k21*k14 + k32*k12*k24 + k14*k24*k34 + k34*k12*k24 + k14*k32*k24
    Sigma = P1 + P2 + P3 + P4 # Normalization factor
    return np.array([P1, P2, P3, P4])/Sigma


@pytest.fixture(scope='function')
def SP5WL(k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54):
    P1 = k35*k54*k42*k21 + k24*k45*k53*k31 + k21*k45*k53*k31 + k42*k21*k53*k31 + k54*k42*k21*k31 + k54*k42*k32*k21 + k45*k53*k23*k31 + k53*k32*k42*k21 + k42*k23*k53*k31 + k54*k42*k23*k31 + k45*k53*k32*k21
    P2 = k12*k35*k54*k42 + k13*k35*k54*k42 + k45*k53*k31*k12 + k42*k53*k31*k12 + k31*k12*k54*k42 + k54*k42*k32*k12 + k45*k53*k13*k32 + k53*k32*k42*k12 + k53*k13*k32*k42 + k54*k42*k13*k32 + k45*k53*k32*k12
    P3 = k12*k24*k45*k53 + k13*k24*k45*k53 + k21*k13*k45*k53 + k42*k21*k13*k53 + k54*k42*k21*k13 + k54*k42*k12*k23 + k45*k53*k23*k13 + k42*k12*k23*k53 + k42*k23*k13*k53 + k54*k42*k23*k13 + k45*k53*k12*k23
    P4 = k12*k24*k35*k54 + k24*k13*k35*k54 + k21*k13*k35*k54 + k53*k31*k12*k24 + k54*k31*k12*k24 + k12*k32*k24*k54 + k13*k23*k35*k54 + k53*k32*k12*k24 + k13*k53*k32*k24 + k13*k32*k24*k54 + k12*k23*k35*k54
    P5 = k35*k12*k24*k45 + k13*k35*k24*k45 + k45*k21*k13*k35 + k42*k21*k13*k35 + k31*k12*k24*k45 + k12*k32*k24*k45 + k13*k23*k35*k45 + k12*k42*k23*k35 + k42*k23*k13*k35 + k13*k32*k24*k45 + k12*k23*k35*k45
    Sigma = P1 + P2 + P3 + P4 + P5 # Normalization factor
    return np.array([P1, P2, P3, P4, P5])/Sigma


@pytest.fixture(scope='function')
def SP6(k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16):
    P1 = k65*k54*k43*k32*k21 + k61*k54*k43*k32*k21 + k56*k61*k43*k32*k21 + k45*k56*k61*k32*k21 + k34*k45*k56*k61*k21 + k23*k34*k45*k56*k61
    P2 = k12*k65*k54*k43*k32 + k61*k12*k54*k43*k32 + k56*k61*k12*k43*k32 + k45*k56*k61*k32*k12 + k34*k45*k56*k61*k12 + k16*k65*k54*k43*k32
    P3 = k12*k23*k65*k54*k43 + k61*k12*k23*k54*k43 + k56*k61*k12*k23*k43 + k45*k56*k61*k12*k23 + k21*k16*k65*k54*k43 + k23*k16*k65*k54*k43
    P4 = k65*k54*k12*k23*k34 + k54*k61*k12*k23*k34 + k56*k61*k12*k23*k34 + k32*k21*k16*k65*k54 + k21*k16*k65*k54*k34 + k16*k65*k54*k23*k34
    P5 = k65*k12*k23*k34*k45 + k61*k12*k23*k34*k45 + k43*k32*k21*k16*k65 + k45*k32*k21*k16*k65 + k34*k45*k21*k16*k65 + k23*k34*k45*k16*k65
    P6 = k12*k23*k34*k45*k56 + k54*k43*k32*k21*k16 + k56*k43*k32*k21*k16 + k45*k56*k32*k21*k16 + k34*k45*k56*k21*k16 + k16*k23*k34*k45*k56
    Sigma = P1 + P2 + P3 + P4 + P5 + P6 # Normalization factor
    return np.array([P1, P2, P3, P4, P5, P6])/Sigma


@pytest.mark.parametrize('k21', [1e0, 1e1])
@pytest.mark.parametrize('k23', [1e0, 1e1])
@pytest.mark.parametrize('k32', [1e0, 1e1])
@pytest.mark.parametrize('k13', [1e0, 1e1])
@pytest.mark.parametrize('k31', [1e0, 1e1])
@pytest.mark.parametrize('k12', [1e0, 1e1])
def test_3(k12, k21, k23, k32, k13, k31, SP3):
    k3 = np.array([[0, k12, k13],
                  [k21, 0, k23],
                  [k31, k32, 0]])
    k3s = np.array([[0, "k12", "k13"],
                    ["k21", 0, "k23"],
                    ["k31", "k32", 0]])
    rate_names3 = ["k12", "k21", "k23", "k32", "k13", "k31"]
    G3 = nx.MultiDiGraph()
    kda.generate_edges(G3, k3, k3s, name_key='name', val_key='val')
    SP3_KDA = kda.calc_state_probs(G3, key='val')
    sympy_funcs3 = kda.calc_state_probs(G3, key='name', output_strings=True)
    state_prob_funcs3 = kda.construct_lambdify_funcs(sympy_funcs3, rate_names3)
    SP3_SymPy = []
    for i in range(G3.number_of_nodes()):
        SP3_SymPy.append(state_prob_funcs3[i](k12, k21, k23, k32, k13, k31))
    SP3_SymPy = np.array(SP3_SymPy)
    p3 = np.array([1, 1, 1])/3
    results3 = kda.solve_ODE(p3, k3, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    SP3_ODE = results3.y.T[-1]
    SP3_SVD = kda.SVD(k3, tol=1e-15)
    assert_almost_equal(SP3, SP3_SVD, decimal=12)
    assert_almost_equal(SP3, SP3_KDA, decimal=15)
    assert_almost_equal(SP3, SP3_SymPy, decimal=15)
    assert_almost_equal(SP3, SP3_ODE, decimal=10)


@pytest.mark.parametrize('k12', [1e0, 1e1])
@pytest.mark.parametrize('k21', [1e0, 1e1])
@pytest.mark.parametrize('k23', [1e0, 1e1])
@pytest.mark.parametrize('k32', [1e0, 1e1])
@pytest.mark.parametrize('k34', [1e0, 1e1])
@pytest.mark.parametrize('k43', [1e0, 1e1])
@pytest.mark.parametrize('k41', [1e0, 1e1])
@pytest.mark.parametrize('k14', [1e0, 1e1])
def test_4(k12, k21, k23, k32, k34, k43, k41, k14, SP4):
    k4 = np.array([[0, k12, 0, k14],
                   [k21, 0, k23, 0],
                   [0, k32, 0, k34],
                   [k41, 0, k43, 0]])
    k4s = np.array([[0, "k12", 0, "k14"],
                    ["k21", 0, "k23", 0],
                    [0, "k32", 0, "k34"],
                    ["k41", 0, "k43", 0]])
    rate_names4 = ["k12", "k21", "k23", "k32", "k34", "k43", "k41", "k14"]
    G4 = nx.MultiDiGraph()
    kda.generate_edges(G4, k4, k4s, name_key='name', val_key='val')
    SP4_KDA = kda.calc_state_probs(G4, key='val')
    sympy_funcs4 = kda.calc_state_probs(G4, key='name', output_strings=True)
    state_prob_funcs4 = kda.construct_lambdify_funcs(sympy_funcs4, rate_names4)
    SP4_SymPy = []
    for i in range(G4.number_of_nodes()):
        SP4_SymPy.append(state_prob_funcs4[i](k12, k21, k23, k32, k34, k43, k41, k14))
    SP4_SymPy = np.array(SP4_SymPy)
    p4 = np.array([1, 1, 1, 1])/4
    results4 = kda.solve_ODE(p4, k4, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    SP4_ODE = results4.y.T[-1]
    SP4_SVD = kda.SVD(k4, tol=1e-15)
    assert_almost_equal(SP4, SP4_SVD, decimal=12)
    assert_almost_equal(SP4, SP4_KDA, decimal=15)
    assert_almost_equal(SP4, SP4_SymPy, decimal=15)
    assert_almost_equal(SP4, SP4_ODE, decimal=10)


@pytest.mark.parametrize('k12', [1e0, 1e1])
@pytest.mark.parametrize('k21', [1e0, 1e1])
@pytest.mark.parametrize('k23', [1e0, 1e1])
@pytest.mark.parametrize('k32', [1e0, 1e1])
@pytest.mark.parametrize('k34', [1e0, 1e1])
@pytest.mark.parametrize('k43', [1e0, 1e1])
@pytest.mark.parametrize('k41', [1e0, 1e1])
@pytest.mark.parametrize('k14', [1e0, 1e1])
@pytest.mark.parametrize('k24', [1e0, 1e1])
@pytest.mark.parametrize('k42', [1e0, 1e1])
def test_4WL(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42, SP4WL):
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
    SP4WL_KDA = kda.calc_state_probs(G4wl, key='val')
    sympy_funcs4wl = kda.calc_state_probs(G4wl, key='name', output_strings=True)
    state_prob_funcs4wl = kda.construct_lambdify_funcs(sympy_funcs4wl, rate_names4wl)
    SP4WL_SymPy = []
    for i in range(G4wl.number_of_nodes()):
        SP4WL_SymPy.append(state_prob_funcs4wl[i](k12, k21, k23, k32, k34, k43, k41, k14, k24, k42))
    SP4WL_SymPy = np.array(SP4WL_SymPy)
    p4wl = np.array([1, 1, 1, 1])/4
    results4wl = kda.solve_ODE(p4wl, k4wl, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    SP4WL_ODE = results4wl.y.T[-1]
    SP4WL_SVD = kda.SVD(k4wl, tol=1e-15)
    assert_almost_equal(SP4WL, SP4WL_SVD, decimal=12)
    assert_almost_equal(SP4WL, SP4WL_KDA, decimal=15)
    assert_almost_equal(SP4WL, SP4WL_SymPy, decimal=15)
    assert_almost_equal(SP4WL, SP4WL_ODE, decimal=10)


@pytest.mark.parametrize('k12', [1e0])
@pytest.mark.parametrize('k21', [1e0, 1e1])
@pytest.mark.parametrize('k23', [1e0])
@pytest.mark.parametrize('k32', [1e0, 1e1])
@pytest.mark.parametrize('k13', [1e0])
@pytest.mark.parametrize('k31', [1e0, 1e1])
@pytest.mark.parametrize('k24', [1e0])
@pytest.mark.parametrize('k42', [1e0, 1e1])
@pytest.mark.parametrize('k35', [1e0])
@pytest.mark.parametrize('k53', [1e0, 1e1])
@pytest.mark.parametrize('k45', [1e0])
@pytest.mark.parametrize('k54', [1e0, 1e1])
def test_5WL(k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54, SP5WL):
    k5wl = np.array([[  0, k12, k13,   0,   0],
                     [k21,   0, k23, k24,   0],
                     [k31, k32,   0,   0, k35],
                     [  0, k42,   0,   0, k45],
                     [  0,   0, k53, k54,   0]])
    k5wls = np.array([[  0, "k12", "k13",   0,   0],
                      ["k21",   0, "k23", "k24",   0],
                      ["k31", "k32",   0,   0, "k35"],
                      [  0, "k42",   0,   0, "k45"],
                      [  0,   0, "k53", "k54",   0]])
    rate_names5wl = ["k12", "k21", "k23", "k32", "k13", "k31", "k24", "k42", "k35", "k53", "k45", "k54"]
    G5wl = nx.MultiDiGraph()
    kda.generate_edges(G5wl, k5wl, k5wls, name_key='name', val_key='val')
    SP5WL_KDA = kda.calc_state_probs(G5wl, key='val')
    sympy_funcs5wl = kda.calc_state_probs(G5wl, key='name', output_strings=True)
    state_prob_funcs5wl = kda.construct_lambdify_funcs(sympy_funcs5wl, rate_names5wl)
    SP5WL_SymPy = []
    for i in range(G5wl.number_of_nodes()):
        SP5WL_SymPy.append(state_prob_funcs5wl[i](k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54))
    SP5WL_SymPy = np.array(SP5WL_SymPy)
    p5wl = np.array([1, 1, 1, 1, 1])/5
    results5wl = kda.solve_ODE(p5wl, k5wl, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    SP5WL_ODE = results5wl.y.T[-1]
    SP5WL_SVD = kda.SVD(k5wl, tol=1e-15)
    assert_almost_equal(SP5WL, SP5WL_SVD, decimal=12)
    assert_almost_equal(SP5WL, SP5WL_KDA, decimal=15)
    assert_almost_equal(SP5WL, SP5WL_SymPy, decimal=15)
    assert_almost_equal(SP5WL, SP5WL_ODE, decimal=10)


@pytest.mark.parametrize('k12', [1e0])
@pytest.mark.parametrize('k21', [1e0, 1e1])
@pytest.mark.parametrize('k23', [1e0])
@pytest.mark.parametrize('k32', [1e0, 1e1])
@pytest.mark.parametrize('k34', [1e0])
@pytest.mark.parametrize('k43', [1e0, 1e1])
@pytest.mark.parametrize('k45', [1e0])
@pytest.mark.parametrize('k54', [1e0, 1e1])
@pytest.mark.parametrize('k56', [1e0])
@pytest.mark.parametrize('k65', [1e0, 1e1])
@pytest.mark.parametrize('k61', [1e0])
@pytest.mark.parametrize('k16', [1e0, 1e1])
def test_6(k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16, SP6):
    k6 = np.array([[  0, k12,   0,   0,   0, k16],
                   [k21,   0, k23,   0,   0,   0],
                   [  0, k32,   0, k34,   0,   0],
                   [  0,   0, k43,   0, k45,   0],
                   [  0,   0,   0, k54,   0, k56],
                   [k61,   0,   0,   0, k65,   0]])
    k6s = np.array([[  0, "k12",   0,   0,   0, "k16"],
                    ["k21",   0, "k23",   0,   0,   0],
                    [  0, "k32",   0, "k34",   0,   0],
                    [  0,   0, "k43",   0, "k45",   0],
                    [  0,   0,   0, "k54",   0, "k56"],
                    ["k61",   0,   0,   0, "k65",   0]])
    rate_names6 = ["k12", "k21", "k23", "k32", "k34", "k43", "k45", "k54", "k56", "k65", "k61", "k16"]
    G6 = nx.MultiDiGraph()
    kda.generate_edges(G6, k6, k6s, name_key='name', val_key='val')
    SP6_KDA = kda.calc_state_probs(G6, key='val')
    sympy_funcs6 = kda.calc_state_probs(G6, key='name', output_strings=True)
    state_prob_funcs6 = kda.construct_lambdify_funcs(sympy_funcs6, rate_names6)
    SP6_SymPy = []
    for i in range(G6.number_of_nodes()):
        SP6_SymPy.append(state_prob_funcs6[i](k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16))
    SP6_SymPy = np.array(SP6_SymPy)
    p6 = np.array([1, 1, 1, 1, 1, 1])/6
    results6 = kda.solve_ODE(p6, k6, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13)
    SP6_ODE = results6.y.T[-1]
    SP6_SVD = kda.SVD(k6, tol=1e-13)
    assert_almost_equal(SP6, SP6_SVD, decimal=12)
    assert_almost_equal(SP6, SP6_KDA, decimal=15)
    assert_almost_equal(SP6, SP6_SymPy, decimal=15)
    assert_almost_equal(SP6, SP6_ODE, decimal=10)


@pytest.mark.parametrize('k12', [1e0])
@pytest.mark.parametrize('k21', [1e0])
@pytest.mark.parametrize('k23', [1e0])
@pytest.mark.parametrize('k32', [1e0])
@pytest.mark.parametrize('k34', [1e0])
@pytest.mark.parametrize('k43', [1e0])
@pytest.mark.parametrize('k41', [1e0])
@pytest.mark.parametrize('k14', [1e0])
@pytest.mark.parametrize('k24', [1e0, 5e0, 1e1, 1.5e1])
@pytest.mark.parametrize('k42', [1e0, 5e0, 1e1, 1.5e1])
def test_generate_flux_diags_4WL(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
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
    G4wl_cycles = kda.find_all_unique_cycles(G4wl)
    flux_diags_G4wl = []
    for cycle in G4wl_cycles:
        flux_diags_G4wl.append(kda.generate_flux_diagrams(G4wl, cycle))
    all_flux_diags_G4wl = kda.generate_all_flux_diagrams(G4wl)
    diags1_G4wl = [diag for diags in all_flux_diags_G4wl for diag in diags]
    diags2_G4wl = [diag for diags in flux_diags_G4wl[1:] for diag in diags]
    assert len(G4wl_cycles) == 3
    assert flux_diags_G4wl[0] == None
    assert len(flux_diags_G4wl[1]) == 2
    assert len(flux_diags_G4wl[2]) == 2
    assert diags1_G4wl[0].edges == diags2_G4wl[0].edges
    assert diags1_G4wl[1].edges == diags2_G4wl[1].edges
    assert diags1_G4wl[2].edges == diags2_G4wl[2].edges
    assert diags1_G4wl[3].edges == diags2_G4wl[3].edges


@pytest.mark.parametrize('k12', [1e0, 2e0])
@pytest.mark.parametrize('k21', [1e0, 2e0])
@pytest.mark.parametrize('k23', [1e0, 2e0])
@pytest.mark.parametrize('k32', [1e0, 2e0])
@pytest.mark.parametrize('k13', [1e0, 2e0])
@pytest.mark.parametrize('k31', [1e0, 2e0])
def test_calc_cycle_flux_3(k12, k21, k23, k32, k13, k31):
    k3 = np.array([[0, k12, k13],
                  [k21, 0, k23],
                  [k31, k32, 0]])
    k3s = np.array([[0, "k12", "k13"],
                    ["k21", 0, "k23"],
                    ["k31", "k32", 0]])
    rate_names3 = ["k12", "k21", "k23", "k32", "k13", "k31"]
    G3 = nx.MultiDiGraph()
    kda.generate_edges(G3, k3, k3s, name_key='name', val_key='val')
    G3_cycles = kda.find_all_unique_cycles(G3)[0]
    cycle_order3 = [0, 1]
    cf_3 = kda.calc_cycle_flux(G3, G3_cycles, cycle_order3, key='val', output_strings=False)
    sympy_cf_3 = kda.calc_cycle_flux(G3, G3_cycles, cycle_order3, key='name', output_strings=True)
    assert cf_3 == (k12*k23*k31 - k13*k21*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)
    assert str(sympy_cf_3) == '(k12*k23*k31 - k13*k21*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)'


@pytest.mark.parametrize('k12', [1e0])
@pytest.mark.parametrize('k21', [1e0])
@pytest.mark.parametrize('k23', [1e0])
@pytest.mark.parametrize('k32', [1e0, 3e0])
@pytest.mark.parametrize('k34', [1e0, 5e0])
@pytest.mark.parametrize('k43', [1e0])
@pytest.mark.parametrize('k41', [1e0])
@pytest.mark.parametrize('k14', [1e0])
@pytest.mark.parametrize('k24', [1e0])
@pytest.mark.parametrize('k42', [1e0])
def test_sigma_K_4WL(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
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
    cycle = [0, 1, 3]
    flux_diags_4wl = kda.generate_flux_diagrams(G4wl, cycle)
    sigma_K_4wl = kda.calculate_sigma_K(G4wl, cycle, flux_diags_4wl, key='val', output_strings=False)
    sigma_K_4wls = kda.calculate_sigma_K(G4wl, cycle, flux_diags_4wl, key='name', output_strings=True)
    assert sigma_K_4wl == k32 + k34
    assert sigma_K_4wls == 'k32+k34'


@pytest.mark.parametrize('k12', [1e0])
@pytest.mark.parametrize('k21', [5e0])
@pytest.mark.parametrize('k23', [1e0])
@pytest.mark.parametrize('k32', [1e0])
@pytest.mark.parametrize('k34', [1e0])
@pytest.mark.parametrize('k43', [1e0])
@pytest.mark.parametrize('k41', [1e0])
@pytest.mark.parametrize('k14', [2e0])
@pytest.mark.parametrize('k24', [1e0])
@pytest.mark.parametrize('k42', [3e0])
def test_thermo_force_4WL(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
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
    cycles = kda.find_all_unique_cycles(G4wl)
    cycle_0321 = cycles[0]
    cycle_031 = cycles[1]
    cycle_132 = cycles[2]
    cycle_order_4wl = [[3, 0], [3, 0], [3, 1]]
    tf_4wl_0321 = kda.calculate_thermo_force(G4wl, cycle_0321, cycle_order_4wl[0], key='val', output_strings=False)
    tf_4wls_0321 = kda.calculate_thermo_force(G4wl, cycle_0321, cycle_order_4wl[0], key='name', output_strings=True)
    tf_4wl_031 = kda.calculate_thermo_force(G4wl, cycle_031, cycle_order_4wl[1], key='val', output_strings=False)
    tf_4wls_031 = kda.calculate_thermo_force(G4wl, cycle_031, cycle_order_4wl[1], key='name', output_strings=True)
    tf_4wl_132 = kda.calculate_thermo_force(G4wl, cycle_132, cycle_order_4wl[2], key='val', output_strings=False)
    tf_4wls_132 = kda.calculate_thermo_force(G4wl, cycle_132, cycle_order_4wl[2], key='name', output_strings=True)
    assert tf_4wl_0321 == np.log(k12*k23*k34*k41/(k14*k21*k32*k43))
    assert str(tf_4wls_0321) == 'log(k12*k23*k34*k41/(k14*k21*k32*k43))'
    assert tf_4wl_031 == np.log(k12*k24*k41/(k14*k21*k42))
    assert str(tf_4wls_031) == 'log(k12*k24*k41/(k14*k21*k42))'
    assert tf_4wl_132 == np.log(k23*k34*k42/(k24*k32*k43))
    assert str(tf_4wls_132) == 'log(k23*k34*k42/(k24*k32*k43))'


def test_find_uncommon_edges():
    edges1 = [(0, 1, 0), (1, 0, 0), (1, 2, 0), (2, 1, 0), (2, 3, 0), (3, 4, 0)]
    edges2 = [(0, 1, 0), (1, 0, 0), (1, 2, 0), (2, 1, 0)]
    uu_edges = kda.find_uncommon_edges(edges1, edges2)
    assert uu_edges == [(2, 3, 0), (3, 4, 0)]
