# Nikolaus Awtrey
# Beckstein Lab
# Arizona State University
#
# Kinetic Diagram Analysis Testing

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_array_equal
from hypothesis import settings, given, strategies as st, HealthCheck
import networkx as nx

from kda import calculations, diagrams, graph_utils, expressions, ode, svd
from kda.exceptions import CycleError


class StateProbs3:
    def __init__(self, k_vals):
        self.k_vals = k_vals

    def return_probs(self, k12, k21, k23, k32, k13, k31):
        P1 = k23 * k31 + k32 * k21 + k31 * k21
        P2 = k13 * k32 + k32 * k12 + k31 * k12
        P3 = k13 * k23 + k12 * k23 + k21 * k13
        Sigma = P1 + P2 + P3  # Normalization factor
        return np.array([P1, P2, P3]) / Sigma


class StateProbs4:
    def __init__(self, k_vals):
        self.k_vals = k_vals

    def return_probs(self, k12, k21, k23, k32, k34, k43, k41, k14):
        P1 = k43 * k32 * k21 + k23 * k34 * k41 + k21 * k34 * k41 + k41 * k32 * k21
        P2 = k12 * k43 * k32 + k14 * k43 * k32 + k34 * k41 * k12 + k32 * k41 * k12
        P3 = k43 * k12 * k23 + k23 * k14 * k43 + k21 * k14 * k43 + k41 * k12 * k23
        P4 = k12 * k23 * k34 + k14 * k23 * k34 + k34 * k21 * k14 + k32 * k21 * k14
        Sigma = P1 + P2 + P3 + P4  # Normalization factor
        return np.array([P1, P2, P3, P4]) / Sigma


class StateProbs4WL:
    def __init__(self, k_vals):
        self.k_vals = k_vals

    def return_probs(self, k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
        P1 = (
            k43 * k32 * k21
            + k23 * k34 * k41
            + k21 * k34 * k41
            + k41 * k32 * k21
            + k32 * k42 * k21
            + k24 * k34 * k41
            + k34 * k42 * k21
            + k32 * k24 * k41
        )
        P2 = (
            k12 * k43 * k32
            + k14 * k43 * k32
            + k34 * k41 * k12
            + k32 * k41 * k12
            + k32 * k42 * k12
            + k34 * k14 * k42
            + k12 * k34 * k42
            + k32 * k14 * k42
        )
        P3 = (
            k43 * k12 * k23
            + k23 * k14 * k43
            + k21 * k14 * k43
            + k41 * k12 * k23
            + k12 * k42 * k23
            + k14 * k24 * k43
            + k12 * k24 * k43
            + k14 * k42 * k23
        )
        P4 = (
            k12 * k23 * k34
            + k14 * k23 * k34
            + k34 * k21 * k14
            + k32 * k21 * k14
            + k32 * k12 * k24
            + k14 * k24 * k34
            + k34 * k12 * k24
            + k14 * k32 * k24
        )
        Sigma = P1 + P2 + P3 + P4  # Normalization factor
        return np.array([P1, P2, P3, P4]) / Sigma


class StateProbs5WL:
    def __init__(self, k_vals):
        self.k_vals = k_vals

    def return_probs(self, k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54):
        P1 = (
            k35 * k54 * k42 * k21
            + k24 * k45 * k53 * k31
            + k21 * k45 * k53 * k31
            + k42 * k21 * k53 * k31
            + k54 * k42 * k21 * k31
            + k54 * k42 * k32 * k21
            + k45 * k53 * k23 * k31
            + k53 * k32 * k42 * k21
            + k42 * k23 * k53 * k31
            + k54 * k42 * k23 * k31
            + k45 * k53 * k32 * k21
        )
        P2 = (
            k12 * k35 * k54 * k42
            + k13 * k35 * k54 * k42
            + k45 * k53 * k31 * k12
            + k42 * k53 * k31 * k12
            + k31 * k12 * k54 * k42
            + k54 * k42 * k32 * k12
            + k45 * k53 * k13 * k32
            + k53 * k32 * k42 * k12
            + k53 * k13 * k32 * k42
            + k54 * k42 * k13 * k32
            + k45 * k53 * k32 * k12
        )
        P3 = (
            k12 * k24 * k45 * k53
            + k13 * k24 * k45 * k53
            + k21 * k13 * k45 * k53
            + k42 * k21 * k13 * k53
            + k54 * k42 * k21 * k13
            + k54 * k42 * k12 * k23
            + k45 * k53 * k23 * k13
            + k42 * k12 * k23 * k53
            + k42 * k23 * k13 * k53
            + k54 * k42 * k23 * k13
            + k45 * k53 * k12 * k23
        )
        P4 = (
            k12 * k24 * k35 * k54
            + k24 * k13 * k35 * k54
            + k21 * k13 * k35 * k54
            + k53 * k31 * k12 * k24
            + k54 * k31 * k12 * k24
            + k12 * k32 * k24 * k54
            + k13 * k23 * k35 * k54
            + k53 * k32 * k12 * k24
            + k13 * k53 * k32 * k24
            + k13 * k32 * k24 * k54
            + k12 * k23 * k35 * k54
        )
        P5 = (
            k35 * k12 * k24 * k45
            + k13 * k35 * k24 * k45
            + k45 * k21 * k13 * k35
            + k42 * k21 * k13 * k35
            + k31 * k12 * k24 * k45
            + k12 * k32 * k24 * k45
            + k13 * k23 * k35 * k45
            + k12 * k42 * k23 * k35
            + k42 * k23 * k13 * k35
            + k13 * k32 * k24 * k45
            + k12 * k23 * k35 * k45
        )
        Sigma = P1 + P2 + P3 + P4 + P5  # Normalization factor
        return np.array([P1, P2, P3, P4, P5]) / Sigma


class StateProbs6:
    def __init__(self, k_vals):
        self.k_vals = k_vals

    def return_probs(self, k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16):
        P1 = (
            k65 * k54 * k43 * k32 * k21
            + k61 * k54 * k43 * k32 * k21
            + k56 * k61 * k43 * k32 * k21
            + k45 * k56 * k61 * k32 * k21
            + k34 * k45 * k56 * k61 * k21
            + k23 * k34 * k45 * k56 * k61
        )
        P2 = (
            k12 * k65 * k54 * k43 * k32
            + k61 * k12 * k54 * k43 * k32
            + k56 * k61 * k12 * k43 * k32
            + k45 * k56 * k61 * k32 * k12
            + k34 * k45 * k56 * k61 * k12
            + k16 * k65 * k54 * k43 * k32
        )
        P3 = (
            k12 * k23 * k65 * k54 * k43
            + k61 * k12 * k23 * k54 * k43
            + k56 * k61 * k12 * k23 * k43
            + k45 * k56 * k61 * k12 * k23
            + k21 * k16 * k65 * k54 * k43
            + k23 * k16 * k65 * k54 * k43
        )
        P4 = (
            k65 * k54 * k12 * k23 * k34
            + k54 * k61 * k12 * k23 * k34
            + k56 * k61 * k12 * k23 * k34
            + k32 * k21 * k16 * k65 * k54
            + k21 * k16 * k65 * k54 * k34
            + k16 * k65 * k54 * k23 * k34
        )
        P5 = (
            k65 * k12 * k23 * k34 * k45
            + k61 * k12 * k23 * k34 * k45
            + k43 * k32 * k21 * k16 * k65
            + k45 * k32 * k21 * k16 * k65
            + k34 * k45 * k21 * k16 * k65
            + k23 * k34 * k45 * k16 * k65
        )
        P6 = (
            k12 * k23 * k34 * k45 * k56
            + k54 * k43 * k32 * k21 * k16
            + k56 * k43 * k32 * k21 * k16
            + k45 * k56 * k32 * k21 * k16
            + k34 * k45 * k56 * k21 * k16
            + k16 * k23 * k34 * k45 * k56
        )
        Sigma = P1 + P2 + P3 + P4 + P5 + P6  # Normalization factor
        return np.array([P1, P2, P3, P4, P5, P6]) / Sigma


class Test_Probability_Calcs:
    @pytest.fixture(scope="class")
    def SP3(k_vals):
        return StateProbs3(k_vals)

    @pytest.fixture(scope="class")
    def SP4(k_vals):
        return StateProbs4(k_vals)

    @pytest.fixture(scope="class")
    def SP4WL(k_vals):
        return StateProbs4WL(k_vals)

    @pytest.fixture(scope="class")
    def SP5WL(k_vals):
        return StateProbs5WL(k_vals)

    @pytest.fixture(scope="class")
    def SP6(k_vals):
        return StateProbs6(k_vals)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=10), min_size=6, max_size=6),
    )
    def test_3_state_probs(self, k_vals, SP3):
        # assign the rates accordingly
        k12, k21, k23, k32, k13, k31 = k_vals
        expected_probs = SP3.return_probs(k12, k21, k23, k32, k13, k31)

        K = np.array([[0, k12, k13], [k21, 0, k23], [k31, k32, 0]])
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # generate the sympy functions for the state probabilities
        sympy_funcs = calculations.calc_state_probs(G, output_strings=True)
        rate_names = ["k12", "k21", "k23", "k32", "k13", "k31"]
        sympy_prob_funcs = expressions.construct_lambda_funcs(
            sympy_funcs=sympy_funcs, rate_names=rate_names
        )

        # use the functions to calculate the state probabilities
        sympy_probs = np.empty(shape=(3,), dtype=float)
        for i in range(3):
            sympy_probs[i] = sympy_prob_funcs[i](k12, k21, k23, k32, k13, k31)

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1]) / 3
        results3 = ode.ode_solver(
            probability_guess, K, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
        )
        ode_probs = results3.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-15)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(sympy_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(sympy_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=12)
        assert_almost_equal(mat_probs, expected_probs, decimal=15)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=10), min_size=8, max_size=8),
    )
    def test_4_state_probs(self, k_vals, SP4):
        # assign the rates accordingly
        k12, k21, k23, k32, k34, k43, k41, k14 = k_vals
        expected_probs = SP4.return_probs(k12, k21, k23, k32, k34, k43, k41, k14)

        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, 0], [0, k32, 0, k34], [k41, 0, k43, 0]]
        )
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # generate the sympy functions for the state probabilities
        sympy_funcs = calculations.calc_state_probs(G, output_strings=True)
        rate_names = ["k12", "k21", "k23", "k32", "k34", "k43", "k41", "k14"]
        sympy_prob_funcs = expressions.construct_lambda_funcs(
            sympy_funcs=sympy_funcs, rate_names=rate_names
        )

        # use the functions to calculate the state probabilities
        sympy_probs = np.empty(shape=(4,), dtype=float)
        for i in range(4):
            sympy_probs[i] = sympy_prob_funcs[i](k12, k21, k23, k32, k34, k43, k41, k14)

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1, 1]) / 4
        ode_results = ode.ode_solver(
            probability_guess, K, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
        )
        ode_probs = ode_results.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-15)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(sympy_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(sympy_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=12)
        assert_almost_equal(mat_probs, expected_probs, decimal=15)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=10), min_size=10, max_size=10),
    )
    def test_4_state_probs_with_leakage(self, k_vals, SP4WL):
        # assign the rates accordingly
        (k12, k21, k23, k32, k34, k43, k41, k14, k24, k42) = k_vals
        expected_probs = SP4WL.return_probs(
            k12, k21, k23, k32, k34, k43, k41, k14, k24, k42
        )

        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, k24], [0, k32, 0, k34], [k41, k42, k43, 0]]
        )
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # generate the sympy functions for the state probabilities
        sympy_funcs = calculations.calc_state_probs(G, output_strings=True)
        rate_names = [
            "k12",
            "k21",
            "k23",
            "k32",
            "k34",
            "k43",
            "k41",
            "k14",
            "k24",
            "k42",
        ]
        sympy_prob_funcs = expressions.construct_lambda_funcs(
            sympy_funcs=sympy_funcs, rate_names=rate_names
        )

        # use the functions to calculate the state probabilities
        sympy_probs = np.empty(shape=(4,), dtype=float)
        for i in range(4):
            sympy_probs[i] = sympy_prob_funcs[i](
                k12, k21, k23, k32, k34, k43, k41, k14, k24, k42
            )

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1, 1]) / 4
        ode_results = ode.ode_solver(
            probability_guess, K, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
        )
        ode_probs = ode_results.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-15)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(sympy_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(sympy_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=12)
        assert_almost_equal(mat_probs, expected_probs, decimal=15)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=10), min_size=12, max_size=12),
    )
    def test_5_state_probs_with_leakage(self, k_vals, SP5WL):
        # assign the rates accordingly
        (k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54) = k_vals
        expected_probs = SP5WL.return_probs(
            k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54
        )

        K = np.array(
            [
                [0, k12, k13, 0, 0],
                [k21, 0, k23, k24, 0],
                [k31, k32, 0, 0, k35],
                [0, k42, 0, 0, k45],
                [0, 0, k53, k54, 0],
            ]
        )
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # generate the sympy functions for the state probabilities
        sympy_funcs = calculations.calc_state_probs(G, output_strings=True)
        rate_names = [
            "k12",
            "k21",
            "k23",
            "k32",
            "k13",
            "k31",
            "k24",
            "k42",
            "k35",
            "k53",
            "k45",
            "k54",
        ]
        sympy_prob_funcs = expressions.construct_lambda_funcs(
            sympy_funcs=sympy_funcs, rate_names=rate_names
        )

        # use the functions to calculate the state probabilities
        sympy_probs = np.empty(shape=(5,), dtype=float)
        for i in range(5):
            sympy_probs[i] = sympy_prob_funcs[i](
                k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54
            )

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1, 1, 1]) / 5
        ode_results = ode.ode_solver(
            probability_guess, K, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
        )
        ode_probs = ode_results.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-15)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(sympy_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(sympy_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=12)
        assert_almost_equal(mat_probs, expected_probs, decimal=15)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)

    @pytest.mark.parametrize(
        "k_vals", [(1e5, 2e-4, 5e-8, 5e-8, 1e5, 2e-4, 6, 4e8, 3e-8, 4e8, 3e-8, 6)]
    )
    def test_5wl_state_probs_bad_apple(self, k_vals, SP5WL):
        # assign the rates accordingly
        (k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54) = k_vals
        expected_probs = SP5WL.return_probs(
            k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54
        )

        K = np.array(
            [
                [0, k12, k13, 0, 0],
                [k21, 0, k23, k24, 0],
                [k31, k32, 0, 0, k35],
                [0, k42, 0, 0, k45],
                [0, 0, k53, k54, 0],
            ]
        )
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1, 1, 1]) / 5
        ode_results = ode.ode_solver(
            probability_guess, K, t_max=1e4, tol=1e-16, atol=1e-17, rtol=1e-13
        )
        ode_probs = ode_results.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-12)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=3)
        assert_almost_equal(mat_probs, expected_probs, decimal=8)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=10), min_size=12, max_size=12),
    )
    def test_6_state_probs(self, k_vals, SP6):
        # assign the rates accordingly
        (k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16) = k_vals
        expected_probs = SP6.return_probs(
            k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16
        )

        K = np.array(
            [
                [0, k12, 0, 0, 0, k16],
                [k21, 0, k23, 0, 0, 0],
                [0, k32, 0, k34, 0, 0],
                [0, 0, k43, 0, k45, 0],
                [0, 0, 0, k54, 0, k56],
                [k61, 0, 0, 0, k65, 0],
            ]
        )
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the state probabilities using KDA
        kda_probs = calculations.calc_state_probs(G)

        # generate the sympy functions for the state probabilities
        sympy_funcs = calculations.calc_state_probs(G, output_strings=True)
        rate_names = [
            "k12",
            "k21",
            "k23",
            "k32",
            "k34",
            "k43",
            "k45",
            "k54",
            "k56",
            "k65",
            "k61",
            "k16",
        ]
        sympy_prob_funcs = expressions.construct_lambda_funcs(
            sympy_funcs=sympy_funcs, rate_names=rate_names
        )

        # use the functions to calculate the state probabilities
        sympy_probs = np.empty(shape=(6,), dtype=float)
        for i in range(6):
            sympy_probs[i] = sympy_prob_funcs[i](
                k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16
            )

        # use the ODE integrator to calculate the state probabilities
        probability_guess = np.array([1, 1, 1, 1, 1, 1]) / 6
        ode_results = ode.ode_solver(
            probability_guess, K, t_max=1e2, tol=1e-12, atol=1e-16, rtol=1e-13
        )
        ode_probs = ode_results.y.T[-1]

        # use the SVD solver to calculate the state probabilities
        svd_probs = svd.svd_solver(K, tol=1e-15)

        # use the matrix solver to calculate the state probabilities
        mat_probs = svd.matrix_solver(K)

        # make sure all probabilities sum to 1
        assert_allclose(np.sum(kda_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(sympy_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(svd_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(mat_probs), 1.0, rtol=1e-09, atol=1e-12)
        assert_allclose(np.sum(ode_probs), 1.0, rtol=1e-09, atol=1e-12)

        # compare all probabilities to the expected outcome
        assert_almost_equal(kda_probs, expected_probs, decimal=15)
        assert_almost_equal(sympy_probs, expected_probs, decimal=15)
        assert_almost_equal(svd_probs, expected_probs, decimal=12)
        assert_almost_equal(mat_probs, expected_probs, decimal=15)
        assert_almost_equal(ode_probs, expected_probs, decimal=10)


class Test_Flux_Diagrams:
    def test_generate_all_flux_diags(self):
        # Test just to verify that generating all flux diagrams returns the
        # same result as generating them one at a time
        # For this test we are using a 4 state model with leakage since it
        # has 2 flux diagrams and 1 all-node cycles (which returns None)

        # assign all rates a value of 1 since we will
        # not be calculating things anyways
        (k12, k21, k23, k32, k34, k43, k41, k14, k24, k42) = np.ones(
            shape=(10,), dtype=int
        )
        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, k24], [0, k32, 0, k34], [k41, k42, k43, 0]]
        )
        # create the graph from the rate matrix
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # get all the cycles in G
        all_cycles = graph_utils.find_all_unique_cycles(G)
        # make sure only 3 cycles are found
        assert len(all_cycles) == 3
        # for each cycle in the diagram, get the flux diagrams
        flux_diags = []
        for cycle in all_cycles:
            flux_diags.append(diagrams.generate_flux_diagrams(G, cycle))
        # make sure first case returns None (for all-node cycle)
        assert flux_diags[0] == None
        # make sure we get 2 flux diagrams for each 3-node cycle
        assert len(flux_diags[1]) == 2
        assert len(flux_diags[2]) == 2
        # use the KDA built-in function to generate all flux diagrams for G
        all_flux_diags = diagrams.generate_all_flux_diagrams(G)
        # flatten the lists of diagrams and skip the None case for method 1
        all_diags_method_1 = [diag for diags in flux_diags[1:] for diag in diags]
        all_diags_method_2 = []
        for diags in all_flux_diags:
            if diags is not None:
                for diag in diags:
                    all_diags_method_2.append(diag)

        # for all 4 flux diagrams, make sure their edges match
        assert all_diags_method_1[0].edges == all_diags_method_2[0].edges
        assert all_diags_method_1[1].edges == all_diags_method_2[1].edges
        assert all_diags_method_1[2].edges == all_diags_method_2[2].edges
        assert all_diags_method_1[3].edges == all_diags_method_2[3].edges

    @settings(deadline=None)
    @given(
        k_vals=st.lists(
            st.floats(min_value=1, max_value=100), min_size=10, max_size=10
        ),
    )
    def test_calc_net_cycle_flux_from_diags_4WL(self, k_vals):
        # assign rates generated by hypothesis and create rate matrix
        (k12, k21, k23, k32, k34, k43, k41, k14, k24, k42) = k_vals
        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, k24], [0, k32, 0, k34], [k41, k42, k43, 0]]
        )
        # create graph and assign edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # generate the directional partial diagrams
        dirpar_edges = diagrams.generate_directional_diagrams(
            G, return_edges=True
        )
        # pick one of the 3-node cycles and the CCW direction
        cycle = [0, 1, 3]
        order = [0, 1]
        # calculate the net cycle flux
        net_cycle_flux = calculations.calc_net_cycle_flux_from_diags(
            G,
            dirpar_edges=dirpar_edges,
            cycle=cycle,
            order=order,
            output_strings=False,
        )
        # generate the net cycle flux function
        net_cycle_flux_sympy_func = calculations.calc_net_cycle_flux_from_diags(
            G,
            dirpar_edges=dirpar_edges,
            cycle=cycle,
            order=order,
            output_strings=True,
        )
        # convert sympy function into lambda function
        rate_names = [
            "k12",
            "k21",
            "k23",
            "k32",
            "k34",
            "k43",
            "k41",
            "k14",
            "k24",
            "k42",
        ]
        net_cycle_flux_lambda_func = expressions.construct_lambda_funcs(
            net_cycle_flux_sympy_func, rate_names=rate_names
        )
        # use the lambda function to calculate the net cycle flux
        sympy_net_cycle_flux = net_cycle_flux_lambda_func(
            k12, k21, k23, k32, k34, k43, k41, k14, k24, k42
        )
        # compare direct calculation to sympy function calculation
        assert_allclose(net_cycle_flux, sympy_net_cycle_flux, atol=1e-14, rtol=1e-14)

    @settings(deadline=None)
    @given(
        k_vals=st.lists(st.floats(min_value=1, max_value=1e2), min_size=6, max_size=6),
    )
    def test_calc_net_cycle_flux_3(self, k_vals):
        # assign rates generated by hypothesis and create rate matrix
        (k12, k21, k23, k32, k13, k31) = k_vals
        K = np.array([[0, k12, k13], [k21, 0, k23], [k31, k32, 0]])
        # create graph and assign edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # use the only cycle in the diagram and the CCW direction
        cycle = [0, 2, 1]
        order = [0, 1]
        # calculate the net cycle flux
        net_cycle_flux = calculations.calc_net_cycle_flux(
            G, cycle=cycle, order=order, output_strings=False
        )
        # generate the sympy net cycle flux function
        sympy_net_cycle_flux_func = calculations.calc_net_cycle_flux(
            G, cycle=cycle, order=order, output_strings=True
        )
        # make sure the sympy function agrees with the known solution
        expected_func = "(k12*k23*k31 - k13*k21*k32)/(k12*k23 + k12*k31 + k12*k32 + k13*k21 + k13*k23 + k13*k32 + k21*k31 + k21*k32 + k23*k31)"
        assert str(sympy_net_cycle_flux_func) == expected_func

        # make sure the calculated net cycle flux agrees with the known solution
        expected_value = (k12 * k23 * k31 - k13 * k21 * k32) / (
            k12 * k23
            + k12 * k31
            + k12 * k32
            + k13 * k21
            + k13 * k23
            + k13 * k32
            + k21 * k31
            + k21 * k32
            + k23 * k31
        )
        assert_allclose(net_cycle_flux, expected_value, atol=1e-14, rtol=1e-14)

    @pytest.mark.parametrize(
        "edge_list, N, expected_truth_value",
        [([(0, 1, 0), (1, 2, 0)], 3, False), ([(0, 1, 0), (1, 0, 0)], 2, False)],
    )
    def test_flux_edge_conditions(self, edge_list, N, expected_truth_value):
        truth_value = diagrams._flux_edge_conditions(
            edge_list=edge_list, n_flux_edges=N
        )
        assert truth_value == expected_truth_value

    @pytest.mark.parametrize(
        "adj_matrix, expected_cycle_count, expected_flux_diag_count",
        [
            (
                # simple 3-state model
                [
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ],
                # has no flux diagrams because there is only 1 cycle,
                # and that cycle contains all of the nodes in the diagram
                1,
                0,
            ),
            (
                # simple 4-state model
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                ],
                # has no flux diagrams for the same reason as the
                # 3-state model
                1,
                0,
            ),
            (
                # 4-state model with leakage (simplified version of
                # Hill's 5-state model below)
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 1],
                    [0, 1, 0, 1],
                    [1, 1, 1, 0],
                ],
                # has 3 cycles, 2 of which are 3-state cycles which each
                # have 2 flux diagrams associated with them
                3,
                4,
            ),
            (
                # 5-state model with leakage, used extensively in T.L. Hill's
                # "Free Energy Transduction and Biochemical Cycle Kinetics"
                [
                    [0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 0, 1, 1, 0],
                ],
                # has 3 cycles, where 2 cycles have corresponding flux diagrams:
                # a 3-state cycle with 3 flux diagrams, and a 4-state cycle
                # with 2 flux diagrams
                3,
                5,
            ),
            (
                # 8-state model with leakage used for flux diagram example in
                # T.L. Hill's "Free Energy Transduction and Biochemical Cycle Kinetics"
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
                # there are 6 cycles in this model, 5 of which contain many flux
                # diagrams each. These are shown on page 51 of the above mentioned
                # book, and total to 37 (not including the all-node cycle g)
                6,
                37,
            ),
            (
                # 8-state model
                [
                    [0, 1, 1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 1, 0],
                ],
                # model has 28 unique cycles with a total of 402 flux diagrams.
                # this model is unique because it has many cycles that contain
                # all nodes and thus do not have any flux diagrams associated
                # with them.
                28,
                402,
            ),
        ],
    )
    def test_flux_diagram_counts(self, adj_matrix, expected_cycle_count, expected_flux_diag_count):
        # generate flux diagrams for different models and verify
        # the number of flux diagrams generated is correct

        A = np.array(adj_matrix)
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, A)

        actual_flux_diag_count = 0
        all_cycles = graph_utils.find_all_unique_cycles(G)
        # check the number of cycles found is correct
        assert len(all_cycles) == expected_cycle_count

        for cycle in all_cycles:
            flux_diagrams = diagrams.generate_flux_diagrams(G, cycle)
            if flux_diagrams:
                for flux_diag in flux_diagrams:
                    # check that flux diagrams have the same nodes
                    # as the input diagram
                    assert sorted(flux_diag.nodes()) == sorted(G.nodes())
                    # collect the cycles in the flux diagram, excluding the
                    # simplest 2-node cycles
                    cycles = [c for c in nx.simple_cycles(flux_diag) if len(c) > 2]
                    # check that there are only 2 cycles, which should be the
                    # same cycle just in the forward/reverse directions
                    assert len(cycles) == 2
                    # check that the forward/reverse cycles are the same
                    assert sorted(cycles[0]) == sorted(cycles[1])
                    actual_flux_diag_count += 1

        assert actual_flux_diag_count == expected_flux_diag_count


class Test_Misc_Funcs:
    @settings(deadline=None)
    @given(
        k_vals=st.lists(
            st.floats(min_value=1, max_value=100), min_size=10, max_size=10
        ),
    )
    def test_sigma_K_4WL(self, k_vals):
        # assign rates generated by hypothesis and create rate matrix
        (k12, k21, k23, k32, k34, k43, k41, k14, k24, k42) = k_vals
        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, k24], [0, k32, 0, k34], [k41, k42, k43, 0]]
        )
        # create graph and assign edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # use one of the 3-node cycles in the diagram
        cycle = [0, 1, 3]
        # generate the flux diagrams to use in sigma_K calculations
        flux_diagrams = diagrams.generate_flux_diagrams(G, cycle)

        # calculate sigma_K value
        sigma_K_val = calculations.calc_sigma_K(
            G, cycle, flux_diagrams, output_strings=False
        )
        # check that the sigma_K value agrees with the known solution
        expected_value = k32 + k34
        assert sigma_K_val == expected_value

        # check that the sigma_K function agrees with the known solution
        sigma_K_func = calculations.calc_sigma_K(
            G, cycle, flux_diagrams, output_strings=True
        )
        expected_func = "k32+k34"
        assert sigma_K_func == expected_func

    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        k_vals=st.lists(
            st.floats(min_value=1, max_value=100), min_size=10, max_size=10
        ),
    )
    @pytest.mark.parametrize(
        "cycle, cycle_order, expected_func",
        [
            ([0, 3, 2, 1], [3, 0], "log(k12*k23*k34*k41/(k14*k21*k32*k43))",),
            ([0, 3, 1], [3, 0], "log(k12*k24*k41/(k14*k21*k42))",),
            ([1, 3, 2], [3, 1], "log(k23*k34*k42/(k24*k32*k43))",),
        ],
    )
    def test_thermo_force_4WL(self, k_vals, cycle, cycle_order, expected_func):
        # assign rates generated by hypothesis and create rate matrix
        (k12, k21, k23, k32, k34, k43, k41, k14, k24, k42) = k_vals
        K = np.array(
            [[0, k12, 0, k14], [k21, 0, k23, k24], [0, k32, 0, k34], [k41, k42, k43, 0]]
        )
        # create graph and assign edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # calculate the thermodynamic force
        thermo_force_val = calculations.calc_thermo_force(
            G, cycle, cycle_order, output_strings=False
        )
        if cycle == [0, 3, 2, 1]:
            expected_value = np.log(k12 * k23 * k34 * k41 / (k14 * k21 * k32 * k43))
        elif cycle == [0, 3, 1]:
            expected_value = np.log(k12 * k24 * k41 / (k14 * k21 * k42))
        elif cycle == [1, 3, 2]:
            expected_value = np.log(k23 * k34 * k42 / (k24 * k32 * k43))

        # check that the calculated value agrees with the expected value
        assert_allclose(thermo_force_val, expected_value, atol=1e-14, rtol=1e-14)
        # generate the equation for the thermodynamic force
        thermo_force_func = calculations.calc_thermo_force(
            G, cycle, cycle_order, output_strings=True
        )
        # check that the generated expression agrees with the expected function
        assert str(thermo_force_func) == str(thermo_force_func)


class Test_Diagram_Generation:
    @pytest.mark.parametrize(
        "K, expected_pars, expected_dirpars",
        [
            (np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), 3, 9),
            (np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]), 4, 16),
            (np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]), 8, 32),
            (
                np.array(
                    [
                        [0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0],
                        [1, 1, 0, 0, 1],
                        [0, 1, 0, 0, 1],
                        [0, 0, 1, 1, 0],
                    ]
                ),
                11,
                55,
            ),
            (
                np.array(
                    [
                        [0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0],
                    ]
                ),
                6,
                36,
            ),
        ],
    )
    def test_diagram_counts(self, K, expected_pars, expected_dirpars):
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # generate the partial diagrams and verify
        # they agree with the expected value
        partials = diagrams.generate_partial_diagrams(G, return_edges=False)
        assert len(partials) == expected_pars
        # generate the directional partial diagrams and verify
        # they agree with the expected value
        dirpars = diagrams.generate_directional_diagrams(G, return_edges=False)
        assert len(dirpars) == expected_dirpars
        # count the number of partial diagrams
        # and verify they agree with the expected value
        n_pars = diagrams.enumerate_partial_diagrams(G)
        assert n_pars == expected_pars

    @pytest.mark.parametrize(
        "K",
        [
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),
            np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]),
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [0, 0, 1, 1, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                ]
            ),
        ],
    )
    def test_partial_diagram_comparison(self, K):
        # compare the edge lists generated by `generate_partial_diagrams`
        # for both the `return_edges=True` and `return_edges=False` cases

        # generate the input diagram
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)

        # generate the partial diagrams and edge lists separately
        partial_edges = diagrams.generate_partial_diagrams(G, return_edges=True)
        partial_diags = diagrams.generate_partial_diagrams(G, return_edges=False)

        # collect the edges from each partial diagram generated
        partial_diag_edges = []
        for diag in partial_diags:
            partial_diag_edges.append(diag.edges())
        partial_diag_edges = np.asarray(partial_diag_edges)

        # sort the edge tuples (since order doesn't matter) first, then
        # sort the edge lists for each diagram so they are easy to compare
        for axis in [2, 1]:
            partial_edges = np.sort(partial_edges, axis=axis)
            partial_diag_edges = np.sort(partial_diag_edges, axis=axis)

        assert_array_equal(partial_edges, partial_diag_edges)

    @pytest.mark.parametrize("n_states", [3, 4, 5, 6])
    def test_max_connected_diagram_counts(self, n_states):
        # create a maximally connected graph
        K = np.ones(shape=(n_states, n_states))
        np.fill_diagonal(K, val=0)
        # generate the diagram and edges
        G = nx.MultiDiGraph()
        graph_utils.generate_edges(G, K)
        # generate the partial diagrams and verify
        # they agree with the expected value
        partial_edges = diagrams.generate_partial_diagrams(G, return_edges=True)
        expected_pars = n_states ** (n_states - 2)
        assert len(partial_edges) == expected_pars
        # generate the directional partial diagrams and verify
        # they agree with the expected value
        dirpar_edges = diagrams.generate_directional_diagrams(
            G, return_edges=True
        )
        expected_dirpars = n_states ** (n_states - 1)
        assert len(dirpar_edges) == expected_dirpars
        # count the number of partial diagrams
        # and verify they agree with the expected value
        n_pars = diagrams.enumerate_partial_diagrams(G)
        assert n_pars == expected_pars


def test_CycleError():
    # function just to check that we can raise a CycleError
    # without an input message
    with pytest.raises(CycleError):
        err = CycleError()
        raise err
    # also try initializing CycleError with a message
    err = CycleError("No cycle found")


def test_get_ordered_cycle():
    # use rate matrix for a 4-state model with leakage
    K = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]])
    # generate the diagram and edges
    G = nx.MultiDiGraph()
    graph_utils.generate_edges(G, K)

    # run for a valid 3-node cycle
    calculations._get_ordered_cycle(G, [0, 1, 3])
    # run for a valid all-node cycle
    calculations._get_ordered_cycle(G, [0, 1, 2, 3])
    # run for a case where input nodes are not within diagram
    with pytest.raises(CycleError):
        calculations._get_ordered_cycle(G, [0, 5, 9, 4])
    # run for an invalid cycle case
    with pytest.raises(CycleError):
        calculations._get_ordered_cycle(G, [0, 1, 2])


def test_get_ordered_cycle_all_node_cycles():
    # use rate matrix for maximally-connected 5-state diagram
    K = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    # generate the diagram and edges
    G = nx.MultiDiGraph()
    graph_utils.generate_edges(G, K)

    nx_version = nx.__version__
    major = int(nx_version[0])
    minor = int(nx_version[2])
    if (major, minor) < (3, 1):
        valid_cycle = [0, 4, 3, 2, 1]
        invalid_cycle = [0, 1, 2, 3, 4]
    else:
        # order must be changed due to updates in
        # `nx.simple_cycles()`. For more details
        # see https://github.com/Becksteinlab/kda/issues/71
        valid_cycle = [0, 1, 2, 3, 4]
        invalid_cycle = [0, 4, 3, 2, 1]
    # run for a valid all-node cycle
    calculations._get_ordered_cycle(G, valid_cycle)
    # run for case where there are more nodes than possible in a cycle
    with pytest.raises(CycleError):
        calculations._get_ordered_cycle(G, [0, 1, 2, 3, 4, 5])
    # run for an all-node case where the order is incorrect
    with pytest.raises(CycleError):
        calculations._get_ordered_cycle(G, invalid_cycle)


def test_retrieve_rate_matrix():
    # regression test for `graph_utils.retrieve_rate_matrix()`
    # checks that input and output rate matrices are the same

    # create 5-state model with all unique values
    K = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 0, 6, 7, 8],
            [9, 10, 0, 11, 12],
            [13, 14, 15, 0, 16],
            [17, 18, 19, 20, 0],
        ]
    )
    # initialize graph object
    G = nx.MultiDiGraph()
    # use KDA utility to generate the edges from K
    graph_utils.generate_edges(G, K)
    # now retrieve K from the diagram
    K_new = graph_utils.retrieve_rate_matrix(G)
    # check that arrays are the same
    assert_array_equal(K, K_new)


def test_add_attributes():
    K = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 0, 6, 7, 8],
            [9, 10, 0, 11, 12],
            [13, 14, 15, 0, 16],
            [17, 18, 19, 20, 0],
        ]
    )
    # initialize graph object
    G = nx.MultiDiGraph()
    # use KDA utility to generate the edges from K
    graph_utils.generate_edges(G, K)

    # calculate the state probabilities using KDA
    kda_probs = calculations.calc_state_probs(G)

    node_data = kda_probs
    node_label = "probability"
    graph_utils.add_node_attribute(G, data=node_data, label=node_label)
    for i in range(G.number_of_nodes()):
        assert G.nodes[i][node_label] == kda_probs[i]

    graph_data = K
    graph_label = "Graph rate matrix"
    graph_utils.add_graph_attribute(G, data=graph_data, label=graph_label)
    assert np.all(G.graph[graph_label] == K)


def test_ccw():
    K = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 0, 6, 7, 8],
            [9, 10, 0, 11, 12],
            [13, 14, 15, 0, 16],
            [17, 18, 19, 20, 0],
        ]
    )
    # initialize graph object
    G = nx.MultiDiGraph()
    # use KDA utility to generate the edges from K
    graph_utils.generate_edges(G, K)

    cycles = graph_utils.find_all_unique_cycles(G)

    for cycle in cycles:
        # TODO: should definitely find out a way
        # to make some assertions here. The problem is defining what CCW means
        # is problem-dependent, and you have to choose node positions to make
        # that decision, which would appear arbitrary here.
        new_cycle = graph_utils.get_ccw_cycle(cycle, order=cycle[:2])
        # since we feed in an order that is just the first 2 nodes of the
        # cycle, this should return the same cycle
        assert cycle == new_cycle

    with pytest.raises(CycleError):
        graph_utils.get_ccw_cycle(cycles[0], [4, 5, 6])
