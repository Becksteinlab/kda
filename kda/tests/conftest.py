# Nikolaus Awtrey
# Beckstein Lab
# Department of Physics
# Arizona State University
#
# Kinetic Diagram Analysis Testing Fixtures

import pytest
import numpy as np


@pytest.fixture(scope="class")
def state_probs_3_state():
	def _state_probs(k12, k21, k23, k32, k13, k31):
	    P1 = k23 * k31 + k32 * k21 + k31 * k21
	    P2 = k13 * k32 + k32 * k12 + k31 * k12
	    P3 = k13 * k23 + k12 * k23 + k21 * k13
	    Sigma = P1 + P2 + P3  # Normalization factor
	    return np.array([P1, P2, P3]) / Sigma
	return _state_probs


@pytest.fixture(scope="class")
def state_probs_4_state():
	def _state_probs(k12, k21, k23, k32, k34, k43, k41, k14):
	    P1 = k43 * k32 * k21 + k23 * k34 * k41 + k21 * k34 * k41 + k41 * k32 * k21
	    P2 = k12 * k43 * k32 + k14 * k43 * k32 + k34 * k41 * k12 + k32 * k41 * k12
	    P3 = k43 * k12 * k23 + k23 * k14 * k43 + k21 * k14 * k43 + k41 * k12 * k23
	    P4 = k12 * k23 * k34 + k14 * k23 * k34 + k34 * k21 * k14 + k32 * k21 * k14
	    Sigma = P1 + P2 + P3 + P4  # Normalization factor
	    return np.array([P1, P2, P3, P4]) / Sigma
	return _state_probs


@pytest.fixture(scope="class")
def state_probs_4wl():
	def _state_probs(k12, k21, k23, k32, k34, k43, k41, k14, k24, k42):
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
	return _state_probs


@pytest.fixture(scope="class")
def state_probs_5wl():
	def _state_probs(k12, k21, k23, k32, k13, k31, k24, k42, k35, k53, k45, k54):
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
	return _state_probs


@pytest.fixture(scope="class")
def state_probs_6_state():
	def _state_probs(k12, k21, k23, k32, k34, k43, k45, k54, k56, k65, k61, k16):
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
	return _state_probs
