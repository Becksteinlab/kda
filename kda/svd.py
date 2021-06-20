# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Kinetic Diagram Analysis: Singular Value Decomposition
=========================================================================
This file contains a host of functions aimed at the analysis of biochemical
kinetic diagrams via singular value decomposition.

.. autofunction:: svd_solver
.. autofunction:: matrix_solver
"""

import numpy as np
import math


def svd_solver(K, tol=1e-12):
    """
    Calculates the steady-state probabilities for an N-state model using
    singular value decomposition.

    Parameters
    ----------
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input K matrix.
    tol : float (optional)
        Tolerance used for singular value determination. Values are considered
        singular if they are less than the input tolerance. Default is 1e-12.

    Returns
    -------
    state_probs : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    """
    # get number of states
    N = K.shape[0]
    # Make a copy of input matrix K
    Kc = K.copy()
    # fill the diagonal elements with zeros
    np.fill_diagonal(Kc, 0)
    # take the transpose
    Kc = Kc.T
    # set the diagonal elements equal to the negative sum of the columns
    for i in range(N):
        Kc[i, i] = -math.fsum(Kc[:, i])
    # create array of ones
    prob_norm = np.ones(N, dtype=float)
    # stack ODE equations with probability equation
    Kcs = np.vstack((Kc, prob_norm))
    # use svd() to generate U, w, and V.T matrices
    U, w, VT = np.linalg.svd(Kcs, full_matrices=False)
    # find any singular values in w matrix
    singular_vals = np.abs(w) < tol
    # Take the inverse of the w matrix
    inv_w = 1 / w
    # Set any singular values to zero
    inv_w[singular_vals] = 0
    # construct the pseudo inverse of Kcs
    Kcs_inv = VT.T.dot(np.diag(inv_w)).dot(U.T)
    # create steady state solution matrix (pdot = 0), add additional entry for probability equation
    pdot = np.zeros(N + 1, dtype=float)
    # set last value to 1 for probability normalization
    pdot[-1] = 1.0
    # dot Kcs and pdot matrices together
    state_probs = np.matmul(Kcs_inv, pdot, dtype=float)
    return state_probs


def _get_linearly_dependent_row_index(matrix, tol=1e-12):
    """
    Uses singular values found by SVD to get index for linearly dependent
    vector(s) in input matrix.
    """
    # input matrix into SVD
    w = np.linalg.svd(matrix, full_matrices=False)[1]
    # values are considered singular if they are smaller than the tolerance
    singular_vals = np.abs(w) < tol
    # get the indices for any singular values
    ld_inds = np.nonzero(singular_vals)[0]
    if ld_inds.size == 0:
        # if there are no linearly dependent vectors detected using the cutoff
        # select the row with the smallest value
        idx = np.nonzero(w == w.min())[0]
        return idx
    elif ld_inds.size == 1:
        # if there is exactly 1 linearly dependent vector detected, replace it
        # select that row for replacement
        return ld_inds[0]
    elif ld_inds.size > 1:
        raise ValueError(f"{ld_inds.size} linearly dependent equations detected.")


def matrix_solver(K):
    """
    Calculates the steady-state probabilities for an N-state model using
    a standard matrix solver.

    Parameters
    ----------
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input K matrix.

    Returns
    -------
    state_probs : NumPy array
        Array of state probabilities for N states, [p1, p2, p3, ..., pN].
    """
    # get number of states
    N = K.shape[0]
    # Make a copy of input matrix K
    Kc = K.copy()
    # fill the diagonal elements with zeros
    np.fill_diagonal(Kc, 0)
    # take the transpose
    Kc = Kc.T
    # set the diagonal elements equal to the negative sum of the columns
    for i in range(N):
        Kc[i, i] = -math.fsum(Kc[:, i])
    # find linearly dependent row/equation, if one exists
    ld_row_idx = _get_linearly_dependent_row_index(matrix=Kc)
    # replace the last equation with the probability sum equation
    # (p1 + p2 + ... + pN = 1) by setting the values in the
    # bottom row equal to 1
    Kc[ld_row_idx, :] = np.ones(N, dtype=float)
    # create probability time-derivative array
    pdot = np.zeros(N, dtype=float)
    # set last value equal to one since
    # the probabilities are normalized to 1
    pdot[ld_row_idx] = 1
    # calculate the solution
    state_probs = np.linalg.solve(Kc, pdot)
    return state_probs
