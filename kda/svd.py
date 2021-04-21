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
"""

import numpy as np


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
    N = len(K)  # get number of states
    Kc = K.copy()  # Make a copy of input matrix K
    np.fill_diagonal(Kc, 0)  # fill the diagonal elements with zeros
    Kc = Kc.T  # take the transpose
    for i in range(N):
        Kc[i, i] = -Kc[:, i].sum(
            axis=0
        )  # set the diagonal elements equal to the negative sum of the columns
    prob_norm = np.ones(N)  # create array of ones
    Kcs = np.vstack((Kc, prob_norm))  # stack ODE equations with probability equation
    U, w, VT = np.linalg.svd(
        Kcs, full_matrices=False
    )  # use svd() to generate U, w, and V.T matrices
    singular_vals = np.abs(w) < tol  # find any singular values in w matrix
    inv_w = 1 / w  # Take the inverse of the w matrix
    inv_w[singular_vals] = 0  # Set any singular values to zero
    Kcs_inv = VT.T.dot(np.diag(inv_w)).dot(U.T)  # construct the pseudo inverse of Kcs
    # create steady state solution matrix (pdot = 0), add additional entry for probability equation
    pdot = np.zeros(N + 1)
    # set last value to 1 for probability normalization
    pdot[-1] = 1
    # dot Kcs and pdot matrices together
    state_probs = Kcs_inv.dot(pdot)
    return state_probs
