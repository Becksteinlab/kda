# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Author: Nikolaus C. Awtrey
#
"""
Differential Equation Solvers
=========================================================================
This file contains a host of functions aimed at the analysis of biochemical
kinetic diagrams via differential equations.

.. autofunction:: ode_solver
"""


import numpy as np
import scipy.integrate


def ode_solver(P, K, t_max, method="LSODA", tol=1e-16, **options):
    """
    Integrates state probability ODE's to find steady state probabilities.

    Parameters
    ----------
    P : array
        'Nx1' matrix of initial state probabilities.
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input k
        matrix.
    t_max : int
        Length of time for integrator to run, in seconds.
    method : str
        Integration method used in `scipy.integrate.solve_ivp()`. Default is
        LSODA since it has automatic stiffness detection, and generally
        requires much less run time to reach convergence than RK45.
    tol : float (optional)
        Tolerance value used as convergence criteria. Once all dp/dt values for
        each state are less than the tolerance the integrator will terminate.
        Default is 1e-16.
    options
        Options passed to scipy.integrate.solve_ivp().

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at t.

    Note:
    For all parameters and returns, view the SciPy.integrate.solve_ivp()
    documentation: "https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.integrate.solve_ivp.html"
    """

    def convert_K(K):
        """
        Sets the diagonal elements of the input k matrix to zero, then takes the
        transpose, and lastly sets the diagonals equal to the negative of the
        sum of the column elements.
        """
        N = len(K)
        np.fill_diagonal(K, 0)
        K = K.T
        for i in range(N):
            K[i, i] = -K[:, i].sum(axis=0)
        return K

    def KdotP(t, y):
        """
        y = [p1, p2, p3, ... , pn]
        """
        return np.matmul(k, y, dtype=np.float64)

    def terminate(t, y):
        y_prime = np.matmul(k, y, dtype=np.float64)
        if all(elem < tol for elem in y_prime):
            print(f"\n kda.ode.ode_solver() reached convergence at t={t}\n")
            return False
        else:
            return True

    terminate.terminal = True
    k = convert_K(K)
    y0 = np.array(P, dtype=np.float64)
    solution = scipy.integrate.solve_ivp(
        fun=KdotP,
        t_span=(0, t_max),
        y0=y0,
        t_eval=np.linspace(0, t_max, 500),
        method=method,
        events=[terminate],
        **options,
    )
    return solution
