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
        ``Nx1`` matrix of initial state probabilities.
    K : ndarray
        Adjacency matrix for the kinetic diagram where each element
        ``kij`` is the edge weight (i.e. transition rate constant).
        For example, for a 2-state model with ``k12=3`` and ``k21=4``,
        ``K=[[0, 3], [4, 0]]``.
    t_max : int
        Length of time for integrator to run in seconds.
    method : str
        Integration method used in ``scipy.integrate.solve_ivp()``. Default is
        ``"LSODA"`` since it has automatic stiffness detection, and generally
        requires much less run time to reach convergence than ``"RK45"``.
    tol : float (optional)
        Tolerance value used as convergence criteria. Once all dp/dt values for
        each state are less than the tolerance the integrator will terminate.
        Default is ``1e-16``.
    options
        Options passed to ``scipy.integrate.solve_ivp()``.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at ``t``.

    Notes
    -----
    For all parameters and returns, view the ``SciPy.integrate.solve_ivp()``
    documentation (see `here <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.integrate.solve_ivp.html>`_).
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
