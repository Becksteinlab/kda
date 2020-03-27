# Nikolaus Awtrey
# Computational model of biochemical kinetic cycle steady state probabilities

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

#===================================================
#==== Functions ====================================
#===================================================

def integrate_prob_ODE(P, K, t_max, max_step):
    """
    Integrates state probabilities to find steady state probabilities.
    Parameters
    ----------
    P : array
    K : array
        'NxN' matrix, where N is the number of states. Element i, j represents
        the rate constant from state i to state j. Diagonal elements should be
        zero, but does not have to be in input k
        matrix.
    N : int
        Number of states
    t_max : int
    max_step : int
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

    def func(t, y):
        """
        y = [p1, p2, p3, ... , pn]
        """
        return np.dot(k, y)

    k = convert_K(K)
    time = (0, t_max)
    y0 = np.array(P, dtype=np.float64)
    return scipy.integrate.solve_ivp(func, time, y0, max_step=max_step)

def plot_ODE_probs(results):
    N = int(len(results.y))
    time = results.t
    p_time_series = results.y[:N]
    p_tot = p_time_series.sum(axis=0)
    fig1 = plt.figure(figsize = (8, 7), tight_layout=True)
    ax = fig1.add_subplot(111)
    for i in range(N):
        ax.plot(time, p_time_series[i], '-', lw=2, label='p{}, final = {}'.format(i+1, p_time_series[i][-1]))
    ax.plot(time, p_tot, '--', lw=2, color="black", label="p_tot, final = {}".format(p_tot[-1]))
    ax.set_title("State Probabilities for {} State Model".format(N))
    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Time (s)")
    ax.legend(loc='best')
    plt.show()
