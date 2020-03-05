# Nikolaus Awtrey
# PHY495 Project Research
# September 10, 2019 - September 17, 2019
# Computational model of the six states of the transporter cycle

import numpy as np
import scipy.integrate

#===================================================
#==== Simulator ====================================
#===================================================

def probability_integrator(p_list, k_off, k_on, k_conf, t_max, max_step, AB_flux=None):
    """
    Parameters
    ----------
    p_list : array
    k_off : float
    k_on : float
    k_conf : float
    t_max : int
    max_step : int
    AB_flux : bool
    """

    def kon(C_x):
        """
        Concentration-dependent rate constant calculation
        """
        return k_on*C_x

    # Function to integrate
    def func(t, y):
        """
        y = [p1, p2, p3, p4, p5, p6,
             CA_in, CA_out, CB_in, CB_out,
             dp1, dp2, dp3, dp4, dp5, dp6,
             dCA_in, dCA_out, dCB_in, dCB_out]
        """
        # p1, p2, p3, p4, p5, p6
        p = y[:6]
        # [A_in], [A_out], [B_in], [B_out]
        CAB = y[6:10]
        # dp1/dt, dp2/dt, dp3/dt, dp4/dt, dp5/dt, dp6/dt
        dp = y[10:16]
        # d[A_in]/dt, d[A_out]/dt, d[B_in]/dt, d[B_out]/dt
        dCAB = y[16:]
        # Update dpi/dt's for all i
        dp[0] = k_off*p[5] - kon(CAB[3])*p[0] + k_off*p[1] - kon(CAB[1])*p[0]
        dp[1] = kon(CAB[1])*p[0] - k_off*p[1] + k_conf*p[2] - k_conf*p[1]
        dp[2] = k_conf*p[1] - k_conf*p[2] + kon(CAB[0])*p[3] - k_off*p[2] 
        dp[3] = p[2]*k_off - kon(CAB[0])*p[3] + p[4]*k_off - kon(CAB[2])*p[3]
        dp[4] = kon(CAB[2])*p[3] - k_off*p[4] + k_conf*p[5] - k_conf*p[4]
        dp[5] = k_conf*p[4] - k_conf*p[5] + kon(CAB[3])*p[0] - k_off*p[5]

        if AB_flux == False:
            # Case 1: only state probabilities change
            dCAB[0] = 0
            dCAB[1] = 0
            dCAB[2] = 0
            dCAB[3] = 0
        elif AB_flux == True:
            # Case 2: state probabilities and A/B concentrations change
            # Update d[A_in]/dt
            dCAB[0] = p[0]*kon(CAB[1]) - p[1]*k_off
            dCAB[1] = p[4]*kon(CAB[0]) - p[3]*k_off
            # Update d[B_in]/dt
            dCAB[2] = p[0]*kon(CAB[3]) - p[5]*k_off
            dCAB[3] = p[2]*kon(CAB[2]) - p[3]*k_off
            # dCAB[1] = 0
            # dCAB[3] = 0
        else:
            # Case 3: state probabilities and B inside concentration changes
            dCAB[0] = 0
            dCAB[1] = 0
            # Update d[B_in]/dt
            dCAB[2] = p[0]*kon(CAB[3]) - p[5]*k_off
            dCAB[3] = 0
        return np.array([dp[0], dp[1], dp[2], dp[3], dp[4], dp[5],
                         dCAB[0], dCAB[1], dCAB[2], dCAB[3],
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0])

    time = (0, t_max)
    p0 = np.array([p_list[0][0], p_list[0][1], p_list[0][2],
                   p_list[0][3], p_list[0][4], p_list[0][5],
                   p_list[1][0], p_list[1][1],
                   p_list[1][2], p_list[1][3],
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0], dtype=np.float64)
    return scipy.integrate.solve_ivp(func, time, p0, max_step=max_step)

def NHE_antiporter(C_list, k_off, k_on, k_conf, t_max, max_step, AB_flux=None):
    """
    Parameters
    ----------
    C_list : array
    k_off : float
    k_on : float
    k_conf : float
    t_max : int
    max_step : int
    AB_flux : bool
    """

    def kon(C_x):
        """
        Concentration-dependent rate constant calculation
        """
        return k_on*C_x

    # Function to integrate
    def func(t, y):
        """
        y = [C1, C2, C3, C4, C5, C6,
             CA_in, CA_out, CB_in, CB_out,
             dC1, dC2, dC3, dC4, dC5, dC6,
             dCA_in, dCA_out, dCB_in, dCB_out]
        """
        # [1], [2], [3], [4], [5], [6]
        C = y[:6]
        # [A_in], [A_out], [B_in], [B_out]
        CAB = y[6:10]
        # d[1]/dt, d[2]/dt, d[3]/dt, d[4]/dt, d[5]/dt, d[6]/dt
        dC = y[10:16]
        # d[A_in]/dt, d[A_out]/dt, d[B_in]/dt, d[B_out]/dt
        dCAB = y[16:]
        # Update d[X]/dt's for all X
        dC[0] = k_off*(C[1] + C[5]) - C[0]*(kon(CAB[1]) + kon(CAB[3]))
        dC[1] = C[2]*k_conf + C[0]*kon(CAB[1]) - C[1]*(k_off + k_conf)
        dC[2] = C[1]*k_conf + C[3]*kon(CAB[0]) - C[2]*(k_off + k_conf)
        dC[3] = k_off*(C[2] + C[4]) - C[3]*(kon(CAB[0]) + kon(CAB[2]))
        dC[4] = C[5]*k_conf + C[3]*kon(CAB[2]) - C[4]*(k_off + k_conf)
        dC[5] = C[4]*k_conf + C[0]*kon(CAB[3]) - C[5]*(k_off + k_conf)
        if AB_flux == False:
            # Case 1: only state concentrations change
            dCAB[0] = 0
            dCAB[1] = 0
            dCAB[2] = 0
            dCAB[3] = 0
        elif AB_flux == True:
            # Case 2: state and A/B inside concentrations change
            # Update d[A_in]/dt
            dCAB[0] = C[0]*kon(CAB[1]) - C[1]*k_off
            dCAB[1] = C[4]*kon(CAB[0]) - C[3]*k_off
            # Update d[B_in]/dt
            dCAB[2] = C[0]*kon(CAB[3]) - C[5]*k_off
            dCAB[3] = C[2]*kon(CAB[2]) - C[3]*k_off
            # dCAB[1] = 0
            # dCAB[3] = 0
        else:
            # Case 3: state and B inside concentrations change
            dCAB[0] = 0
            dCAB[1] = 0
            # Update d[B_in]/dt
            dCAB[2] = C[0]*kon(CAB[3]) - C[5]*k_off
            dCAB[3] = 0
        return np.array([dC[0], dC[1], dC[2], dC[3], dC[4], dC[5],
                         dCAB[0], dCAB[1], dCAB[2], dCAB[3],
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0])

    time = (0, t_max)
    C0 = np.array([C_list[0][0], C_list[0][1], C_list[0][2],
                   C_list[0][3], C_list[0][4], C_list[0][5],
                   C_list[1][0], C_list[1][1],
                   C_list[1][2], C_list[1][3],
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0], dtype=np.float64)
    return scipy.integrate.solve_ivp(func, time, C0, max_step=max_step)

#===================================================
#==== Simulator ====================================
#===================================================

def Zuckerman_antiporter(C_list, k_off, k_on, k_conf, t_max, max_step, AB_flux=None):
    """
    Parameters
    ----------
    C_list : array
    k_off : float
    k_on : float
    k_conf : float
    t_max : int
    max_step : int
    AB_flux : bool
    """

    def kon(C_x):
        """
        Concentration-dependent rate constant calculation
        """
        return k_on*C_x

    # Function to integrate
    def func(t, y):
        """
        y = [C1, C2, C3, C4, C5, C6,
             CA_in, CA_out, CB_in, CB_out,
             dC1, dC2, dC3, dC4, dC5, dC6,
             dCA_in, dCA_out, dCB_in, dCB_out]
        """
        # [1], [2], [3], [4], [5], [6]
        C = y[:6]
        # [A_in], [A_out], [B_in], [B_out]
        CAB = y[6:10]
        # d[1]/dt, d[2]/dt, d[3]/dt, d[4]/dt, d[5]/dt, d[6]/dt
        dC = y[10:16]
        # d[A_in]/dt, d[A_out]/dt, d[B_in]/dt, d[B_out]/dt
        dCAB = y[16:]
        # Update d[X]/dt's for all X
        dC[0] = k_off*(C[1] + C[5]) - C[0]*(kon(CAB[1]) + kon(CAB[3]))
        dC[1] = C[2]*k_conf + C[0]*kon(CAB[1]) - C[1]*(k_off + k_conf)
        dC[2] = C[3]*k_off + C[1]*k_conf - C[2]*(k_conf + kon(CAB[2]))
        dC[3] = C[2]*kon(CAB[2]) + C[4]*kon(CAB[0]) - 2*C[3]*k_off
        dC[4] = C[3]*k_off + C[5]*k_conf - C[4]*(k_conf + kon(CAB[0]))
        dC[5] = C[4]*k_conf + C[0]*kon(CAB[3]) - C[5]*(k_conf + k_off)
        if AB_flux == False:
            # Case 1: only state concentrations change
            dCAB[0] = 0
            dCAB[1] = 0
            dCAB[2] = 0
            dCAB[3] = 0
        elif AB_flux == True:
            # Case 2: state and A/B inside concentrations change
            # Update d[A_in]/dt
            dCAB[0] = C[0]*kon(CAB[1]) - C[1]*k_off
            dCAB[1] = C[4]*kon(CAB[0]) - C[3]*k_off
            # Update d[B_in]/dt
            dCAB[2] = C[0]*kon(CAB[3]) - C[5]*k_off
            dCAB[3] = C[2]*kon(CAB[2]) - C[3]*k_off
            # dCAB[1] = 0
            # dCAB[3] = 0
        else:
            # Case 3: state and B inside concentrations change
            dCAB[0] = 0
            dCAB[1] = 0
            # Update d[B_in]/dt
            dCAB[2] = C[0]*kon(CAB[3]) - C[5]*k_off
            dCAB[3] = 0
        return np.array([dC[0], dC[1], dC[2], dC[3], dC[4], dC[5],
                         dCAB[0], dCAB[1], dCAB[2], dCAB[3],
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0])

    time = (0, t_max)
    C0 = np.array([C_list[0][0], C_list[0][1], C_list[0][2],
                   C_list[0][3], C_list[0][4], C_list[0][5],
                   C_list[1][0], C_list[1][1],
                   C_list[1][2], C_list[1][3],
                   0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0], dtype=np.float64)
    return scipy.integrate.solve_ivp(func, time, C0, max_step=max_step)

#===================================================
#==== Misc =========================================
#===================================================

def Arrhenius(T, A, E_A):
    """
    Calculates reaction rates via the Arrhenius equation.

    Parameters
    ----------
    T : float
        Temperature in Kelvin
    A : float
        Frequency of collisions in the correct orientation, expressed in Hz
    E_A : float
        Activation energy measured in Joules (same as k_B*T)
    """
    k_B = 1.38064852e-23    # m^2 kg s^-2 K^-1
    Beta = 1/(k_B*T)
    return A*np.log(-Beta*E_A)

def time_derivative(C, time):
    results = np.zeros((len(time)-1))
    for t in range(len(C)-1):
        results[t] = (C[t+1] - C[t])/(time[t+1] - time[t])
    return np.array(results)
