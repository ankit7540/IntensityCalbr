import numpy as np
import math
# Constants ------------------------------
K = np.float64(1.38064852e-23)   # J/K
H = np.float64(6.626070040e-34)  # J.s
C = np.float64(2.99792458e+10)   # cm/s
# ----------------------------------------

##################################
############ COMMON ##############

eJH2 = np.genfromtxt("./energy_levels/H2.txt", delimiter="\t")
eJHD = np.genfromtxt("./energy_levels/HD.txt", delimiter="\t")
eJD2 = np.genfromtxt("./energy_levels/D2.txt", delimiter="\t")


#********************************************************************

# set of functions for computing the Sum of states of molecular hydrogen and
#   its isotopologues at given temperature.
# Data on energy levels is needed for the specific molecule

#********************************************************************

# set of functions for computing the Sum of states of molecular hydrogen and
#   its isotopologues at given temperature.
# Data on energy levels is needed for the specific molecule

def sumofstate_H2(T):
    """calculate the sum of state for H2 molecule at T """

    Q = np.float64(0.0)

    #--- nuclear spin statistics ------------
    g_even = 1 	# hydrogen
    g_odd = 3
    # ---------------------------------------

    data = eJH2

    nCols = data.shape[1]
    # nCols is equal to the number of vibrational
    #  states included in the summation

    # generate Q using each energy from the dataset
    for i in range(0, nCols):

        # select row for v=i
        row = data[:,i]

        # remove nan values
        x = row[np.logical_not(np.isnan(row))]

        # get the dimension (equal to J_max)
        nRows = x.shape[0]

        # iterate over the available energies
        for j in range(0, nRows):
            E = x[j]
            energy = (-1*E*H*C)

            factor = (2*j+1)*math.exp(energy/(K*T))

            if j % 2 == 0:
                factor = factor*g_even
            else:
                factor = factor*g_odd
            Q = Q+factor



    #   return the sum of states for H2
    return Q
#********************************************************************

#********************************************************************
#********************************************************************

# compute the temperature dependent sum of state for HD which includes contributions
# from the ground and first vibrational state of electronic ground state.

def sumofstate_HD(T):
    """calculate the sum of state for HD molecule at T """

    Q = np.float64(0.0)

    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen deuteride
    g_odd = 1
    # ---------------------------------------

    data = eJHD

    nCols = data.shape[1]
    # nCols is equal to the number of vibrational
    #  states included in the summation

    # generate Q using each energy from the dataset
    for i in range(0, nCols):

        # select row for v=i
        row = data[:,i]

        # remove nan values
        x = row[np.logical_not(np.isnan(row))]

        # get the dimension (equal to J_max)
        nRows = x.shape[0]

        # iterate over the available energies
        for j in range(0, nRows):
            E = x[j]
            energy = (-1*E*H*C)

            factor = (2*j+1)*math.exp(energy/(K*T))

            if j % 2 == 0:
                factor = factor*g_even
            else:
                factor = factor*g_odd
            Q = Q+factor



    #   return the sum of states for HD
    return Q
#********************************************************************
#********************************************************************

# compute the temperature dependent sum of state for D2 which includes contributions
# from the ground and first vibrational state of electronic ground state.

def sumofstate_D2(T):
    """calculate the sum of state for D2 molecule at T """

    Q = np.float64(0.0)

    #--- nuclear spin statistics ------------
    g_even = 6        # deuterium
    g_odd = 3
    # ---------------------------------------

    data = eJD2

    nCols = data.shape[1]
    # nCols is equal to the number of vibrational
    #  states included in the summation

    # generate Q using each energy from the dataset
    for i in range(0, nCols):

        # select row for v=i
        row = data[:,i]

        # remove nan values
        x = row[np.logical_not(np.isnan(row))]

        # get the dimension (equal to J_max)
        nRows = x.shape[0]

        # iterate over the available energies
        for j in range(0, nRows):
            E = x[j]
            energy = (-1*E*H*C)

            factor = (2*j+1)*math.exp(energy/(K*T))

            if j % 2 == 0:
                factor = factor*g_even
            else:
                factor = factor*g_odd
            Q = Q+factor



    #   return the sum of states for H2
    return Q
#********************************************************************

def popln_H2_v0(T, J):
    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_H2(T)
    E = eJH2[J, 0]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor

#********************************************************************

def popln_H2_v1(T, J):
    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_H2(T)
    E = eJH2[J, 1]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor

#********************************************************************

def popln_D2_v0(T, J):
    #--- nuclear spin statistics ------------
    g_even = 6        # deuterium
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_D2(T)
    E = eJD2[J, 0]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor

#********************************************************************

def popln_D2_v1(T, J):
    #--- nuclear spin statistics ------------
    g_even = 6        # deuterium
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_D2(T)
    E = eJD2[J, 1]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor

#********************************************************************

def popln_HD_v0(T, J):
    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen deuteride
    g_odd = 1
    # ---------------------------------------

    sos = sumofstate_HD(T)
    E = eJHD[J, 0]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor
#********************************************************************

def popln_HD_v1(T, J):
    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen deuteride
    g_odd = 1
    # ---------------------------------------

    sos = sumofstate_HD(T)
    E = eJHD[J, 1]
    energy = (-1*E*H*C)
    factor = ((2*J+1)*math.exp(energy/(K*T)))/sos

    if J % 2 == 0:
        factor = factor*g_even
    else:
        factor = factor*g_odd

    return factor
#********************************************************************
