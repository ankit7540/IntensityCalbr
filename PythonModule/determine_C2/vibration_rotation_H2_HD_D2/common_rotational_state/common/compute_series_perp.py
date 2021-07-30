#!/usr/bin/python
# pylint: disable=wildcard-import, method-hidden,C0103
'''Module for computing the pure rotational Raman spectra from H2, HD and D2'''

import math
import numpy as np

# FOR PERPENDICULAR POLARIZATION

# Constants ------------------------------
K = np.float64(1.38064852e-23) # J/K
H = np.float64(6.626070040e-34)  # J.s
C = np.float64(2.99792458e+10)   # cm/s
# ----------------------------------------

# Laser properties------------------------
omega = 18789.9850    #absolute cm-1
omega_sc = omega / 1e4  # scaled frequency (for better numerical accuracy)
# ----------------------------------------

# Load data on the energy levels and the polarizability anisotropy

# Data on the rovibrational energy levels has been extracted from
#   the calculated dissociation
# energy data published in the following works :
#   a)  J. Komasa, K. Piszczatowski, G.  Lach, M. Przybytek, B. Jeziorski,
#        and K. Pachucki, J. Chem. Theory and Comput. 7, 3105 (2011).
#
#   b) K. Pachucki and J. Komasa, Phys. Chem. Chem. Phys. 12, 9188 (2010).
# ----------------------------------------

eJH2v0 = np.loadtxt("./energy_levels_and_ME/H2eV0.dat")
eJH2v1 = np.loadtxt("./energy_levels_and_ME/H2eV1.dat")
eJHDv0 = np.loadtxt("./energy_levels_and_ME/HDeV0.dat")
eJHDv1 = np.loadtxt("./energy_levels_and_ME/HDeV1.dat")
eJD2v0 = np.loadtxt("./energy_levels_and_ME/D2eV0.dat")
eJD2v1 = np.loadtxt("./energy_levels_and_ME/D2eV1.dat")

#   Data on the matrix elements of polarizability anisotropy has been taken
#    from our previous work.
#   c) A. Raj, H. Hamaguchi, and H. A. Witek, J. Chem. Phys. 148, 104308 (2018)

ME_alpha_H2_532_Q1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_mp_Q1.dat")
ME_alpha_HD_532_Q1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_mp_Q1.dat")
ME_alpha_D2_532_Q1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_mp_Q1.dat")

ME_gamma_H2_532_Q1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_Q1.dat")
ME_gamma_HD_532_Q1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_Q1.dat")
ME_gamma_D2_532_Q1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_Q1.dat")

ME_gamma_H2_532_O1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_O1.dat")
ME_gamma_H2_532_S1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_S1.dat")

ME_gamma_HD_532_O1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_O1.dat")
ME_gamma_HD_532_S1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_S1.dat")

ME_gamma_D2_532_O1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_O1.dat")
ME_gamma_D2_532_S1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_S1.dat")



# ********************************************************************
# ********************************************************************

# common objects

# header, incase output is saved as txt and if header is needed in np.savetxt
header_str = 'J_val\tfreq\tintensity\tabs_wavenum'

# ********************************************************************
# ********************************************************************

# set of functions for computing the Sum of states of molecular hydrogen and
#   its isotopologues at given temperature.
# Data on energy levels is needed for the specific molecule


def sumofstate_H2(T):
    """calculate the sum of state for H2 molecule at T """

    Q = np.float64(0.0)

    # --- nuclear spin statistics ------------
    g_even = 1 	# hydrogen molecule
    g_odd = 3
    # ---------------------------------------

    jmaxv0 = len(eJH2v0)
    jmaxv1 = len(eJH2v1)

    # contribution from the ground vibrational state
    for i in range(0, jmaxv0):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

    # contribution from the first vibrational state
    for i in range(0, jmaxv1):
        E = eJH2v1[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

#   return the sum of states for H2
    return Q

#********************************************************************
#********************************************************************

# compute the temperature dependent sum of state for HD which includes
#  contributions from the ground and first vibrational state of electronic
#  ground state.


def sumofstate_HD(T):
    """calculate the sum of state for HD molecule at T """

    Q = np.float64(0.0)

    # --- nuclear spin statistics ------------
    g_even = 1        # hydrogen deuteride
    g_odd = 1
    # ---------------------------------------

    jmaxv0 = len(eJHDv0)
    jmaxv1 = len(eJHDv1)

    # contribution from the ground vibrational state
    for i in range(0, jmaxv0):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

    # contribution from the first vibrational state
    for i in range(0, jmaxv1):
        E = eJHDv1[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

#   return the sum of states for HD
    return Q

# *****************************************************************************
# *****************************************************************************

# compute the temperature dependent sum of state for D2 which
#  includes contributions from the ground and first vibrational state
#  of electronic ground state.


def sumofstate_D2(T):
    """calculate the sum of state for D2 molecule at T """

    Q = np.float64(0.0)

    # --- nuclear spin statistics ------------
    g_even = 6        # deuterium molecule
    g_odd = 3
    # ----------------------------------------

    jmaxv0 = len(eJD2v0)
    jmaxv1 = len(eJD2v1)

    # contribution from the ground vibrational state
    for i in range(0, jmaxv0):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

    # contribution from the first vibrational state
    for i in range(0, jmaxv1):
        E = eJD2v1[i]
        energy = (-1*E*H*C)
        factor = (2*i+1)*math.exp(energy/(K*T))
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        Q = Q+factor

#   return the sum of states for D2
    return Q


# *****************************************************************************
#                      COMPUTING SPECTRAL INTENSITIES
# *****************************************************************************
#  OUTPUT : nx4 array (where n is the number of rows)
#   For every band 4 parameters are given,
#       J           = rotational quantum number
#       freq        = frequency (Ramanshift)
#       intensity   = intensity, arbitrary units
#       abs_wavenum = frequency in absolute wavenumbers
# *****************************************************************************


def HD_S1(T, JMax, sos):
    '''compute the intensity for HD, S1 bands upto given JMax for T
    sum of states has to be supplied as argument '''

    specHD = np.zeros(shape=(JMax + 1, 4))

    # S1 bands ----------------------------------
    for i in range(0, JMax+1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i+1)*(i+2))/((2*i+1)*(2*i+3))
        position = (eJHDv1[i+2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_S1[i][4]
        #print(i, E, popn, position ,gamma)

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos

        specHD[i][0] = i
        specHD[i][1] = position
        specHD[i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[i][3] = omega-position

    return specHD

# *****************************************************************************

def HD_O1(T, JMax, sos):
    '''compute the intensity for HD O1 bands upto given JMax and T
    sum of states has to be supplied as argument  '''

    specHD = np.zeros(shape=(JMax - 1, 4))

    # O1 bands ----------------------------------

    for i in range(JMax, 1, -1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i)*(i-1))/((2*i-1)*(2*i+1))
        position = (eJHDv1[i-2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_O1[i-2][4]

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos

        #print(JMax-i)

        specHD[JMax-i][0] = i
        specHD[JMax-i][1] = position
        specHD[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[JMax-i][3] = omega-position

    return specHD
# *****************************************************************************


def HD_Q1(T, JMax, sos):
    '''compute the intensity for HD Q1 bands upto given JMax and given T
    sum of states has to be supplied as argument  '''

    specHD = np.zeros(shape=(JMax+1, 4))

    # Q-branch ----------------------------------
    for i in range(0, JMax+1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJHDv1[i]-eJHDv0[i])
        alpha = ME_alpha_HD_532_Q1[i][4]
        gamma = ME_gamma_HD_532_Q1[i][4]
        #print(i, E, popn, position, alpha, gamma)

        factor = (popn/sos)*omega_sc*(((omega-position)/1e4)**3)*\
                (bj*(1/15)*(gamma**2)+ alpha**2)

        specHD[JMax-i][0] = i
        specHD[JMax-i][1] = position
        specHD[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[JMax-i][3] = (omega-position)

    return specHD
# *****************************************************************************

def spectra_HD(T, OJ, QJ, SJ, sos):
    """Compute in intensities and position for rotational Raman bands of HD
        where OJ = max J state for O(v = 1) bands
              QJ = max J state for Q(v = 1) bands
              SJ = max J state for S(v = 1) bands
     """

    # call individual functions ------------------------
    O1 = HD_O1(T, OJ, sos)
    Q1 = HD_Q1(T, QJ, sos)
    S1 = HD_S1(T, SJ, sos)
    #---------------------------------------------------
    out = np.concatenate((O1, Q1, S1))
    return out  # return the output
    # --------------------------------------------------
# *****************************************************************************

def spectra_HD_o1s1(T, OJ, SJ, sos):
    """Compute in intensities and position for rotational Raman bands of HD
        where OJ = max J state for O(v = 1) bands
              SJ = max J state for S(v = 1) bands
     """

    # call individual functions ------------------------
    O1 = HD_O1(T, OJ, sos)
    S1 = HD_S1(T, SJ, sos)
    # --------------------------------------------------
    out = np.concatenate((O1, S1))
    return out  # return the output
    # --------------------------------------------------


# *****************************************************************************
# ----------  D2  ----------
# *****************************************************************************

def D2_S1(T, JMax, sos):
    '''compute the intensity for D2, S1 bands upto given JMax and T '''

    specD2 = np.zeros(shape=(JMax+1, 4))
    # --- nuclear spin statistics ------------
    g_even = 6        # deuterium molecule
    g_odd = 3
    # ----------------------------------------
    # S1 bands ----------------------------------
    for i in range(0, JMax+1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i+1)*(i+2))/((2*i+1)*(2*i+3))
        position = (eJD2v1[i+2]-eJD2v0[i])
        gamma = ME_gamma_D2_532_S1[i][4]
        #print(i, E, popn, position ,gamma)

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        specD2[i][0] = i
        specD2[i][1] = position
        specD2[i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[i][3] = omega-position

    return specD2

# *****************************************************************************

def D2_O1(T, JMax, sos):
    '''compute the intensity for D2, O1 bands upto
    given JMax and sum of state '''

    specD2 = np.zeros(shape=(JMax-1, 4))
    # --- nuclear spin statistics ------------
    g_even = 6        # deuterium molecule
    g_odd = 3
    # ----------------------------------------
    # O1 bands ----------------------------------

    for i in range(JMax, 1, -1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i)*(i-1))/((2*i-1)*(2*i+1))
        position = (eJD2v1[i-2]-eJD2v0[i])
        gamma = ME_gamma_D2_532_O1[i-2][4]

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        specD2[JMax-i][0] = i
        specD2[JMax-i][1] = position
        specD2[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[JMax-i][3] = omega-position

    return specD2
# *****************************************************************************


def D2_Q1(T, JMax, sos):
    '''compute the intensity for D2, Q1 bands upto given JMax and sum of state '''

    specD2 = np.zeros(shape=(JMax+1, 4))
    # --- nuclear spin statistics ------------
    g_even = 6        # deuterium molecule
    g_odd = 3
    # ----------------------------------------
    # Q-branch ----------------------------------
    for i in range(0, JMax+1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJD2v1[i]-eJD2v0[i])
        alpha = ME_alpha_D2_532_Q1[i][4]
        gamma = ME_gamma_D2_532_Q1[i][4]

        factor = (popn/sos)*omega_sc*(((omega-position)/1e4)**3)*\
                (bj*(1/15)*(gamma**2)+ alpha**2)
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        specD2[JMax-i][0] = i
        specD2[JMax-i][1] = position
        specD2[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[JMax-i][3] = (omega-position)

    return specD2
# *****************************************************************************


def spectra_D2(T, OJ, QJ, SJ, sos):
    """Compute in intensities and position for rotational Raman bands of D2
        where OJ = max J state for O(v = 1) bands
              QJ = max J state for Q(v = 1) bands
              SJ = max J state for S(v = 1) bands
     """
    # call individual functions ------------------------

    O1 = D2_O1(T, OJ, sos)
    Q1 = D2_Q1(T, QJ, sos)
    S1 = D2_S1(T, SJ, sos)
    # --------------------------------------------------
    out = np.concatenate((O1, Q1, S1))
    return out
    # --------------------------------------------------

# *****************************************************************************


def spectra_D2_o1s1(T, OJ, SJ, sos):
    """Compute in intensities and position for rotational Raman bands of D2
        where OJ = max J state for O(v = 1) bands
              QJ = max J state for Q(v = 1) bands
              SJ = max J state for S(v = 1) bands
     """

    # call individual functions ------------------------

    O1 = D2_O1(T, OJ, sos)
    S1 = D2_S1(T, SJ, sos)
    # --------------------------------------------------
    out = np.concatenate((O1, S1))
    return out
    # --------------------------------------------------

# *****************************************************************************
# ----------  H2  ----------
# *****************************************************************************


def H2_S1(T, JMax, sos):
    '''compute the intensity for H2, S1 bands upto given JMax and T '''

    specH2 = np.zeros(shape=(JMax+1, 4))
    # --- nuclear spin statistics ------------
    g_even = 1 	# hydrogen molecule
    g_odd = 3
    # ---------------------------------------
    # S1 bands ----------------------------------
    for i in range(0, JMax+1):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i+1)*(i+2))/((2*i+1)*(2*i+3))
        position = (eJH2v1[i+2]-eJH2v0[i])
        gamma = ME_gamma_H2_532_S1[i][4]
        #print(i, E, popn, position ,gamma)

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd
        specH2[i][0] = i
        specH2[i][1] = position
        specH2[i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[i][3] = omega-position

    return specH2

# *****************************************************************************

def H2_O1(T, JMax, sos):
    '''compute the intensity for HD O1 bands upto given
     JMax and sum of state '''

    specH2 = np.zeros(shape=(JMax-1, 4))
    # --- nuclear spin statistics ------------
    g_even = 1 	# hydrogen molecule
    g_odd = 3
    # ---------------------------------------
    # O1 bands ----------------------------------

    for i in range(JMax, 1, -1):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = ((i)*(i-1))/((2*i-1)*(2*i+1))
        position = (eJH2v1[i-2]-eJH2v0[i])
        gamma = ME_gamma_H2_532_O1[i-2][4]

        factor = popn*bj*omega_sc*(((omega-position)/1e4)**3)\
            *(1/10)*(gamma**2)/sos
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd

        #print(JMax-i)

        specH2[JMax-i][0] = i
        specH2[JMax-i][1] = position
        specH2[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[JMax-i][3] = omega-position

    return specH2
# *****************************************************************************


def H2_Q1(T, JMax, sos):
    '''compute the intensity for H2 Q-branch upto given
     JMax and sum of state '''

    specH2 = np.zeros(shape=(JMax+1, 4))
    # --- nuclear spin statistics ------------
    g_even = 1 	# hydrogen molecule
    g_odd = 3
    # ---------------------------------------
    # Q-branch ----------------------------------
    for i in range(0, JMax+1):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJH2v1[i]-eJH2v0[i])
        alpha = ME_alpha_H2_532_Q1[i][4]
        gamma = ME_gamma_H2_532_Q1[i][4]

        factor = (popn/sos)*omega_sc*(((omega-position)/1e4)**3)*\
                (bj*(1/15)*(gamma**2)+ alpha**2)
        if i % 2 == 0:
            factor = factor*g_even
        else:
            factor = factor*g_odd

        specH2[JMax-i][0] = i
        specH2[JMax-i][1] = position
        specH2[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[JMax-i][3] = (omega-position)

    return specH2

# *****************************************************************************
# *****************************************************************************


def spectra_H2(T, OJ, QJ, SJ, sos):
    """Compute in intensities and position for rotational Raman bands of H2
        where OJ = max J state for O(v = 1) bands
              QJ = max J state for Q(v = 1) bands
              SJ = max J state for S(v = 1) bands
     """
    # call individual functions ------------------------

    O1 = H2_O1(T, OJ, sos)
    Q1 = H2_Q1(T, QJ, sos)
    S1 = H2_S1(T, SJ, sos)
    # --------------------------------------------------
    out = np.concatenate((O1, Q1, S1))
    return out
    # --------------------------------------------------

# *****************************************************************************
# *****************************************************************************


def spectra_H2_c(T, OJ, QJ, sos):
    """Compute in intensities and position for rotational Raman bands of H2
        where OJ = max J state for O(v = 1) bands
              QJ = max J state for Q(v = 1) bands
              SJ = max J state for S(v = 1) bands
              This does not include S1 bands, specific for the spectral
              range where no S1 bands observed.
     """
    # call individual functions ------------------------

    O1 = H2_O1(T, OJ, sos)
    Q1 = H2_Q1(T, QJ, sos)

    # --------------------------------------------------
    out = np.concatenate((O1, Q1))
    return out
    # --------------------------------------------------
# *****************************************************************************
# *****************************************************************************
