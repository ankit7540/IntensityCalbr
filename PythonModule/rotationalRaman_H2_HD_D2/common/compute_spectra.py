#!/usr/bin/python
'''Module for computing the pure rotational Raman spectra from H2, HD and D2'''

import math
import numpy as np

# Constants ------------------------------
K = np.float64(1.38064852e-23) # J/K
H = np.float64(6.626070040e-34)  # J.s
C = np.float64(2.99792458e+10)   # cm/s
# ----------------------------------------

# Laser properties------------------------
omega = 18789.9850    #absolute cm-1
omega_sc = omega/1e4  # scaled frequency (for better numerical accuracy)
# ----------------------------------------

# Load data on the energy levels and the polarizability anisotropy

# Data on the rovibrational energy levels has been extracted from the calculated dissociation
# energy data published in the following works :
#   a)  J. Komasa, K. Piszczatowski, G.  Lach, M. Przybytek, B. Jeziorski,
#        and K. Pachucki, J. Chem. Theory and Comput. 7, 3105 (2011).
#
#   b) K. Pachucki and J. Komasa, Phys. Chem. Chem. Phys. 12, 9188 (2010).
# ----------------------------------------

eJH2v0 = np.loadtxt("./energy_levels_and_ME/H2eV0.txt")
eJH2v1 = np.loadtxt("./energy_levels_and_ME/H2eV1.txt")
eJHDv0 = np.loadtxt("./energy_levels_and_ME/HDeV0.txt")
eJHDv1 = np.loadtxt("./energy_levels_and_ME/HDeV1.txt")
eJD2v0 = np.loadtxt("./energy_levels_and_ME/D2eV0.txt")
eJD2v1 = np.loadtxt("./energy_levels_and_ME/D2eV1.txt")

#   Data on the matrix elements of polarizability anisotropy has been taken from
#   our previous work.
#   c) A. Raj, H. Hamaguchi, and H. A. Witek, J. Chem. Phys. 148, 104308 (2018).

ME_alpha_H2_532_Q1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_mp_Q1.txt")
ME_alpha_HD_532_Q1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_mp_Q1.txt")
ME_alpha_D2_532_Q1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_mp_Q1.txt")

ME_gamma_H2_532_Q1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_Q1.txt")
ME_gamma_HD_532_Q1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_Q1.txt")
ME_gamma_D2_532_Q1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_Q1.txt")

ME_gamma_H2_532_O1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_O1.txt")
ME_gamma_H2_532_S1 = np.loadtxt("./energy_levels_and_ME/H2_532.2_gamma_S1.txt")

ME_gamma_HD_532_O1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_O1.txt")
ME_gamma_HD_532_S1 = np.loadtxt("./energy_levels_and_ME/HD_532.2_gamma_S1.txt")

ME_gamma_D2_532_O1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_O1.txt")
ME_gamma_D2_532_S1 = np.loadtxt("./energy_levels_and_ME/D2_532.2_gamma_S1.txt")

#print(eJH2v0)
#print(eJH2v1)
#print(eJD2v0)
#print(eJD2v1)
#print(eJHDv0)
#print(eJHDv1)

#print(ME_H2_532)
#print(ME_HD_532)
#print(ME_D2_532)

#print(ME_gamma_H2_532_O1)
#print(ME_gamma_H2_532_O1[:,4])

#********************************************************************
#********************************************************************

# common objects

# incase output is saved as txt and if header is needed in np.savetxt
header_str='J_val\tfreq\tintensity\tabs_wavenum'

#********************************************************************
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

# compute the temperature dependent sum of state for HD which includes contributions
# from the ground and first vibrational state of electronic ground state.

def sumofstate_HD(T):
    """calculate the sum of state for HD molecule at T """

    Q = np.float64(0.0)

    #--- nuclear spin statistics ------------
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

#********************************************************************
#********************************************************************

def normalize1d(array_name):
    """Normalize a 1D array using the max value"""
    max_val = np.max(array_name)
    size = len(array_name)
    for i in range(0, size, 1):
        array_name[i] = array_name[i]/max_val

#********************************************************************
#********************************************************************

def spectra_H2_o1s1(T, Js, Jas):
    """Compute in intensities and position for rotational Raman bands of H2 """

    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_H2(T)

    # define arrays for intensity, position and J_{i}
    specH2 = np.zeros(shape=(Js+Jas, 4))
    posnH2 = np.zeros(shape=(Js+Jas, 1))
    spectraH2 = np.zeros(shape=(Js+Jas, 1))

    # S1 bands
    for i in range(0, Js+1):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i+1)*(i+2))/(2*(2*i+1)*(2*i+3))
        position = (eJH2v1[i+2] - eJH2v0[i])
        gamma = ME_gamma_H2_532_S1 [i][4]
        print(i, E, popn, position ,gamma)

        if i % 2 == 0:
            factor = popn*g_even*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos
        else:
            factor = popn*g_odd*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos

        specH2[(Jas-1)+i][0] = i
        specH2[(Jas-1)+i][1] = position
        specH2[(Jas-1)+i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[(Jas-1)+i][3] = (omega-position)

        posnH2[(Jas-1)+i] = position
        spectraH2[(Jas-1)+i] = factor

    # O1 bands
    i = Jas
    for i in range(Jas, 1, -1):
        E = eJH2v0[i]
        #print(E)
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i)*(i-1))/(2*(2*i-1)*(2*i+1))
        position = (eJH2v1[i-2] - eJH2v0[i])
        gamma = ME_gamma_H2_532_O1[i-2][4]
        print(i, E, popn, position, gamma)

        if i % 2 == 0:
            factor = popn*g_even*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos
        else:
            factor = popn*g_odd*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos

        specH2[Jas-i][0] = i
        specH2[Jas-i][1] = position
        specH2[Jas-i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[Jas-i][3] = (omega-position)

        posnH2[Jas-i] = position
        spectraH2[Jas-i] = factor

    # return the output

    normalize1d(spectraH2)
    specH2[:, 2] = spectraH2[:, 0]
    return specH2
    #print("Normalized spectra H2: \n", spectraH2)
    #print(specH2)  # assign normalized intensity
    #np.savetxt('posnH2.txt', (posnH2), fmt='%+6.6f') #save the band position
    #np.savetxt('spectraH2.txt', (spectraH2), fmt='%+5.11f') #save the band intensity
    #np.savetxt('specH2.txt', (specH2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity,position in abs. wavenumbers

#********************************************************************
#********************************************************************

def spectra_H2_q1(T, Jmax):
    """Compute in intensities and position for rotational Raman bands of H2 """

    #--- nuclear spin statistics ------------
    g_even = 1        # hydrogen
    g_odd = 3
    # ---------------------------------------

    Js=0
    sos = sumofstate_H2(T)

    # define arrays for intensity, position and J_{i}
    specH2 = np.zeros(shape=(Jmax+1, 4))
    posnH2 = np.zeros(shape=(Jmax+1, 1))
    spectraH2 = np.zeros(shape=(Jmax+1, 1))

    # Q-branch ----------------------------------
    for i in range(Js, Jmax+1):
        E = eJH2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJH2v1[i]-eJH2v0[i])
        alpha = ME_alpha_H2_532_Q1[i][4]
        gamma = ME_gamma_H2_532_Q1[i][4]
        print(i, E, popn, position, alpha, gamma)

        if i % 2 == 0:
            factor = (popn/sos)*g_even*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)
        else:
            factor = (popn/sos)*g_odd*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)

        specH2[i][0] = i
        specH2[i][1] = position
        specH2[i][2] = factor  # unnormalized intensity, arbitrary unit
        specH2[i][3] = (omega-position)

        posnH2[i] = position
        spectraH2[i] = factor

    # -----------------------------------------------

    # return the output

    normalize1d(spectraH2)
    specH2[:, 2] = spectraH2[:, 0]

    #save the data
    np.savetxt('spectraH2_q1_298.txt', (specH2),delimiter='\t',\
               header=header_str, fmt='%+5.11f')

    return specH2
    #print("Normalized spectra H2: \n", spectraH2)
    #print(specH2)  # assign normalized intensity
    #np.savetxt('posnH2.txt', (posnH2), fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraH2_q1_298.txt', (spectraH2), fmt='%+5.11f') #save the band intensity
    #np.savetxt('specH2.txt', (specH2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers

#********************************************************************
#********************************************************************

def spectra_HD_o1s1(T, Js, Jas):
    """Compute in intensities and position for rotational Raman bands of HD """

    #--- nuclear spin statistics ------------
    #g_even = 1        # hydrogen deuteride
    #g_odd = 1
    # ---------------------------------------

    sos = sumofstate_HD(T)

    # define arrays for intensity, position and J_{i}
    specHD = np.zeros(shape=(Js+Jas, 4))
    posnHD = np.zeros(shape=(Js+Jas, 1))
    spectraHD = np.zeros(shape=(Js+Jas, 1))

    # S1 bands ----------------------------------
    for i in range(0, Js+1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i+1)*(i+2))/(2*(2*i+1)*(2*i+3))
        position = (eJHDv1[i+2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_S1 [i][4]
        print(i, E, popn, position ,gamma)

        factor = popn*bj*omega_sc*((omega_sc-position/1e4)**3)\
            *(gamma**2)/sos

        specHD[(Jas-1)+i][0] = i
        specHD[(Jas-1)+i][1] = position
        specHD[(Jas-1)+i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[(Jas-1)+i][3] = omega-position
        posnHD[(Jas-1)+i] = position
        spectraHD[(Jas-1)+i] = factor


    # O1 bands ----------------------------------
    i = Jas
    for i in range(Jas, 1, -1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i)*(i-1))/(2*(2*i-1)*(2*i+1))
        position = (eJHDv1[i-2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_O1[i-2][4]

        factor = popn*bj*omega_sc*((omega_sc-position/1e4)**3)\
            *(gamma**2)/sos

        specHD[Jas-i][0] = i
        specHD[Jas-i][1] = position
        specHD[Jas-i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[Jas-i][3] = omega-position
        posnHD[Jas-i] = position
        spectraHD[Jas-i] = factor


#   return the output
    normalize1d(spectraHD)
    specHD[:, 2] = spectraHD[:, 0]  # assign normalized intensity
    #print("Normalized spectra HD: \n", spectraHD)
    return specHD

    #np.savetxt('posnHD.txt', (posnHD), fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraHD.txt', (spectraHD), fmt='%+5.11f') #save the band intensity
    #np.savetxt('specHD.txt', (specHD), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers

#********************************************************************
#********************************************************************

#********************************************************************

def spectra_HD_q1(T, Jmax):
    """Compute in intensities and position for rotational Raman bands of H2 """

    #--- nuclear spin statistics ------------
    #g_even = 1        # hydrogen deuteride
    #g_odd = 1
    # ---------------------------------------

    Js=0
    sos = sumofstate_HD(T)

    # define arrays for intensity, position and J_{i}
    specHD = np.zeros(shape=(Jmax+1, 4))
    posnHD = np.zeros(shape=(Jmax+1, 1))
    spectraHD = np.zeros(shape=(Jmax+1, 1))

    # Q-branch ----------------------------------
    for i in range(Js, Jmax+1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJHDv1[i]-eJHDv0[i])
        alpha = ME_alpha_HD_532_Q1[i][4]
        gamma = ME_gamma_HD_532_Q1[i][4]
        print(i, E, popn, position, alpha, gamma)

        factor = (popn/sos)*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)

        specHD[i][0] = i
        specHD[i][1] = position
        specHD[i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[i][3] = (omega-position)

        posnHD[i] = position
        spectraHD[i] = factor

    # -----------------------------------------------

    # return the output

    normalize1d(spectraHD)
    specHD[:, 2] = spectraHD[:, 0]
    np.savetxt('spectraHD_q1_298.txt', (specHD),delimiter='\t',\
               header=header_str, fmt='%+5.11f') #save the band intensity
    return specHD
    #print("Normalized spectra H2: \n", spectraH2)
    #print(specH2)  # assign normalized intensity
    #np.savetxt('posnH2.txt', (posnH2), fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraH2_q1_298.txt', (spectraH2), fmt='%+5.11f') #save the\ band intensity
    #np.savetxt('specH2.txt', (specH2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers

#********************************************************************
#********************************************************************

def spectra_D2(T, Js, Jas):
    """Compute in intensities and position for rotational Raman bands of D2 """

    #--- nuclear spin statistics ------------
    g_even = 6        # deuterium
    g_odd = 3
    # ---------------------------------------

    sos = sumofstate_D2(T)

    # define arrays for intensity, position and J_{i}
    specD2 = np.zeros(shape=(Js+Jas, 4))
    posnD2 = np.zeros(shape=(Js+Jas, 1))
    spectraD2 = np.zeros(shape=(Js+Jas, 1))

    # S1 bands ----------------------------------
    for i in range(0, Js+1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i+1)*(i+2))/(2*(2*i+1)*(2*i+3))
        position = (eJD2v1[i+2]-eJD2v0[i])
        gamma = ME_gamma_D2_532_S1 [i][4]

        if i % 2 == 0:
            factor = popn*g_even*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos
        else:
            factor = popn*g_odd*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos

        specD2[(Jas-1)+i][0] = i
        specD2[(Jas-1)+i][1] = position
        specD2[(Jas-1)+i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[(Jas-1)+i][3] = omega-position
        posnD2[(Jas-1)+i] = position
        spectraD2[(Jas-1)+i] = factor


    # O1 bands ----------------------------------
    i = Jas
    for i in range(Jas, 1, -1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i)*(i-1))/(2*(2*i-1)*(2*i+1))
        position = (eJD2v1[i-2]-eJD2v0[i])
        gamma = ME_gamma_D2_532_O1[i-2][4]

        if i % 2 == 0:
            factor = popn*g_even*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos
        else:
            factor = popn*g_odd*bj*omega_sc*((omega_sc-position/1e4)**3)\
                *(gamma**2)/sos

        specD2[Jas-i][0] = i
        specD2[Jas-i][1] = position
        specD2[Jas-i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[Jas-i][3] = omega-position
        posnD2[Jas-i] = position
        spectraD2[Jas-i] = factor

#   return the output
    normalize1d(spectraD2)
    specD2[:, 2] = spectraD2[:, 0]  # assign normalized intensity
    return specD2
    #print("Normalized spectra D2 :\n", spectraD2)

    #np.savetxt('posnD2.txt', posnD2, fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraD2.txt', spectraD2, fmt='%+5.11f') #save the band intensity
    #np.savetxt('specD2.txt', (specD2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers

#********************************************************************
#********************************************************************


def spectra_D2_q1(T, Jmax):
    """Compute in intensities and position for rotational Raman bands of H2 """
    #--- nuclear spin statistics ------------
    g_even = 6        # deuterium
    g_odd = 3
    # ---------------------------------------

    Js=0
    sos = sumofstate_D2(T)

    # define arrays for intensity, position and J_{i}
    specD2 = np.zeros(shape=(Jmax+1, 4))
    posnD2 = np.zeros(shape=(Jmax+1, 1))
    spectraD2 = np.zeros(shape=(Jmax+1, 1))

    # Q1 bands ----------------------------------
    for i in range(Js, Jmax+1):
        E = eJD2v0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (i*(i+1))/((2*i-1)*(2*i+3))
        position = (eJD2v1[i]-eJD2v0[i])
        alpha = ME_alpha_D2_532_Q1[i][4]
        gamma = ME_gamma_D2_532_Q1[i][4]
        print(i, E, popn, position, alpha, gamma)

        if i % 2 == 0:
            factor = (popn/sos)*g_even*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)
        else:
            factor = (popn/sos)*g_odd*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)


        specD2[i][0] = i
        specD2[i][1] = position
        specD2[i][2] = factor  # unnormalized intensity, arbitrary unit
        specD2[i][3] = (omega-position)

        posnD2[i] = position
        spectraD2[i] = factor

    # -----------------------------------------------

    # return the output

    normalize1d(spectraD2)
    specD2[:, 2] = spectraD2[:, 0]
    np.savetxt('spectraD2_q1_298.txt', (specD2),delimiter='\t',\
               header=header_str, fmt='%+5.11f') #save the band intensity
    return specD2
    #print("Normalized spectra H2: \n", spectraH2)
    #print(specH2)  # assign normalized intensity
    #np.savetxt('posnH2.txt', (posnH2), fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraH2_q1_298.txt', (spectraH2), fmt='%+5.11f') #save the band intensity
    #np.savetxt('specH2.txt', (specH2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers



#*******************************************************************************
#*******************************************************************************

def HD_S1(T, JMax, sos):
    '''compute the intensity for HD upto given JMax and sum of state '''

    specHD = np.zeros(shape=(JMax+1, 4))

    # S1 bands ----------------------------------
    for i in range(0, JMax+1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i+1)*(i+2))/(2*(2*i+1)*(2*i+3))
        position = (eJHDv1[i+2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_S1 [i][4]
        #print(i, E, popn, position ,gamma)

        factor = popn*bj*omega_sc*((omega_sc-position/1e4)**3)\
            *(gamma**2)/sos

        specHD[i][0] = i
        specHD[i][1] = position
        specHD[i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[i][3] = omega-position

    return specHD

#*******************************************************************************

def HD_O1(T, JMax, sos):
    '''compute the intensity for HD upto given JMax and sum of state '''

    specHD = np.zeros(shape=(JMax-1, 4))

    # O1 bands ----------------------------------

    for i in range(JMax, 1, -1):
        E = eJHDv0[i]
        energy = (-1*E*H*C)
        popn = (2*i+1)*math.exp(energy/(K*T))
        bj = (3*(i)*(i-1))/(2*(2*i-1)*(2*i+1))
        position = (eJHDv1[i-2]-eJHDv0[i])
        gamma = ME_gamma_HD_532_O1[i-2][4]

        factor = popn*bj*omega_sc*((omega_sc-position/1e4)**3)\
            *(gamma**2)/sos
            
        #print(JMax-i)    

        specHD[JMax-i][0] = i
        specHD[JMax-i][1] = position
        specHD[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[JMax-i][3] = omega-position

    return specHD
#*******************************************************************************


def HD_Q1(T, JMax, sos):
    '''compute the intensity for HD upto given JMax and sum of state '''

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

        factor = (popn/sos)*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gamma**2)+ alpha**2)

        specHD[JMax-i][0] = i
        specHD[JMax-i][1] = position
        specHD[JMax-i][2] = factor  # unnormalized intensity, arbitrary unit
        specHD[JMax-i][3] = (omega-position)

    return specHD
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************


def spectra_series_HD(T, OJ, QJ, SJ, reverse=0):
    """Compute in intensities and position for rotational Raman bands of HD
        where OJ = max J state for O(v=1) bands
              QJ = max J state for Q(v=1) bands
              SJ = max J state for S(v=1) bands

              reverse=0 or 1, will reverse the output

     """

    sos = sumofstate_HD(T)

    # define arrays for intensity, position and J_{i}
    print(OJ, QJ, SJ)
    print(OJ-1+QJ+SJ+1)

    # Q1 bands ----------------------------------
    #print(HD_S1 (T, 4, sos) )
    O1=HD_O1(T, 4, sos)
    Q1=HD_Q1(T, 4, sos)
    S1=HD_S1(T, 4, sos)
    
    print('\n',HD_O1(T, 4, sos))

    print('\n',HD_Q1(T, 4, sos))
    
    print('\n',HD_S1(T, 4, sos))

    out=np.concatenate((O1, Q1, S1 ))
    
    sp_HD=out[:,2]
    print(sp_HD)
    normalize1d(sp_HD)
    print(sp_HD)
    
    return(out)

    # -----------------------------------------------



    # return the output

    #normalize1d(spectraD2)
    #specD2[:, 2] = spectraD2[:, 0]
    #np.savetxt('spectraD2_q1_298.txt', (specD2),delimiter='\t',\
    #           header=header_str, fmt='%+5.11f') #save the band intensity
    #return specD2
    #print("Normalized spectra H2: \n", spectraH2)
    #print(specH2)  # assign normalized intensity
    #np.savetxt('posnH2.txt', (posnH2), fmt='%+6.6f') #save the band position.
    #np.savetxt('spectraH2_q1_298.txt', (spectraH2), fmt='%+5.11f') #save the band intensity
    #np.savetxt('specH2.txt', (specH2), fmt='%+5.12f') #save the 2D array which has data on
    #  initial J number, band position, band intensity and the position in abs. wavenumbers

#********************************************************************



#print(" Sum of state for  H2 at 333 K : ", sumofstate_H2(333))

#print(" Sum of state for  HD at 375 K : ", sumofstate_HD(375))

#print(" Sum of state for  D2 at 298 K : ", sumofstate_D2(298))

#********************************************************************


#  Spectra can  be plotted using matplotlib simply using the band posn and the spectra

#********************************************************************
