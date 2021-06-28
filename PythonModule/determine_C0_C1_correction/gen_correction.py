#############################################################################

import numpy as np
from numpy.polynomial import Polynomial
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#############################################################################

axis=np.loadtxt('axis.txt')
wl=np.loadtxt('wl.txt')


# This file has function(s) for determination of the C0 and C1 corrections

def gen_C0_C1 (Ramanshift, laser_nm, wl_spectra, norm_pnt, mask=None, set_mask_nan=None):

    '''Ramanshift = vector, the x-axis in wavenumbers
       laser_nm = scalar, the laser wavelength in nanometers,
       wl_spectra = broadband whitelight spectra (1D or 2D),
       norm_pnt =  normalization point (corrections will be set
                                        to unity at this point),

       OPTIONAL PARAMETERS
       mask = vector, mask wave for selecting specific region to fit ,
       set_mask_nan= boolean, 1 will set the masked region in
       the output correction to
          nan, 0 will not '''


    C0 = gen_C0 (Ramanshift, norm_pnt)

    # check the whitelight spectrum provided
    dim = wl_spectra.shape
    if(dim[0] != Ramanshift.shape[0]):
        print ("\t Error : Dimension mismatch for wl spectra and the xaxis")

    if (dim[1] > 1 and dim[1] ):
        print ("\t wl spectra is 2D")
        wl_spectra = np.mean(wl_spectra, axis=1)
        print(wl_spectra.shape)

    # normalize the wl
    wl_norm = wl_spectra / np.amax(wl_spectra)

    # correct wl with C0
    wl_norm_C0 = np.multiply(wl_norm , C0)

    # check if mask supplied
    if (mask==None):
        print ("\t Mask not available. Proceeding with fit..")
    else (isinstance(obj, np.ndarray) == 1):
        print ("\t Mask is available.")


    #----------------------------------------------------------


#############################################################################

def gen_C0 (Ramanshift, norm_pnt):
    '''Ramanshift = vector, the x-axis in wavenumbers
       norm_pnt =  normalization point (corrections will be set
                                        to unity at this point) '''

    nPnts = Ramanshift.shape[0]


    spacing = np.diff(Ramanshift)

    # normalization
    wavenum_corr  = spacing / spacing [norm_pnt]
    temp_xaxis = np.arange(spacing.shape[0])


    # fitting the wavenumber diff
    c = Polynomial.fit(temp_xaxis , wavenum_corr, deg=3)

    # extrpolate the last point
    wavenum_corr = np.append(wavenum_corr, c(spacing.shape[0]+1))

    return wavenum_corr


#############################################################################

def photons_per_unit_wavenum_abs(x,a,T) :
    return (a*599584916*x**2)/(math.exp(0.1438776877e-1*x/T)-1)

#############################################################################
#############################################################################

def gen_C1 (Ramanshift, laser_nm ,  wl_spectra ,   norm_pnt):
    '''Ramanshift = vector, the x-axis in wavenumbers
       norm_pnt =  normalization point (corrections will be set
                                        to unity at this point) '''

    abs_wavenumber = ((1e7/laser_nm)-Ramanshift)*100

    # perform fit
    popt, pcov = curve_fit(photons_per_unit_wavenum_abs,
    abs_wavenumber, wl_spectra, bounds=([0.856e-30,0]], [300., 10000.]))

    print("\t Optimized coefs :", popt)

    # generate fit
    fit = func(Ramanshift, *popt)


    plt.plot(abs_wavenumber, wl_spectra,'o',abs_wavenumber, fit)
    plt.grid()
    plt.show()


    return wavenum_corr

#############################################################################

def gen_C1_with_mask (Ramanshift, norm_pnt):
    '''Ramanshift = vector, the x-axis in wavenumbers
       norm_pnt =  normalization point (corrections will be set
                                        to unity at this point) '''

    nPnts = Ramanshift.shape[0]


    spacing = np.diff(Ramanshift)

    # normalization
    wavenum_corr  = spacing / spacing [norm_pnt]
    temp_xaxis = np.arange(spacing.shape[0])


    # fitting the wavenumber diff
    c = Polynomial.fit(temp_xaxis , wavenum_corr, deg=3)

    # extrpolate the last point
    wavenum_corr = np.append(wavenum_corr, c(spacing.shape[0]+1))

    return wavenum_corr
#############################################################################
