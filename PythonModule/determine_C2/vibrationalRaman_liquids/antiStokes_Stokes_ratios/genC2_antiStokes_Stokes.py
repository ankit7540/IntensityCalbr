#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Module describing the weighted non-linear optimization scheme used to
determine the wavelength sensitivity (C2 correction)
of the spectrometer using a  polynomial as a model function.

This scheme is based on using anti-Stokes and Stokes Raman
intensity ratios of the liquids, for a given temperature."""

import os
import sys
import math
import logging
from datetime import datetime
import numpy as np

import scipy.optimize as opt
#import matplotlib.pyplot as plt

# ------------------------------------------------------

# LOAD BAND AREA DATA

#  | band area | error |
# without header in the following files

# Experimental data
# Change following paths

data_CCl4 = np.loadtxt("model_CCl4")
data_C6H6 = np.loadtxt("model_C6H6")
data_C6H12 = np.loadtxt("model_C6H12")
xaxis = np.loadtxt("Wavenumber_axis.dat")

# ------------------------------------------------------

# ------------------------------------------------------
# ------------------------------------------------------
print('Dimension of input data')
#print('\t', data_CCl4.shape)
#print('\t', data_C6H6.shape)
#print('\t', data_C6H12.shape)
# ------------------------------------------------------

# define the known temperature in Kelvin

refT=298

# frequency of the laser (in absolute wavenumbers)
laser_wavenum = 18790.0125

# ------------------------------------------------------
#                COMMON SETTINGS
# ------------------------------------------------------

# Constants ------------------------------
# these are used for scaling the coefs
scale1 = 1e4
scale2 = 1e7
scale3 = 1e9
scale4 = 1e12
scale5 = 1e14
# ----------------------------------------


#      SET INIT COEFS

param_linear=np.zeros((1))
param_linear[0]= -1.0

#----------------------------
param_quadratic=np.zeros((2))
param_quadratic[0]= 1.0
param_quadratic[1]= -0.25

#----------------------------
param_cubic=np.zeros((3))
param_cubic[0]= 1.0
param_cubic[1]= -0.2140
param_cubic[2]= -0.00100

param_quartic=np.zeros((4))
param_quartic[0]= 1.0
param_quartic[1]= -0.2140
param_quartic[2]= -0.00100
param_quartic[3]= -0.000001

# initial run will be with above parameters
# ------------------------------------------------
# Set logging ------------------------------------------
fileh = logging.FileHandler('./logfile_antiStokes_StokesR', 'w+')
formatter = logging.Formatter('%(message)s')
fileh.setFormatter(formatter)

log = logging.getLogger()  # root logger
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(fileh)      # set the new handler
# ------------------------------------------------------

# Logging starts here
logger = logging.getLogger(os.path.basename(__file__))
log.info(logger)
logging.getLogger().setLevel(logging.INFO)
log.warning(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log.warning('\n',)
log.error("------------ Run log ------------\n")
log.error("--- Generate C2 correction from anti-Stokes and Stokes ---")
log.error("--- Raman intensities  ---\n")
# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------
#
# ------------------------------------------------------
print('\t**********************************************************')

print('\t ')
print('\t This module is for generating the wavenumber-dependent')
print('\t intensity correction curve termed as C2 from ')
print('\t  experimental Raman intensities using anti-Stokes')
print('\t and Stokes intensity ratios.  ')

print('\n\t >> The reference intensity ratio is computed  << ')
print('\n\t >> using known temperature provided by the user.' )

print('\n\t This module requires edit on line 24 to 55 to ')
print('\n\t  load and set parameters for the analysis.')
print('\t Residual is defined as sum of squares of the difference. ')
print('\t**********************************************************')
print('\n\t\t  Analysis parameters:')

print("\t\t scaling factors (for c1 to c3) ", scale1, scale2, scale3)

#------------------------------------------------
#############################################################################

# write analysis data to log
log.info('\n\t Input data')

log.info('\n\t Parameters:')
log.info('\t\t scaling factors (c1 to c4):\t %s %s %s %s', scale1, scale2, scale3, scale4)

#############################################################################
#------------------------------------------------
#                COMMON FUNCTIONS
#------------------------------------------------
#------------------------------------------------

def gen_s_linear(data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as line"""
    size=int(data.shape[0] )
    corr = np.zeros( size)

    for i in range(size):
        corr[i] = 1+(param[0]/scale1)*data[i, 0]

    return corr

#------------------------------------------------

def gen_s_quadratic( data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as quadratic polynomial"""

    size=int(data.shape[0] )
    corr = np.zeros( size)

    for i in range(size):
        corr[i] = 1+((param[0]/scale1)*data[i, 0]) \
            + (param[1]/scale2) * (data[i, 0]**2)

    return corr

#------------------------------------------------

def gen_s_cubic( data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as cubic polynomial"""

    size=int(data.shape[0] )
    corr = np.zeros( size)

    for i in range(size):
        corr[i] = 1 + ((param[0]/scale1)*data[i, 0]) \
            + (param[1]/scale2) * (data[i, 0]**2) \
                + (param[2]/scale3) * (data[i, 0]**3)

    return corr

#------------------------------------------------

def gen_s_quartic( data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as quartic polynomial"""

    size=int(data.shape[0] )
    corr = np.zeros( size)

    for i in range(size):
        corr[i] = 1+(param[0]/scale1)*data[i, 0] \
            + (param[1]/scale2) * (data[i, 0]**2) \
                + (param[2]/scale3) * (data[i, 0]**3) \
                    + (param[3]/scale4) * (data[i, 0]**4)

    return corr

#------------------------------------------------


def int_ratio_as_s( freq, iStokes, iantiStokes, refT, laser_abs_wavenum):
    """Generate difference in the intensity ratios from anti-Stokes
    and Stokes ratio (reference) and computed from known temperature
    """
    # define constants --------
    c=2.99792458e+10
    h=6.6260e-34
    k=1.38064e-23
    # --------------------------------
    term_expt = (iantiStokes/iStokes)* \
          ( ((laser_abs_wavenum-freq)**3 )/((laser_abs_wavenum+freq)**3)  )

    term_calc  = math.exp( (-1*h*c*freq) / (k*refT ) )

    #print(  term_expt, term_calc)
    diff = term_expt -  term_calc

    return diff

#------------------------------------------------

def gen_diff(data_expt):
    '''
    Parameters
    ----------
    data_expt : 2D array
        experimental band area (freq area error)

    Returns
    -------
    output : 1D array
        sum of difference for each of the band

    '''

    size = int(data_expt.shape[0]/2)
    output=np.zeros(size)

    for i in range(size):
        v=data_expt[i, 0]

        IntAStokes = data_expt[i, 1]

        index = np.where(np.isclose(data_expt[:,0] , np.abs(v)))

        IntStokes = data_expt[index[0], 1]

        diff = int_ratio_as_s( np.abs(v), IntStokes, IntAStokes,\
                              refT, laser_wavenum)

        output[i] = diff**2
    return np.sum(output )

#------------------------------------------------

# GENERATE  INIT COEFS

param_linear=np.zeros((1))
param_linear[0]= -1.045

#----------------------------
param_quadratic=np.zeros((2))
param_quadratic[0]= -0.923
param_quadratic[1]= -0.254

#----------------------------
param_cubic=np.zeros((3))
param_cubic[0]= -0.9340
param_cubic[1]= -0.2140
param_cubic[2]= -0.00100

param_quartic=np.zeros((4))
param_quartic[0]= -0.9340
param_quartic[1]= -0.2140
param_quartic[2]= -0.00100
param_quartic[3]= -0.000001

#*******************************************************************

#*******************************************************************
#*******************************************************************
# Define the residual function
#*******************************************************************

def residual_linear(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to reference intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x )

    param : c1

    '''

    corr_linear_C6H6 = gen_s_linear(data_C6H6, param )
    corr_linear_C6H12 = gen_s_linear(data_C6H12, param )
    corr_linear_CCl4 = gen_s_linear(data_CCl4, param )

    #make a copy
    dC6H6 = np.copy(data_C6H6)
    dC6H12 = np.copy(data_C6H12)
    dCCl4 = np.copy(data_CCl4)

    # modify the intensity of expt
    dC6H6[:, 1] = np.multiply(corr_linear_C6H6 , dC6H6[:, 1])
    dC6H12[:, 1] = np.multiply(corr_linear_C6H12 , dC6H12[:, 1])
    dCCl4[:, 1] = np.multiply(corr_linear_CCl4 , dCCl4[:, 1])


    diff_C6H6 = gen_diff(dC6H6)
    diff_C6H12 = gen_diff(dC6H12)
    diff_CCl4 = gen_diff(dCCl4)

    E = diff_C6H6 + diff_C6H12 + diff_CCl4
    return E

#*******************************************************************
#*******************************************************************

def residual_quadratic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 )

    param : c1, c2

    '''
    corr_quadratic_C6H6 = gen_s_quadratic(data_C6H6, param )
    corr_quadratic_C6H12 = gen_s_quadratic(data_C6H12, param )
    corr_quadratic_CCl4 = gen_s_quadratic(data_CCl4, param )

    #make a copy
    dC6H6 = np.copy(data_C6H6)
    dC6H12 = np.copy(data_C6H12)
    dCCl4 = np.copy(data_CCl4)

    # modify the intensity of expt
    dC6H6[:, 1] = np.multiply(corr_quadratic_C6H6 , dC6H6[:, 1])
    dC6H12[:, 1] = np.multiply(corr_quadratic_C6H12 , dC6H12[:, 1])
    dCCl4[:, 1] = np.multiply(corr_quadratic_CCl4 , dCCl4[:, 1])


    diff_C6H6 = gen_diff(dC6H6)
    diff_C6H12 = gen_diff(dC6H12)
    diff_CCl4 = gen_diff(dCCl4)

    E = diff_C6H6 + diff_C6H12 + diff_CCl4
    return E

#*******************************************************************
#*******************************************************************

def residual_cubic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : c1, c2, c3

    '''
    corr_cubic_C6H6 = gen_s_cubic(data_C6H6, param )
    corr_cubic_C6H12 = gen_s_cubic(data_C6H12, param )
    corr_cubic_CCl4 = gen_s_cubic(data_CCl4, param )

    #make a copy
    dC6H6 = np.copy(data_C6H6)
    dC6H12 = np.copy(data_C6H12)
    dCCl4 = np.copy(data_CCl4)

    # modify the intensity of expt
    dC6H6[:, 1] = np.multiply(corr_cubic_C6H6 , dC6H6[:, 1])
    dC6H12[:, 1] = np.multiply(corr_cubic_C6H12 , dC6H12[:, 1])
    dCCl4[:, 1] = np.multiply(corr_cubic_CCl4 , dCCl4[:, 1])


    diff_C6H6 = gen_diff(dC6H6)
    diff_C6H12 = gen_diff(dC6H12)
    diff_CCl4 = gen_diff(dCCl4)

    E = diff_C6H6 + diff_C6H12 + diff_CCl4
    return E

#*******************************************************************
#*******************************************************************

def residual_quartic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : c1, c2, c3

    '''
    corr_quartic_C6H6 = gen_s_quartic(data_C6H6, param )
    corr_quartic_C6H12 = gen_s_quartic(data_C6H12, param )
    corr_quartic_CCl4 = gen_s_quartic(data_CCl4, param )

    #make a copy
    dC6H6 = np.copy(data_C6H6)
    dC6H12 = np.copy(data_C6H12)
    dCCl4 = np.copy(data_CCl4)

    # modify the intensity of expt
    dC6H6[:, 1] = np.multiply(corr_quartic_C6H6 , dC6H6[:, 1])
    dC6H12[:, 1] = np.multiply(corr_quartic_C6H12 , dC6H12[:, 1])
    dCCl4[:, 1] = np.multiply(corr_quartic_CCl4 , dCCl4[:, 1])


    diff_C6H6 = gen_diff(dC6H6)
    diff_C6H12 = gen_diff(dC6H12)
    diff_CCl4 = gen_diff(dCCl4)

    E = diff_C6H6 + diff_C6H12 + diff_CCl4
    return E

#***************************************************************
#***************************************************************
#                        Fit functions
#***************************************************************
#***************************************************************

def run_fit_linear ( init_k1 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1  ])
    print("**********************************************************")
    print("\t\t -- Linear fit -- ")

    #print("Testing the residual function with data")
    print("Initial coef :  k1={0} output = {1}".format( init_k1, \
          (residual_linear(param_init))))

    print("\nOptimization run     \n")
    res = opt.minimize(residual_linear, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9,\
                                       'maxiter': 2500})

    print(res)
    optk1 = res.x[0]
    print("\nOptimized result : k1={0} \n".format(round(optk1, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis )     # generate the correction curve

    np.savetxt("correction_linear.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_linear', comments='')

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : c1 = %4.8f\n', init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f\n', optk1 )
    log.info(' *******************************************')
    # --------------------
    return res.fun

#***************************************************************

def run_fit_quadratic ( init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1, init_k2 : Intial guess

    param_init = np.array([   init_k1 , init_k2  ])
    print("**********************************************************")
    print("\t\t -- Quadratic fit -- ")

    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1} output = {2}".format( init_k1, \
         init_k2, (residual_quadratic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quadratic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9,\
                                       'maxiter': 2500})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    print("\nOptimized result : k1={0}, k2={1} \n".format( round(optk1, 6),
                                                          round(optk2, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis ) \
        + ((optk2/scale2)*(xaxis )**2)  # generate the\
    #correction curve

    np.savetxt("correction_quadratic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quadratic', comments='')

    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Quadratic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f\n', init_k1,
             init_k2 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f\n', optk1, optk2 )
    log.info(' *******************************************')
    # --------------------
    return res.fun

#***************************************************************


def run_fit_cubic ( init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
    print("\t\t -- Cubic fit -- ")

    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, output = {3}".format( init_k1, \
         init_k2, init_k3, (residual_cubic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_cubic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, \
                                       'maxiter':2500})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    print("\nOptimized result : k1={0}, k2={1}, k3={2} \n".format( round(optk1, 6),
                                                                  round(optk2, 6),
                                                                  round(optk3, 6)))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis )) \
        + ((optk2/scale2)*(xaxis )**2) \
            + ((optk3/scale3)*(xaxis )**3)

    np.savetxt("correction_cubic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_cubic', comments='')

    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Cubic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f, c3=%4.8f\n', init_k1,
             init_k2, init_k3 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f\n',
             optk1, optk2, optk3 )
    log.info(' *******************************************')
    # --------------------
    return res.fun

#***************************************************************


def run_fit_quartic ( init_k1, init_k2, init_k3, init_k4 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3 , init_k4  ])
    print("**********************************************************")
    print("\t\t -- Quartic fit -- ")

    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, k4={3}, output = {4}".format( init_k1, \
         init_k2, init_k3, init_k4, (residual_quartic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quartic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, \
                                       'maxiter':2500})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    optk4 = res.x[3]
    print("\nOptimized result : k1={0}, k2={1}, k3={2}, k4={3} \n".format(
        round(optk1, 6), round(optk2, 6), round(optk3, 6) ,round(optk4, 6) ))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis ))\
        + ((optk2/scale2)*(xaxis )**2) \
            + ((optk3/scale3)*(xaxis )**3) \
            + ((optk4/scale4)*(xaxis )**4)

    np.savetxt("correction_quartic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quartic', comments='')

    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Cubic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f, c3=%4.8f, c4=%4.8f\n', init_k1,
             init_k2, init_k3, init_k4 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f, c4 = %4.8f\n',
             optk1, optk2, optk3, optk4 )
    log.info(' *******************************************')
    # --------------------
    return res.fun

#***************************************************************



#***************************************************************

def run_all_fit():
    """
    Run fit using predefined initial coefs
    """
    run =1
    if (run == 1):
        resd_1 = 0
        resd_2 = 0
        resd_3 = 0
        resd_4 = 0

        resd_1 = run_fit_linear(  param_linear[0] )
        resd_2 = run_fit_quadratic( param_quadratic[0], param_quadratic[1] )
        resd_3 = run_fit_cubic(  param_cubic[0], param_cubic[1], param_cubic[2] )
        resd_4 = run_fit_quartic( param_quartic[0], param_quartic[1],
                        param_quartic[2], param_quartic[3] )

    out = np.array([resd_1, resd_2, resd_3, resd_4 ])
    return out

#***************************************************************
#***************************************************************
