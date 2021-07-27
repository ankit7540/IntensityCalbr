#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=method-hidden,C0103,E265,E303,R0914,W0621,W503

"""Module describing the weighted non-linear optimization scheme used to
determine the wavelength sensitivity of the spectrometer using a  polynomial
as a model function"""

import os
import sys
import math
import logging
from datetime import datetime
import numpy as np

import scipy.optimize as opt
import matplotlib.pyplot as plt
from common import compute_series_para
from common import utils
# ------------------------------------------------------

# Set logging ------------------------------------------
fileh = logging.FileHandler('logfile.txt', 'w+')
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
# ------------------------------------------------------

# LOAD EXPERIMENTAL BAND AREA DATA
#  | band area | error |
# without header in the following files

# Change following paths
dataH2 = np.loadtxt("./run_parallel/BA_H2_1.txt")
dataHD = np.loadtxt("./run_parallel/BA_HD_1.txt")
dataD2 = np.loadtxt("./run_parallel/BA_D2_1.txt")
xaxis = np.loadtxt("./run_parallel/Ramanshift_axis_para.txt")
# ------------------------------------------------------
# PARALLEL POLARIZATION

# set indices for OJ,QJ and SJ for H2, HD and D2
# these are required for computing spectra for given T

OJ_H2 = 3
QJ_H2 = 4

OJ_HD = 3
QJ_HD = 3
SJ_HD = 2

OJ_D2 = 4
QJ_D2 = 6
SJ_D2 = 3
# ------------------------------------------------------
print('Dimension of input data')
print('\t', dataH2.shape)
print('\t', dataHD.shape)
print('\t', dataD2.shape)
# ------------------------------------------------------
# SET  INIT COEFS

param_linear = np.zeros((2))
param_linear[0] = -1.045

# ----------------------------
param_quadratic = np.zeros((3))
param_quadratic[0] = -0.931
param_quadratic[1] = -0.242

# ----------------------------
param_cubic = np.zeros((4))
param_cubic[0] = -0.9340
param_cubic[1] = -0.2140
param_cubic[2] = -0.00100

param_quartic = np.zeros((5))
param_quartic[0] = -0.9340
param_quartic[1] = -0.2140
param_quartic[2] = -0.00100
param_quartic[3] = -0.000001

# initial run will be with above parameters
# ------------------------------------------------

# ------------------------------------------------------
#      RUN PARAMETERS (CHANGE THESE BEFORE RUNNING
#                   FINAL OPTIMIZATION
# ------------------------------------------------------

# AVAILABLE FUNCTIONS TO USER :

# run_all_fit()
#    Runs the fitting up to quartic polynomial
#    Returns : np array of residuals, with 4 elements


# plot_curves(residual_array="None")
#   Plotting the curves (from fit)
#   and plot the residuals over the number of unknown variables
#   np array of residuals to be passed for plot of residuals

# ------------------------------------------------------
def run_all_fit():
    '''
    Runs the fitting up to quartic polynomial
    Returns : np array of residuals, with 4 elements
    '''
    resd_1 = 0
    resd_2 = 0
    resd_3 = 0
    resd_4 = 0

    run_fit_linear( 1.04586)
    resd_1 = run_fit_linear( -1.04586)

    resd_2 = run_fit_quadratic( -0.285, 0.052)
    resd_2 = run_fit_quadratic( -0.5435, -0.352)

    run_fit_cubic( -0.536, -0.3192, 0.015)
    resd_3 = run_fit_cubic( -0.4840, -0.355, +0.0205)

    resd_4 = run_fit_quartic( -0.483, -0.38, 0.195, +0.02)

    out = np.array([resd_1, resd_2, resd_3, resd_4])
    return out

# *******************************************************************

# ------------------------------------------------------
# ------------------------------------------------------
#                COMMON SETTINGS
# ------------------------------------------------------

# Constants ------------------------------
# these are used for scaling the coefs
scale1 = 1e3
scale2 = 1e6
scale3 = 1e9
scale4 = 1e12
# ----------------------------------------
scenter = 3316.3  # center of the spectra
# used to scale the xaxis

# ------------------------------------------------
#                COMMON FUNCTIONS
# ------------------------------------------------


def gen_intensity_mat(arr, index):
    """To obtain the intensity matrix for the numerator or denominator\
        in the Intensity ratio matrix

        array  =  2D array of data where index column contains the intensity
                  data
        index  =  corresponding to the column which has intensity data

        returns => square matrix of intensity ratio : { I(v1)/I(v2) } """

    spec1D = arr[:, index]
    spec_mat = np.zeros((spec1D.shape[0], spec1D.shape[0]))

    for i in range(spec1D.shape[0]):
        spec_mat[:, i] = spec1D / spec1D[i]

    return spec_mat

# ------------------------------------------------


def clean_mat(square_array):
    """Set the upper triangular portion of square matrix to zero
        including the diagonal
        input = any square array     """
    np.fill_diagonal(square_array, 0)
    return np.tril(square_array, k=0)

# ------------------------------------------------


def gen_weight(expt_data):
    """To generate the weight matrix from the experimental data 2D array
        expt_data  =  2D array of expt data where
                      0 index column is the band area
                      and
                      1 index column is the error
    """
    error_mat = np.zeros((expt_data.shape[0], expt_data.shape[0]))

    for i in range(expt_data.shape[0]):
        for j in range(expt_data.shape[0]):
            error_mat[i, j] = (expt_data[i, 0] / expt_data[j, 0]) \
                * math.sqrt((expt_data[i, 1] / expt_data[i, 0])**2
                            + (expt_data[j, 1] / expt_data[j, 0])**2)
    # return factor * inverse_square(error_mat)
    return inverse_square(error_mat)

# ------------------------------------------------


def inverse_square(array):
    """return the inverse square of array, for all elements"""
    return 1 / (array**2)

# ------------------------------------------------


def scale_elements(array, index_array, factor):
    """scale the elements of array using the index_array and factor"""

    array[index_array] = array[index_array] * factor
    return array

# ------------------------------------------------


def clean_and_scale_elements(array, index_array, factor):
    """scale the elements of array using the index_array and factor"""

    np.fill_diagonal(array, 0)
    array = np.tril(array, k=0)

    array[index_array] = array[index_array] * factor
    return array

# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------


# ------------------------------------------------

def T_independent_index():

    TK = 298  #  --------------------------------
    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2( TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD( TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    calc_298_D2 = gen_intensity_mat(computed_D2, 2)
    calc_298_HD = gen_intensity_mat(computed_HD, 2)

    TK = 1000  #  -------------------------------
    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2( TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD( TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    calc_600_D2=gen_intensity_mat (computed_D2, 2)
    calc_600_HD=gen_intensity_mat (computed_HD, 2)

    diff_D2 = calc_298_D2 - calc_600_D2
    diff_HD = calc_298_HD - calc_600_HD

    cr_D2 = clean_mat(diff_D2)
    cr_HD = clean_mat(diff_HD)

    index_D2 = np.nonzero(np.abs(cr_D2) >1e-10)
    index_HD = np.nonzero(np.abs(cr_HD) >1e-10)

    return index_D2, index_HD
# ------------------------------------------------


def gen_s_linear(computed_data, param):
    """Generate the sensitivity matrix assuming the wavelength
    dependent sensitivity as a line. Elements are the ratio of
    sensitivity at two wavenumber/wavelength points"""

    mat = np.zeros((computed_data.shape[0], computed_data.shape[0]))

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter

            # param[0] = c1

            mat[i, j] = (1 + (param[0] / scale1) * v1)/ \
                (1 + (param[0] / scale1) * v2)

    return mat

# ------------------------------------------------


def gen_s_quadratic(computed_data, param):
    """Generate the sensitivity matrix assuming the wavelength
    dependent sensitivity as a quadratic polynomial. Elements are
    the ratio of sensitivity at two wavenumber/wavelength points"""

    mat = np.zeros((computed_data.shape[0], computed_data.shape[0]))

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter

            # param[0] = c1
            # param[1] = c2

            mat[i, j] = (1 + (param[0] / scale1) * v1 + (param[1]
                                                         / scale2) * v1**2)\
                / (1 + (param[0] / scale1) * v2 + (param[1] / scale2) * v2**2)

    return mat

# ------------------------------------------------


def gen_s_cubic(computed_data, param):
    """Generate the sensitivity matrix assuming the wavelength
    dependent sensitivity as a cubic polynomial. Elements are
    the ratio of sensitivity at two wavenumber/wavelength points"""

    mat = np.zeros((computed_data.shape[0], computed_data.shape[0]))
    # print('norm')
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter

            # param[0] = c1
            # param[1] = c2
            # param[2] = c3

            mat[i, j] = (1 + (param[0] / scale1) * v1 + (param[1] / scale2)
                          * v1**2 + (param[2] / scale3) * v1**3) \
                / (1 + (param[0] / scale1) * v2 + (param[1] / scale2) * v2**2
                   + (param[2] / scale3) * v2**3)

    return mat

# ------------------------------------------------


def gen_s_quartic(computed_data, param):
    """Generate the sensitivity matrix assuming the wavelength
    dependent sensitivity as quartic polynomial. Elements are
    the ratio of sensitivity at two wavenumber/wavelength points"""

    mat = np.zeros((computed_data.shape[0], computed_data.shape[0]))

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter

            # param[0] = c1
            # param[1] = c2
            # param[2] = c3
            # param[3] = c4

            mat[i, j] = (1+(param[0]/scale1)*v1 + (param[1]/scale2)*v1**2 +\
                       (param[2]/scale3)*v1**3 + (param[3]/scale4)*v1**4)/ \
                (1+(param[0]/scale1)*v2 + (param[1]/scale2)*v2**2 \
                 + (param[2]/scale3)*v2**3 + (param[3]/scale4)*v2**4)

    return mat

# ------------------------------------------------
# ------------------------------------------------
# *******************************************************************
#     RESIDUAL FUNCTIONS DEFINED BELOW
# *******************************************************************
t_independent_index = T_independent_index()
indexD2 = t_independent_index[0]
indexHD = t_independent_index[1]
# Using common rotational states from HD and D2

wMat_D2 = 1
wMat_HD = 1


def residual_linear(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x )

    param : T, c1

    '''

    TK = 298

    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)
    expt_D2 = gen_intensity_mat(dataD2, 0)
    I_D2 = np.divide(expt_D2, trueR_D2)
    #I_D2 = clean_mat(I_D2)
    # ----------------

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(dataHD, 0)
    I_HD = np.divide(expt_HD, trueR_HD)
    #I_HD = clean_mat(I_HD)
    # ----------------

    #I_D2[indexD2] = 0
    #I_HD[indexHD] = 0

    # generate the RHS : sensitivity factor
    sD2 = gen_s_linear(computed_D2, param)
    sHD = gen_s_linear(computed_HD, param)

    # weight
    #errD2 = gen_weight(expt_D2)
    #errHD = gen_weight(expt_HD)
    #errD2 = clean_mat(errD2)
    #errD2[indexD2] = 0
    #np.savetxt("error_test", errD2, fmt='%3.3f')

    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)

    # E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    eD2[indexD2] = 0
    eHD[indexHD] = 0
    np.savetxt("errD2_test", eD2, fmt='%3.3f')
    np.savetxt("errHD_test", eHD, fmt='%3.3f')

    #E = np.sum(np.abs(eD2)) + np.sum(np.abs(eHD))
    E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    return(E)

# *******************************************************************
# *******************************************************************


def residual_quadratic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 )

    param : T, c1, c2

    '''
    TK = 298

    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2, SJ_D2,
                                                 sosD2)
    computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD, SJ_HD,
                                                 sosHD)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)
    expt_D2 = gen_intensity_mat(dataD2, 0)
    I_D2 = np.divide(expt_D2, trueR_D2)

    # ----------------

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(dataHD, 0)
    I_HD = np.divide(expt_HD, trueR_HD)

    # ----------------

    # I_HD = clean_and_scale_elements(I_HD, index_HD, 2)
    # I_D2 = clean_and_scale_elements(I_D2, index_D2, 2)

    # generate the RHS : sensitivity factor
    sD2 = gen_s_quadratic(computed_D2, param)
    sHD = gen_s_quadratic(computed_HD, param)

    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)

    # E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    eD2[indexD2] = 0
    eHD[indexHD] = 0

    #E = np.sum(np.abs(eD2)) + np.sum(np.abs(eHD))
    E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    return(E)

# *******************************************************************
# *******************************************************************


def residual_cubic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : T, c1, c2, c3

    '''
    TK = 298

    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)
    expt_D2 = gen_intensity_mat(dataD2, 0)
    I_D2 = np.divide(expt_D2, trueR_D2)
    #I_D2 = clean_mat(I_D2)
    # ----------------

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(dataHD, 0)
    I_HD = np.divide(expt_HD, trueR_HD)
    #I_HD = clean_mat(I_HD)
    # ----------------

    # I_HD = clean_and_scale_elements(I_HD, index_HD, 2)
    # I_D2 = clean_and_scale_elements(I_D2, index_D2, 2)

    # generate the RHS : sensitivity factor
    sD2 = gen_s_cubic(computed_D2, param)
    sHD = gen_s_cubic(computed_HD, param)

    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)

    # E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    eD2[indexD2] = 0
    eHD[indexHD] = 0

    #E = np.sum(np.abs(eD2)) + np.sum(np.abs(eHD))
    E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    return(E)

# *******************************************************************
# *******************************************************************


def residual_quartic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 + c4*x**4 )

    param : T, c1, c2, c3, c4

    '''
    TK = 298

    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)
    expt_D2 = gen_intensity_mat(dataD2, 0)
    I_D2 = np.divide(expt_D2, trueR_D2)
    #I_D2 = clean_mat(I_D2)
    # ----------------

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(dataHD, 0)
    I_HD = np.divide(expt_HD, trueR_HD)
    #I_HD = clean_mat(I_HD)
    # ----------------

    # generate the RHS : sensitivity factor
    sD2 = gen_s_quartic(computed_D2, param)
    sHD = gen_s_quartic(computed_HD, param)

    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)


    eD2[indexD2] = 0
    eHD[indexHD] = 0

    #E = np.sum(np.abs(eD2)) + np.sum(np.abs(eHD))
    E = np.sum(np.square(eD2)) + np.sum(np.square(eHD))

    return(E)

# *******************************************************************
# *******************************************************************
# Fit functions
# *******************************************************************
# *******************************************************************


def run_fit_linear(init_k1):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0} output = {1}".format( init_k1, \
          (residual_linear(param_init))))

    print("\nOptimization run: Linear     \n")
    res = opt.minimize(residual_linear, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optk1 = res.x[0]
    print("\nOptimized result : k1={0} \n".format(round(optk1, 6)))

    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  # generate the correction curve

    np.savetxt("correction_linear.txt", correction_curve, fmt='%2.9f',\
               header='corrn_curve_linear')

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : c1 = %4.8f\n', init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f\n', optk1 )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
# *******************************************************************

def run_fit_quadratic ( init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([  init_k1 , init_k2  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1} output = {2}".format( init_k1, \
         init_k2, (residual_quadratic(param_init))))


    print("\nOptimization run: Quadratic     \n")
    res = opt.minimize(residual_quadratic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':1500})

    print(res)
    optk1 = res.x[0]
    optk2 = res.x[1]
    print("\nOptimized result : k1={0}, k2={1} \n".format(round(optk1, 6), \
                                                          round(optk2, 6)))

     # generate the correction curve
    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  +(optk2/scale2)\
                                                       * (xaxis-scenter)**2

    np.savetxt("correction_quadratic.txt", correction_curve, fmt='%2.9f',\
               header='corrn_curve_quadratic' )

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Quadratic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f\n', init_k1, init_k2 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f\n', optk1, optk2 )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
# *******************************************************************

def run_fit_cubic ( init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef : k1={0}, k2={1}, k3={2}, output = {3}".\
          format( init_k1, init_k2, init_k3, (residual_cubic(param_init))))


    print("\nOptimization run : Cubic     \n")
    res = opt.minimize(residual_cubic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':2500})

    print(res)
    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    print("\nOptimized result : k1={0}, k2={1}, k3={2} \n".\
          format(  round(optk1, 6), round(optk2, 6),\
                 round(optk3, 6)))

    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  + (optk2/scale2)*(xaxis-scenter)**2  +\
        +(optk3/scale3)*(xaxis-scenter)**3 # generate the correction curve

    np.savetxt("correction_cubic.txt", correction_curve, fmt='%2.9f',\
               header='corrn_curve_cubic')

    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Cubic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f\n', init_k1,\
             init_k2, init_k3 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f\n',\
             optk1, optk2, optk3 )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
# *******************************************************************

def run_fit_quartic ( init_k1, init_k2, init_k3, init_k4 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3, init_k4])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef : k1={0}, k2={1}, k3={2}, k4={3} output = {4}".\
          format(init_k1, init_k2, init_k3, init_k4, (residual_cubic(param_init))))


    print("\nOptimization run : Quartic     \n")
    res = opt.minimize(residual_quartic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':2000})

    print(res)
    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    optk4 = res.x[3]
    print("\nOptimized result : k1={0}, k2={1}, k3={2}, k4={3} \n".\
          format( round(optk1, 6), round(optk2, 6),\
                 round(optk3, 6), round(optk4, 6)))

    # generate the correction curve
    correction_curve= 1+(optk1/scale1)*(xaxis-scenter)  +(optk2/scale2)*(xaxis-scenter)**2  +\
        +(optk3/scale3)*(xaxis-scenter)**3 +(optk4/scale4)*(xaxis-scenter)**4

    np.savetxt("correction_quartic.txt", correction_curve, fmt='%2.9f',\
               header='corrn_curve_quartic')

    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Quartic  *******')
    log.info('\n\t Initial : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f, c4 = %4.8f\n', init_k1,\
             init_k2, init_k3, init_k4 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f, c2 = %4.8f, c3 = %4.8f, c4 = %4.8f\n',\
             optk1, optk2, optk3, optk4 )
    log.info(' *******************************************')
    return res.fun
    # --------------------


# *******************************************************************
# *******************************************************************

def plot_curves(residual_array="None"):
    '''
    option = 1 : plot
           = 0 : do not plot

    '''
    # Load the saved correction curves for  plotting
    # outputs from last run will be loaded
    correction_line = np.loadtxt("./correction_linear.txt", skiprows=1)
    correction_quad = np.loadtxt("./correction_quadratic.txt", skiprows=1)
    correction_cubic = np.loadtxt("./correction_cubic.txt", skiprows=1)
    correction_quartic = np.loadtxt("./correction_quartic.txt", skiprows=1)

    # ---------------------------------------------------------------------

    # Plotting the data

    txt = ("*Generated from 'wavelength_sensitivity.py' on the\
          \nGitHub Repository: IntensityCalbr ")

    # FIGURE 0 INITIALIZED

    plt.figure(0)
    ax0 = plt.axes()
    plt.title('Fitting result', fontsize=22)

    plt.plot(xaxis,  correction_line, 'r', linewidth=3, label='line_fit')
    plt.plot(xaxis,  correction_quad, 'g', linewidth=4.2, label='quad_fit')
    plt.plot(xaxis,  correction_cubic, 'b--', linewidth=2.65, label='cubic_fit')
    plt.plot(xaxis,  correction_quartic, 'k--', linewidth=2.65, label='quartic_fit')

    plt.xlabel('Wavenumber / $cm^{-1}$', fontsize=20)
    plt.ylabel('Relative sensitivity', fontsize=20)
    plt.grid(True , which='both')  # ax.grid(True, which='both')

    # change following as needed
    ax0.tick_params(axis='both', labelsize =20)

    xmin = np.amin(xaxis-10)
    xmax = np.amax(xaxis+10)

    plt.xlim((xmax, xmin)) #  change this if the xlimit is not correct
    ax0.set_ylim([0, 2.1]) # change this if the ylimit is not enough

    ax0.minorticks_on()
    ax0.tick_params(which='minor', right='on')
    ax0.tick_params(axis='y', labelleft='on', labelright='on')
    plt.text(0.05, 0.0095, txt, fontsize=6, color="dimgrey",
             transform=plt.gcf().transFigure)
    plt.legend(loc='upper left', fontsize=16)


    # markers for the band which are analyzed at present
    plt.plot(freqD2, dummyD2, 'kD' )
    plt.plot(freqHD, dummyHD, 'ms' )

    if type(residual_array) != str:
        if isinstance(residual_array, (list,np.ndarray)):
        # ---------------------------------------------------------------------
            # FIGURE 1 INITIALIZED

            xv = np.arange(1, 5, 1)
            plt.figure(1)
            ax1 = plt.axes()
            plt.title('Residuals', fontsize=21)
            plt.plot(xv, residual_array, 'ro--')
            plt.xlabel('degree of polynomial', fontsize=20)
            plt.ylabel('Residual', fontsize=20)
            plt.grid(True)  # ax.grid(True, which='both')
            ax1.tick_params(axis='both', labelsize=20)
        else:
            print('\tWrong type of parameter : residual_array. Quitting plotting.')
    else:
        print('\tResidual array not provided. plot of residuals not made!')

# ***************************************************************

# ******************** CHECKS FOR INPUTS ************************
# ***************************************************************


# ------------------------------------------------

wMat_D2 = 1
wMat_HD = 1.2
wMat_H2 = 1

# checks for input done here

# generate calculated data for the entered J values
TK=299
sosD2 = compute_series_para.sumofstate_D2(TK)
sosHD = compute_series_para.sumofstate_HD(TK)
sosH2 = compute_series_para.sumofstate_H2(TK)

computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2, SJ_D2, sosD2)
computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD, SJ_HD, sosHD)
computed_H2 = compute_series_para.spectra_H2_c(TK, OJ_H2, QJ_H2, sosH2)

# checks for dimension match done here
if (computed_D2.shape[0] != dataD2.shape[0]):
    print('D2 : Dimension of input data does not match with the calculated\
           spectra. Check input expt data or the J-indices entered.')
    sys.exit("\tError: Quitting.")

if (computed_HD.shape[0] != dataHD.shape[0]):
    print('H2 : Dimension of input data does not match with the calculated\
           spectra. Check input expt data or the J-indices entered.')
    sys.exit("\tError: Quitting.")

if (computed_H2.shape[0] != dataH2.shape[0]):
    print('H2 : Dimension of input data does not match with the calculated\
           spectra. Check input expt data or the J-indices entered.')
    sys.exit("\tError: Quitting.")

# ------------------------------------------------

# TESTS

trueR_D2 = gen_intensity_mat(computed_D2, 2)
expt_D2 = gen_intensity_mat(dataD2, 0)

trueR_HD = gen_intensity_mat(computed_HD, 2)
expt_HD = gen_intensity_mat(dataHD, 0)

I_D2 = np.divide(expt_D2, trueR_D2)
I_HD = np.divide(expt_HD, trueR_HD)

print(I_D2.shape)
#I_D2 = clean_mat(I_D2)
#I_HD = clean_mat(I_HD)

index = T_independent_index()
indexD2=index[0]
indexHD=index[1]

I_D2[indexD2] = 0
I_HD[indexHD] = 0
a=I_D2
#print(I_D2)

errD2_output = gen_weight(dataD2)
errHD_output = gen_weight(dataHD)



sD2 = gen_s_linear(computed_D2, param_linear)
sHD = gen_s_linear(computed_HD, param_linear)

eD2 = (np.multiply(errD2_output, I_D2) - sD2)
eHD = (np.multiply(errHD_output, I_HD) - sHD)


eD2 = clean_mat(eD2)
eHD = clean_mat(eHD)


resd_lin = residual_linear(param_linear)
resd_quad = residual_quadratic(param_quadratic)
resd_cubic = residual_cubic(param_cubic)
resd_quar = residual_quartic(param_quartic)

print('Value of residuals with default coefs are')
print('\t linear \t:', resd_lin)
print('\t quadratic \t:', resd_quad)
print('\t cubic  \t:', resd_cubic)
print('\t quartic \t:', resd_quar)
# ********************************************************************

test_mat = np.arange(196).reshape(14,14)
test_mat = clean_mat(test_mat)
test_mat[indexD2] = 0

# -----------------------------------------------------
#  Dummy value for plot (frequencies)
val=0.125
dummyD2 = np.full(len(computed_D2), val)
dummyHD = np.full(len(computed_HD), val)
dummyH2 = np.full(len(computed_H2), val)

# -----------------------------------------------------
freqD2 = computed_D2[:,1]
freqHD = computed_HD[:,1]

# ------------------------------------------------
# For setting the bands which are not analyzed to nan in dummy array
#  dummy array used for plot
def T_independent_D2_set_nan(array):
    '''
    elements in 'array' which correspond to frequencies not
    analyzed are set to nan, for D2
    '''
    TK = 298  #  --------------------------------
    sosD2 = compute_series_para.sumofstate_D2(TK)

    computed_D2 = compute_series_para.spectra_D2( TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    calc_298_D2 = gen_intensity_mat(computed_D2, 2)

    TK = 1000  #  -------------------------------
    sosD2 = compute_series_para.sumofstate_D2(TK)
    computed_D2 = compute_series_para.spectra_D2( TK, OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    calc_600_D2=gen_intensity_mat (computed_D2, 2)

    diff_D2 = calc_298_D2 - calc_600_D2
    cr_D2 = clean_mat(diff_D2)

    return set_nan_if_foundzero(cr_D2, array)
# ------------------------------------------------
# ------------------------------------------------
# For setting the bands which are not analyzed to nan in dummy array
#  dummy array used for plot
def T_independent_HD_set_nan( array):
    '''
    elements in 'array' which correspond to frequencies not
    analyzed are set to nan, for HD
    '''
    TK = 298  #  --------------------------------

    sosHD = compute_series_para.sumofstate_HD(TK)
    computed_HD = compute_series_para.spectra_HD( TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    calc_298_HD = gen_intensity_mat(computed_HD, 2)
    TK = 1000  #  -------------------------------
    sosHD = compute_series_para.sumofstate_HD(TK)
    computed_HD = compute_series_para.spectra_HD( TK, OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)

    calc_600_HD=gen_intensity_mat (computed_HD, 2)
    diff_HD = calc_298_HD - calc_600_HD
    cr_HD = clean_mat(diff_HD)

    return set_nan_if_foundzero(cr_HD, array)

# ------------------------------------------------

def set_nan_if_foundzero(matrix, output):
    # check over cols

    # scheme: check over for max in a col, if max is zero
    #   then that freq was unused in the analysis
    #and similarly for rows

    for i in range(len(output)):
        col=matrix[:,i]
        val=np.amax(col)
        if (np.abs(val)<1e-7):
            output[i]=np.nan

    # check over rows
    for i in range(len(output)):
        val=np.amax(matrix[i, :])
        if (np.abs(val)<1e-7):
            output[i]=np.nan
    return output
# -----------------------------------------------------
#print (freqD2, T_independent_HD(  freqD2  )  )
dummyHD = T_independent_HD_set_nan( dummyHD )
#print(freqHD,  T_independent_HD( dummyHD ))
print(freqHD.shape[0], dummyHD.shape[0])

dummyD2 = T_independent_D2_set_nan( dummyD2 )
print(freqD2, dummyD2, dummyD2.shape[0])
