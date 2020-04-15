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
# ------------------------------------------------------

# Set logging ------------------------------------------
fileh = logging.FileHandler('./log.txt', 'w+')
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

# Q(J) band intensities --------------------------------

dataD2Q = np.loadtxt("./run_parallel_afterC2/BA_D2_q1.txt")
dataHDQ = np.loadtxt("./run_parallel_afterC2/BA_HD_q1.txt")
dataH2Q = np.loadtxt("./run_parallel_afterC2/BA_H2_q1.txt")

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


print('Dimension of input data of Q bands')
print('\t', dataH2Q.shape)
print('\t', dataHDQ.shape)
print('\t', dataD2Q.shape)
# ------------------------------------------------------
# SET  INIT COEFS

temp_init = np.zeros((1))
temp_init[0] = 298

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
    inputT = 300
    run_fit_D2(inputT)
    run_fit_HD(inputT)
    run_fit_H2(inputT)
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
# *******************************************************************

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
    return inverse(error_mat)

# ------------------------------------------------


def inverse_square(array):
    """return the inverse square of array, for all elements"""
    return 1 / (array**2)


def inverse (array):
    """return the inverse square of array, for all elements"""
    return 1 / (array)

# ------------------------------------------------


def scale_elements(array, index_array, factor):
    """scale the elements of array using the index_array and factor"""

    array[index_array] = array[index_array] * factor
    return array

# ------------------------------------------------

# *******************************************************************
#     RESIDUAL FUNCTIONS DEFINED BELOW
# *******************************************************************


def residual_Q_D2(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated 
    ratios are computed for given T.

    Param : T

    '''

    TK = param 
    sosD2 = compute_series_para.sumofstate_D2(TK)
    QJ_D2=4  # max J value of analyzed Q-bands
    computed_D2 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)
    expt_D2 = gen_intensity_mat(dataD2Q, 0)
    
    errD2_output = gen_weight(dataD2Q)
    errorP = errD2_output  
    
    #np.savetxt("exptD2",clean_mat(expt_D2),fmt='%2.4f')
    #np.savetxt("errD2",clean_mat(errorP),fmt='%2.4f')
    
    calc_D2 = clean_mat(trueR_D2)
    expt_D2 = clean_mat(expt_D2)
    errorP = clean_mat(errorP)
    # ----------------
    
    diffD2 = expt_D2 - calc_D2
    
    # scale by weights
    #diffD2 = (np.multiply(errorP , diffD2))
    
    # remove redundant terms
    diffD2 = clean_mat(diffD2)
    np.savetxt("diff_D2", diffD2,fmt='%2.4f')

    # return the residual
    E=np.sum(np.square(diffD2))

    return E

# *******************************************************************


def residual_Q_HD(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated 
    ratios are computed for given T.

    Param : T

    '''
    TK = param 
    sosHD = compute_series_para.sumofstate_HD(TK)
    QJ_HD=3
    computed_HD = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(dataHDQ, 0)
    
    errHD_output = gen_weight(dataHDQ)
    errorP = errHD_output
    #errorP = 1/(np.divide( errHD_output, expt_HD))
    
    
    calc_HD = clean_mat(trueR_HD)
    expt_HD = clean_mat(expt_HD)
    errorP = clean_mat(errorP)
    # ----------------
    
    diffHD = expt_HD - calc_HD
    
    # scale by weights
    #diffHD = (np.multiply(errorP , diffHD))
    
    # remove redundant terms
    diffHD = clean_mat(diffHD)

    # return the residual
    E=np.sum(np.square(diffHD))
    return E

# *******************************************************************   

def residual_Q_H2(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated 
    ratios are computed for given T.

    Param : T

    '''

    TK = param 
    sosH2 = compute_series_para.sumofstate_H2(TK)
    QJ_H2=3
    computed_H2 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # ------ H2 ------
    trueR_H2 = gen_intensity_mat(computed_H2, 2)
    expt_H2 = gen_intensity_mat(dataH2Q, 0)
    
    errH2_output = gen_weight(dataH2Q)
    errorP = errH2_output
    #errorP = 1/(np.divide( errH2_output, expt_H2))
    
    calc_H2 = clean_mat(trueR_H2)
    expt_H2 = clean_mat(expt_H2)
    errorP = clean_mat(errorP)
    # ----------------
    
    diffH2 = expt_H2 - calc_H2
    
    # scale by weights
    #diffH2 = (np.multiply(errorP , diffH2))
    
    # remove redundant terms
    diffH2 = clean_mat(diffH2)

    # return the residual
    E=np.sum(np.square(diffH2))
    return E

# *******************************************************************    
# *******************************************************************
# Fit functions
# *******************************************************************
# *******************************************************************


def run_fit_D2(init_T ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0},   output = {1}".format(init_T, \
          (residual_Q_D2(param_init))))


    print("\nOptimization run: D2     \n")
    res = opt.minimize(residual_Q_D2, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]

    print("\nOptimized result : T={0}  \n".format(round(optT, 6)))
    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : D2  *******')
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
    

def run_fit_HD(init_T ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0},   output = {1}".format(init_T, \
          (residual_Q_HD(param_init))))


    print("\nOptimization run: HD     \n")
    res = opt.minimize(residual_Q_HD, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]

    print("\nOptimized result : T={0}  \n".format(round(optT, 6)))
    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : HD  *******')
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************


def run_fit_H2(init_T ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0},   output = {1}".format(init_T, \
          (residual_Q_H2(param_init))))


    print("\nOptimization run: H2     \n")
    res = opt.minimize(residual_Q_H2, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]

    print("\nOptimized result : T={0}  \n".format(round(optT, 6)))
    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : H2  *******')
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
# *******************************************************************

# TESTS

# ------------------------------------------------


# checks for input done here

# generate calculated data for the entered J values
TK=299
sosD2 = compute_series_para.sumofstate_D2(TK)
sosHD = compute_series_para.sumofstate_HD(TK)
sosH2 = compute_series_para.sumofstate_H2(TK)

computed_D2 = compute_series_para.spectra_D2(TK, OJ_D2, QJ_D2, SJ_D2, sosD2)
computed_HD = compute_series_para.spectra_HD(TK, OJ_HD, QJ_HD, SJ_HD, sosHD)
computed_H2 = compute_series_para.spectra_H2_c(TK, OJ_H2, QJ_H2, sosH2)

# ------------------------------------------
# ratio 
trueR_D2 = gen_intensity_mat (computed_D2, 2)
expt_D2 = gen_intensity_mat (dataD2, 0)


trueR_HD = gen_intensity_mat (computed_HD, 2)
expt_HD = gen_intensity_mat (dataHD, 0)


trueR_H2 = gen_intensity_mat (computed_H2, 2)
expt_H2 = gen_intensity_mat (dataH2, 0)

# ------------------------------------------

errH2_output = gen_weight(dataH2)
errHD_output = gen_weight(dataHD)
errD2_output = gen_weight(dataD2)

diffD2 = trueR_D2 - expt_D2
diffHD = trueR_HD - expt_HD 
diffH2 = trueR_H2 - expt_H2 

eD2 = (np.multiply(errD2_output, diffD2))
eHD = (np.multiply(errHD_output, diffHD))
eH2 = (np.multiply(errH2_output, diffH2))

eD2 = clean_mat(eD2)
eHD = clean_mat(eHD)
eH2 = clean_mat(eH2)

print(np.sum(np.square(eD2)))
print(np.sum(np.square(eHD))) 
print(np.sum(np.square(eH2)))
# ------------------------------------------
test = np.loadtxt("./run_parallel/test_data.txt")

def residual_Q_test(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated 
    ratios are computed for given T.

    Param : T

    '''
    TK = param 
    sosHD = compute_series_para.sumofstate_HD(TK)
    QJ_HD=3
    computed_HD = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    expt_HD = gen_intensity_mat(test, 0)
    
    errHD_output = gen_weight(dataHDQ)
    errorP = errHD_output
    
    
    calc_HD = clean_mat(trueR_HD)
    expt_HD = clean_mat(expt_HD)
    errorP = clean_mat(errorP)
    # ----------------
    
    diffHD = expt_HD - calc_HD
    
    # scale by weights
    diffHD = (np.multiply(errorP , diffHD))
    
    # remove redundant terms
    diffHD = clean_mat(diffHD)

    # return the residual
    #E=np.sum(np.square(diffHD))/1e2
    return(errorP)

# *******************************************************************   

