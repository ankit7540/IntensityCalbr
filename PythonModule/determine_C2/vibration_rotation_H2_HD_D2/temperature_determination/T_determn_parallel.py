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

import compute_series_para
import boltzmann_popln as bp
# ------------------------------------------------------

# ------------------------------------------------------
#      RUN PARAMETERS (CHANGE THESE BEFORE RUNNING
#                   OPTIMIZATION
# ------------------------------------------------------

# LOAD EXPERIMENTAL BAND AREA DATA
#  | band area | error |
#  | value | value |
#  | value | value |
#  | value | value |

# without header in the following files

# Change following paths to load expt data

#xaxis = np.loadtxt("Ramanshift_axis")

# Q(J) band intensities --------------------------------

dataD2Q = np.loadtxt("BA_D2_q1.txt")
dataHDQ = np.loadtxt("BA_HD_q1.txt")
dataH2Q = np.loadtxt("BA_H2_q1.txt")

dataD2_Q2 = np.loadtxt("D2_Q2_testdata")

dataD2Q4 = np.loadtxt("BA_D2_q1_J4.txt")
dataD2OS = np.loadtxt("D2_model_O2S0")
# ------------------------------------------------------
# PARALLEL POLARIZATION

# set indices for OJ,QJ and SJ for H2, HD and D2 in the  residual functions below

# ------------------------------------------------------
# ----------------------------------------

# norm type 
# Do not change the variable name on the LHS 
# available norm types : Frobenius, Frobenius_sq, absolute
# lower case :           frobenius, frobenius_sq, absolute
# or abbreviations:      F  , FS , A

norm =  'Frobenius'

# if norm is not set then the default is sum of absolute values 
# See readme for more details

# ----------------------------------------

print('Dimension of input data of Q bands')
print('\t', dataH2Q.shape)
print('\t', dataHDQ.shape)
print('\t', dataD2Q.shape)

print('\t', dataD2_Q2.shape)
print('\t', dataD2Q4.shape)
print('\t', dataD2OS.shape)
# ------------------------------------------------------
# SET  INIT COEFS

temp_init = np.zeros((1))
temp_init[0] = 296

# initial run will be with above parameters
# ------------------------------------------------

# ------------------------------------------------------



print('\t**********************************************************')

print('\t ')
print('\t This module is for determining the temperature from ')
print('\t observed vibration-rotation Raman intensities of H2, HD and D2. ')
print('\t  This module is useful for testing the accuracy of the intensity ')
print('\t  calibration procedure. ')

print('\n\t >> Ratios of all observed Raman intensities are treated here as a matrix. << ')
print('\n\t >> This function deals with parallel polarized intensities. << ')

print('\n\t >> Temperature is the only fit parameter here << ')

print('\n\t This modeule requires edit on line 32 to 74 to ')
print('\n\t  load and set parameters for the analysis.')
print('\t ')
print('\t**********************************************************')
print('\n\t\t Checking imported data and set params')

data_error=0

if isinstance(dataH2Q, np.ndarray):
    print("\t\t ", "dataH2Q found, OK")
else:
    print("\t\t ", "dataH2Q not found.")
    data_error=1
    
if isinstance(dataHDQ, np.ndarray):
    print("\t\t ", "dataHDQ found, OK")
else:
    print("\t\t ", "dataHDQ not found.")
    data_error=1
    
if isinstance(dataD2Q, np.ndarray):
    print("\t\t ", "dataD2Q found, OK")
else:
    print("\t\t ", "dataD2Q not found.")
    data_error=1
    


print('\n\t\t  Analysis parameters:')

print("\t\t Norm (defn of residual): ", norm)



print('\t**********************************************************')
print('\n\t REQUIRED DATA')
print('\t\t\t Ramanshift = vector, the x-axis in relative wavenumbers')
print('\t\t\t band area and error = 2D (2 columns), for H2, HD and D2')
print('\n\t\t\t J_max = scalar, for H2, HD and D2 (to compute reference')
print('\t\t\t\t    spectra), See residual functions')



print('\t**********************************************************')

print('\n\t\t\t  Example:')

print('\t\t\t  run_fit_D2_O2S0 (298 )')

print('\t**********************************************************')

# ------------------------------------------------------
# ------------------------------------------------------

# *******************************************************************

# Set logging ------------------------------------------
fileh = logging.FileHandler('./log_temperature_determination', 'w+')
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
log.error("---Temperature determination from Raman intensities---\n")
log.error("---Parallel polarization---\n")

# ------------------------------------------------------

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

# ------------------------------------------------


def inverse(array):
    """return the inverse square of array, for all elements"""
    return 1 / (array)

# ------------------------------------------------
# *******************************************************************
#     RESIDUAL FUNCTIONS DEFINED BELOW

#     These functions will require edit based on the name of the numpy 
#     array containing the experimental data.
#     Also, the J-index for the rotational level also requires edit 
#     depending on the length of the data ( number of states included).

# *******************************************************************


def residual_Q_D2(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.

    Param : T

    '''

    TK = param
    sosD2 = bp.sumofstate_D2(TK)
    QJ_D2 = 2  # max J value of analyzed Q-bands (change this value depending
    #            on the experimental data
    
    #             compute_series_para.D2_Q1(temperature, J-level, sumofstates)
    computed_D2 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    # ------ D2 ------
    trueR_D2 = gen_intensity_mat(computed_D2, 2)

    # experimental data is used in the following two lines
    #  modify these lines as required (make sure to edit the 
    #                 JMax level defined above as well)
    expt_D2 = gen_intensity_mat(dataD2_Q2, 0)
    errD2_output = gen_weight(dataD2_Q2)
    
    #print(computed_D2.shape, dataD2Q.shape)
    
    errorP = errD2_output

    calc_D2 = clean_mat(trueR_D2)
    expt_D2 = clean_mat(expt_D2)
    errorP = clean_mat(errorP)
    # ----------------

    diffD2 = expt_D2 - calc_D2

    # scale by weights 
    #diffD2 = (np.multiply(errorP , diffD2))

    # remove redundant terms
    diffD2 = clean_mat(diffD2)
    #np.savetxt("diff_D2", diffD2,fmt='%2.4f')


    #  choosing norm ----------
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(diffD2)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(diffD2)))  
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(diffD2)) 
    # -------------------------

    return E

# *******************************************************************



def residual_Q_D2_234(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.

    Param : T

    '''

    TK = param
    sosD2 = bp.sumofstate_D2(TK)
    QJ_D2 = 4  # max J value of analyzed Q-bands (change this value depending
    #            on the experimental data
    
    computed_D2 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    # ------ D2 ------
    
    #print(computed_D2)
    computed_D2=computed_D2[:-2, :]
    #print(computed_D2)
    
    # experimental data is used in the following two lines
    #  modify these lines as required (make sure to edit the 
    #                 JMax level defined above as well)    
    
    dataD2Q = dataD2Q4[:-2, :]  # subset of datapoints included here
    
    #print(computed_D2.shape, dataD2Q.shape)
    
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
    #np.savetxt("diff_D2", diffD2,fmt='%2.4f')

    #  choosing norm ----------
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(diffD2)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(diffD2)))  
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(diffD2)) 
    # -------------------------

    return E

# *******************************************************************


def residual_Q_HD(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.

    Param : T

    '''
    TK = param
    sosHD = bp.sumofstate_HD(TK)
    QJ_HD = 3  # max J value of analyzed Q-bands (change this value depending
    #            on the experimental data
    computed_HD = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)    

    # ------ HD ------
    trueR_HD = gen_intensity_mat(computed_HD, 2)
    
    # experimental data is used in the following two lines
    #  modify these lines as required (make sure to edit the 
    #                 JMax level defined above as well)      
    expt_HD = gen_intensity_mat(dataHDQ, 0)

    errHD_output = gen_weight(dataHDQ)
    
    errorP = errHD_output
    # errorP = 1/(np.divide( errHD_output, expt_HD))

    calc_HD = clean_mat(trueR_HD)
    expt_HD = clean_mat(expt_HD)
    errorP = clean_mat(errorP)
    # ----------------

    diffHD = expt_HD - calc_HD

    # scale by weights
    # diffHD = (np.multiply(errorP , diffHD))

    # remove redundant terms
    diffHD = clean_mat(diffHD)

    #  choosing norm ----------
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(diffHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(diffHD)))  
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(diffHD)) 
    # -------------------------
    return E

# *******************************************************************

def residual_Q_H2(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.

    Param : T

    '''

    TK = param
    sosH2 = bp.sumofstate_H2(TK)
    QJ_H2= 3   # max J value of analyzed Q-bands (change this value depending
    #            on the experimental data
    computed_H2 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # ------ H2 ------
    trueR_H2 = gen_intensity_mat(computed_H2, 2)

    # experimental data is used in the following two lines
    #  modify these lines as required (make sure to edit the 
    #                 JMax level defined above as well)      
    
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

    #  choosing norm ----------
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(diffH2)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(diffH2)))  
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(diffH2)) 
    # -------------------------
    return E


# *******************************************************************


def residual_O2S0_D2(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.

    Param : T

    '''

    TK = param
    sosD2 = bp.sumofstate_D2(TK)
    O1 = compute_series_para.D2_O1(TK, 2, sosD2)
    S1 = compute_series_para.D2_S1(TK, 0, sosD2)
    
    #print(O1,"\n",S1)
    
    O1calc = O1[0][2]
    S1calc = S1[0][2]

    r_calc = (O1calc/ S1calc)
    
    r_expt = (dataD2OS[0]/dataD2OS[1])
    #print(r_calc, r_expt)

    diff = r_expt - r_calc    
    
    return diff**2


# *******************************************************************

# *******************************************************************
# *******************************************************************
# Fit functions
# *******************************************************************
# *******************************************************************


def run_fit_D2(init_T ):
    '''Function performing the actual fit using the residual function
    defined earlier.
    Edit the name of the residual function to choose the different
     residual functions defined above.
     For example, residual_Q_D2 
                  residual_Q_D2_234
    '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    print("\t\t -- Temperature determination -- ")
    print("\t\tNorm (defn of residual): ", norm)  
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
    log.info('\n ***** temperature determination *****')
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************


def run_fit_HD(init_T ):
    '''Function performing the actual fit using the residual function
    defined earlier.
    Edit the name of the residual function to choose the different
     residual functions defined above.
     For example, residual_Q_HD 
    '''

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
    log.info('\n ***** temperature determination *****')    
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************


def run_fit_H2(init_T ):
    '''Function performing the actual fit using the residual function
    defined earlier.
    Edit the name of the residual function to choose the different
     residual functions defined above.
     For example, residual_Q_H2 
    '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    print("\t\t -- Temperature determination -- ")
    print("\t\tNorm (defn of residual): ", norm)      
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
    log.info('\n ***** temperature determination *****')    
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************


def run_fit_D2_O2S0(init_T ):
    '''Function performing the actual fit using the residual  function
    defined earlier named as residual_O2S0_D2 '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    print("\t\t -- Temperature determination -- ")
    print("\t\tNorm (defn of residual): ", norm)      
    #print("Testing the residual function with data")
    print("Initial coef :  T={0},   output = {1}".format(init_T, \
          (residual_O2S0_D2(param_init))))


    print("\nOptimization run: D2, O2S0     \n")
    res = opt.minimize(residual_O2S0_D2, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]

    print("\nOptimized result : T={0}  \n".format(round(optT, 6)))
    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : D2, O2S0  *******')
    log.info('\n ***** temperature determination *****')    
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------
    
# *******************************************************************


def run_fit_D2_234(init_T ):
    '''Function performing the actual fit using the residual function
    defined earlier, named as residual_Q_D2_234 
     which corresponds to Q-band intensities of J-level 2, 3 and 4 in D2'''

    # init_k1 : Intial guess

    param_init = np.array([ init_T  ])
    print("**********************************************************")
    print("\t\t -- Temperature determination -- ")
    print("\t\tNorm (defn of residual): ", norm)      
    #print("Testing the residual function with data")
    print("Initial coef :  T={0},   output = {1}".format(init_T, \
          (residual_Q_D2_234(param_init))))


    print("\nOptimization run: D2, O2S0     \n")
    res = opt.minimize(residual_Q_D2_234, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]

    print("\nOptimized result : T={0}  \n".format(round(optT, 6)))
    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : D2, o2s0  *******')
    log.info('\n ***** temperature determination *****')    
    log.info('\n\t Initial : T = %4.8f \n', init_T  )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f \n', optT  )
    log.info(' *******************************************')
    return res.fun
    # --------------------    

# *******************************************************************
# *******************************************************************


def residual_Q_test(param):
    '''Function which computes the residual   comparing the
    ratio of expt with the corresponding calculated ratios. The calculated
    ratios are computed for given T.
    >> This function is for testing purpose
    Param : T

    '''
    TK = param
    sosHD = bp.sumofstate_HD(TK)
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

    #  choosing norm ----------
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(diffHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(diffHD)))  
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(diffHD)) 
    # -------------------------
    return(E)

# *******************************************************************

# TESTS

# ------------------------------------------------


# checks for input done here

# generate calculated data for the entered J values
def test():
    TK=299

    sosD2 = bp.sumofstate_D2(TK)
    sosHD = bp.sumofstate_HD(TK)
    sosH2 = bp.sumofstate_H2(TK)

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

