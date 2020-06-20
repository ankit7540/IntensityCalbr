#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=method-hidden,C0103,E265,E303,R0914,W0621,E305

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



# ------------------------------------------------------

# Set logging ------------------------------------------
fileh = logging.FileHandler('./expt_data_wch/logfile_para.txt', 'w+')
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

# LOAD BAND AREA DATA

#  | band area | error |
# without header in the following files

# Experimental data
# Change following paths
data_CCl4 = np.loadtxt("./expt_data_wch/BA_CCl4.txt")
data_C6H6 = np.loadtxt("./expt_data_wch/BA_C6H6.txt")
data_C6H12 = np.loadtxt("./expt_data_wch/BA_C6H12.txt")
xaxis = np.loadtxt("./expt_data_wch/Wavenumber_axis_pa.txt")

# ------------------------------------------------------

# Reference data
# Change following paths
ref_CCl4 = np.loadtxt("./expt_data_wch/BA_ref_CCl4.txt")
ref_C6H6 = np.loadtxt("./expt_data_wch/BA_ref_C6H6.txt")
ref_C6H12 = np.loadtxt("./expt_data_wch/BA_ref_C6H12.txt")

# ------------------------------------------------------
# ------------------------------------------------------
print('Dimension of input data')
print('\t', data_CCl4.shape)
print('\t', data_C6H6.shape)
print('\t', data_C6H12.shape)
# ------------------------------------------------------
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

scenter = 0 # center of the spectra

#      SET INIT COEFS

param_linear=np.zeros((1))
param_linear[0]= -1.045

#----------------------------
param_quadratic=np.zeros((2))
param_quadratic[0]= -0.931
param_quadratic[1]= -0.242

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

# Example :
#    resd = run_all_fit()
#    plot_curves(resd)
#
#    resd = run_all_fit() ; plot_curves(resd) ;
#
# ------------------------------------------------------

#------------------------------------------------
#                COMMON FUNCTIONS
#------------------------------------------------
def gen_intensity_mat (arr, index):
    """To obtain the intensity matrix for the numerator or denominator\
        in the Intensity ratio matrix

        array  =  2D array of data where index column contains the intensity data
        index  =  corresponding to the column which has intensity

        returns => square matrix of intensity ratio : { I(v1)/I(v2) } """

    spec1D=arr[:, index]
    spec_mat=np.zeros((spec1D.shape[0],spec1D.shape[0]))

    for i in range(spec1D.shape[0]):
        spec_mat[:,i]=spec1D/spec1D[i]

    return spec_mat

#------------------------------------------------

def scale_opp_diagonal (square_array, multiplicative_factor):
    """Scale the elements, scale down the non-diagonal elements if
    value larger than 0.4 of the max value,
    opposite diagonal of a sqaure array with the
    multiplicative factor"""

    Y = square_array[:, ::-1]

    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            val=Y [i,j]
            if (val> (0.40*np.amax(square_array)) ):
                Y [i,j]=Y[i,j]/200
            if (i==j):
                Y [i,j]=Y[i,j]*multiplicative_factor

    return  Y[:, ::-1]
#------------------------------------------------


def clean_mat(square_array):
    """Set the upper triangular portion of square matrix to zero
        input = any square array     """
    np.fill_diagonal(square_array, 0)
    return ( np.tril(square_array, k=0) )

#------------------------------------------------

def gen_weight(expt_data, factor):
    """To generate the weight matrix from the experimental data 2D array
        expt_data  =  2D array of expt data where
                      0th column is the band area

                      1st column is the error
    """
    error_mat=np.zeros((expt_data.shape[0],expt_data.shape[0]))

    for i in range(expt_data.shape[0]):
        for j in range(expt_data.shape[0]):
            error_mat [i,j]=(expt_data[i,0]/expt_data[j,0])*\
                math.sqrt( (expt_data[i,1]/expt_data[i,0])**2 + \
                     (expt_data[j,1]/expt_data[j,0])**2   )


    return  inverse_square(error_mat)  #  factor not used
    #return  np.abs(error_mat)

#------------------------------------------------

#@utils.MeasureTime
def inverse_square(array):
    """return the inverse square of array, for all elements"""
    return 1/(array**2)

#------------------------------------------------

def gen_s_linear(computed_data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity modeled as line"""
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 0] - scenter  # col 0 has position
            v2 = computed_data[j, 0] - scenter  # col 0 has position

            #print(v1, v2)

            c1 = param[0]

            mat [i,j]=(1+ (c1/scale1)*v1 )/ \
                (1+ (c1/scale1)*v2)

    return mat

#------------------------------------------------

def gen_s_quadratic(computed_data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as quadratic polynomial"""
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 0] - scenter  # col 0 has position
            v2 = computed_data[j, 0] - scenter  # col 0 has position

            #print(v1, v2)

            c1 = param[0]
            c2 = param[1]

            mat [i,j]=(1+((c1/scale1)*v1) + (c2/scale2)*v1**2 )/ \
                ( (1+(c1/scale1)*v2) + (c2/scale2)*v2**2 )

    return mat

#------------------------------------------------

def gen_s_cubic(computed_data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as cubic polynomial"""
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 0] - scenter  # col 0 has position
            v2 = computed_data[j, 0] - scenter  # col 0 has position

            #print(v1, v2)

            c1 = param[0]
            c2 = param[1]
            c3 = param[2]

            mat [i,j]=(1+(c1/scale1)*v1 + (c2/scale2)*v1**2 +\
                       (c3/scale3)*v1**3 )/ \
                (1+(c1/scale1)*v2 + (c2/scale2)*v2**2 \
                 + (c3/scale3)*v2**3 )

    return mat

#------------------------------------------------

def gen_s_quartic(computed_data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as quartic polynomial"""
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)

    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 0] - scenter  # col 0 has position
            v2 = computed_data[j, 0] - scenter  # col 0 has position

            c1 = param[0]
            c2 = param[1]
            c3 = param[2]
            c4 = param[3]

            mat [i,j]=(1+(c1/scale1)*v1 + (c2/scale2)*v1**2 +\
                       (c3/scale3)*v1**3 + (c4/scale4)*v1**4  )/ \
                (1+(c1/scale1)*v2 + (c2/scale2)*v2**2 \
                 + (c3/scale3)*v2**3 + (c4/scale4)*v2**4 )

    return mat

#------------------------------------------------
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
#  GENERATE WEIGHT MATRICES

wMat_C6H6 = gen_weight(data_C6H6, 1)
wMat_C6H6 = clean_mat(wMat_C6H6)

wMat_C6H12 = gen_weight(data_C6H12, 1)
wMat_C6H12 = clean_mat(wMat_C6H12)

wMat_CCl4 = gen_weight(data_CCl4, 1)
wMat_CCl4 = clean_mat(wMat_CCl4)


print(np.amax(wMat_C6H6))
print(np.amax(wMat_C6H12))
print(np.amax(wMat_CCl4))

wMat_C6H6 = np.divide(wMat_C6H6, np.amax(wMat_C6H6))
wMat_C6H12 = np.divide(wMat_C6H12, np.amax(wMat_C6H12))
wMat_CCl4 = np.divide(wMat_CCl4, np.amax(wMat_CCl4))

#wMat_HD = gen_weight(dataHD, 0.2)
#wMat_D2 = gen_weight(dataD2, 0.2)

#print(wMat_H2 )

#wMat_H2 = np.divide(wMat_H2, 300)
#wMat_HD = np.divide(wMat_HD, 300)
#wMat_D2 = np.divide(wMat_D2, 300)

#wMat_H2=scale_opp_diagonal (wMat_H2, 500)
#wMat_HD=scale_opp_diagonal (wMat_HD, 500)
#wMat_D2=scale_opp_diagonal (wMat_D2, 500)

#wMat_C6H6=1
#wMat_C6H12=1
#wMat_CCl4=1
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

    # ------ C6H6 ------
    trueR_C6H6 = gen_intensity_mat (ref_C6H6, 1)  # col 1 has ref. area
    expt_C6H6 = gen_intensity_mat (data_C6H6, 0)
    I_C6H6 = np.divide(expt_C6H6, trueR_C6H6)
    I_C6H6 = clean_mat(I_C6H6)
    # ----------------

    # ------ C6H12 ------
    trueR_C6H12 = gen_intensity_mat (ref_C6H12, 1)
    expt_C6H12 = gen_intensity_mat (data_C6H12, 0)
    I_C6H12 = np.divide(expt_C6H12, trueR_C6H12)
    I_C6H12 = clean_mat(I_C6H12)
    # ----------------

    # ------ CCl4 ------
    trueR_CCl4 = gen_intensity_mat (ref_CCl4, 1)
    expt_CCl4 = gen_intensity_mat (data_CCl4, 0)
    I_CCl4 = np.divide(expt_CCl4, trueR_CCl4)
    I_CCl4 = clean_mat(I_CCl4)
    # ----------------

    # generate the RHS : sensitivity factor
    sC6H6 = gen_s_linear(ref_C6H6, param )
    sC6H12 = gen_s_linear(ref_C6H12, param )
    sCCl4 = gen_s_linear(ref_CCl4, param )

    # residual matrix
    e_C6H6 = I_C6H6 - sC6H6
    e_C6H12 = I_C6H12 - sC6H12
    e_CCl4 = I_CCl4 - sCCl4

    e_C6H6 = np.multiply(wMat_C6H6, e_C6H6)
    e_C6H12 = np.multiply(wMat_C6H12, e_C6H12)
    e_CCl4 = np.multiply(wMat_CCl4, e_CCl4)

    e_CCl4 = clean_mat(e_CCl4)
    e_C6H6 = clean_mat(e_C6H6)
    e_C6H12 = clean_mat(e_C6H12)

    e_C6H6 = np.abs(e_C6H6)
    e_C6H12 = np.abs(e_C6H12)
    e_CCl4 = np.abs(e_CCl4)    

    # savetxt
    #np.savetxt('linear_e_C6H6.txt', e_C6H6, fmt='%5.3f', delimiter='\t')
    #np.savetxt('linear_e_C6H12.txt', e_C6H12, fmt='%5.3f', delimiter='\t')
    #np.savetxt('linear_e_CCl4.txt', e_CCl4, fmt='%5.3f', delimiter='\t')      

    E = np.sum(e_C6H6) + np.sum(e_C6H12) \
       + np.sum(e_CCl4)        

    return E

#*******************************************************************
#*******************************************************************

def residual_quadratic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 )

    param : c1, c2

    '''
    # ------ C6H6 ------
    trueR_C6H6 = gen_intensity_mat (ref_C6H6, 1)  # col 1 has ref. area
    expt_C6H6 = gen_intensity_mat (data_C6H6, 0)
    I_C6H6 = np.divide(expt_C6H6, trueR_C6H6)
    I_C6H6 = clean_mat(I_C6H6)
    # ----------------

    # ------ C6H12 ------
    trueR_C6H12 = gen_intensity_mat (ref_C6H12, 1)
    expt_C6H12 = gen_intensity_mat (data_C6H12, 0)
    I_C6H12 = np.divide(expt_C6H12, trueR_C6H12)
    I_C6H12 = clean_mat(I_C6H12)
    # ----------------

    # ------ CCl4 ------
    trueR_CCl4 = gen_intensity_mat (ref_CCl4, 1)
    expt_CCl4 = gen_intensity_mat (data_CCl4, 0)
    I_CCl4 = np.divide(expt_CCl4, trueR_CCl4)
    I_CCl4 = clean_mat(I_CCl4)
    # ----------------

    # generate the RHS : sensitivity factor
    sC6H6 = gen_s_quadratic(ref_C6H6, param )
    sC6H12 = gen_s_quadratic(ref_C6H12, param )
    sCCl4 = gen_s_quadratic(ref_CCl4, param )

    # residual matrix
    e_C6H6 = I_C6H6 - sC6H6
    e_C6H12 = I_C6H12 - sC6H12
    e_CCl4 = I_CCl4 - sCCl4
    
    e_C6H6 = np.multiply(wMat_C6H6, e_C6H6)
    e_C6H12 = np.multiply(wMat_C6H12, e_C6H12)
    e_CCl4 = np.multiply(wMat_CCl4, e_CCl4)

    e_CCl4 = clean_mat(e_CCl4)
    e_C6H6 = clean_mat(e_C6H6)
    e_C6H12 = clean_mat(e_C6H12)
    
    e_C6H6 = np.abs(e_C6H6)
    e_C6H12 = np.abs(e_C6H12)
    e_CCl4 = np.abs(e_CCl4)    

    # savetxt
    np.savetxt('quadratic_e_C6H6.txt', e_C6H6, fmt='%2.6f', delimiter='\t')
    np.savetxt('quadratic_e_C6H12.txt', e_C6H12, fmt='%2.6f', delimiter='\t')
    np.savetxt('quadratic_e_CCl4.txt', e_CCl4, fmt='%2.6f', delimiter='\t')      

    E = np.sum(np.abs(e_C6H6)) + np.sum(np.abs(e_C6H12)) \
       + np.sum(np.abs(e_CCl4))    

    #E = np.sum(np.square(e_C6H6)) + np.sum(np.square(e_C6H12)) \
    #   + np.sum(np.square(e_CCl4))   

    return E

#*******************************************************************
#*******************************************************************

def residual_cubic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : c1, c2, c3

    '''
    # ------ C6H6 ------
    trueR_C6H6 = gen_intensity_mat (ref_C6H6, 1)  # col 1 has ref. area
    expt_C6H6 = gen_intensity_mat (data_C6H6, 0)
    I_C6H6 = np.divide(expt_C6H6, trueR_C6H6)
    I_C6H6 = clean_mat(I_C6H6)
    # ----------------

    # ------ C6H12 ------
    trueR_C6H12 = gen_intensity_mat (ref_C6H12, 1)
    expt_C6H12 = gen_intensity_mat (data_C6H12, 0)
    I_C6H12 = np.divide(expt_C6H12, trueR_C6H12)
    I_C6H12 = clean_mat(I_C6H12)
    # ----------------

    # ------ CCl4 ------
    trueR_CCl4 = gen_intensity_mat (ref_CCl4, 1)
    expt_CCl4 = gen_intensity_mat (data_CCl4, 0)
    I_CCl4 = np.divide(expt_CCl4, trueR_CCl4)
    I_CCl4 = clean_mat(I_CCl4)
    # ----------------

    # generate the RHS : sensitivity factor
    sC6H6 = gen_s_cubic(ref_C6H6, param )
    sC6H12 = gen_s_cubic(ref_C6H12, param )
    sCCl4 = gen_s_cubic(ref_CCl4, param )

    # residual matrix
    e_C6H6 = I_C6H6 - sC6H6
    e_C6H12 = I_C6H12 - sC6H12
    e_CCl4 = I_CCl4 - sCCl4

    e_C6H6 = np.multiply(wMat_C6H6, e_C6H6)
    e_C6H12 = np.multiply(wMat_C6H12, e_C6H12)
    e_CCl4 = np.multiply(wMat_CCl4, e_CCl4)

    e_CCl4 = clean_mat(e_CCl4)
    e_C6H6 = clean_mat(e_C6H6)
    e_C6H12 = clean_mat(e_C6H12)        

    e_C6H6 = np.abs(e_C6H6)
    e_C6H12 = np.abs(e_C6H12)
    e_CCl4 = np.abs(e_CCl4)    

    # savetxt
    #np.savetxt('cubic_e_C6H6.txt', e_C6H6, fmt='%5.3f', delimiter='\t')
    #np.savetxt('cubic_e_C6H12.txt', e_C6H12, fmt='%5.3f', delimiter='\t')
    #np.savetxt('cubic_e_CCl4.txt', e_CCl4, fmt='%5.3f', delimiter='\t')      

    E = np.sum(np.abs(e_C6H6)) + np.sum(np.abs(e_C6H12)) \
       + np.sum(np.abs(e_CCl4))   

    #E = np.sum(np.square(e_C6H6)) + np.sum(np.square(e_C6H12)) \
    #   + np.sum(np.square(e_CCl4))   

    return E

#*******************************************************************
#*******************************************************************

def residual_quartic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : c1, c2, c3

    '''
    # ------ C6H6 ------
    trueR_C6H6 = gen_intensity_mat (ref_C6H6, 1)  # col 1 has ref. area
    expt_C6H6 = gen_intensity_mat (data_C6H6, 0)
    I_C6H6 = np.divide(expt_C6H6, trueR_C6H6)
    I_C6H6 = clean_mat(I_C6H6)
    # ----------------

    # ------ C6H12 ------
    trueR_C6H12 = gen_intensity_mat (ref_C6H12, 1)  # col 1 has ref. area
    expt_C6H12 = gen_intensity_mat (data_C6H12, 0)
    I_C6H12 = np.divide(expt_C6H12, trueR_C6H12)
    I_C6H12 = clean_mat(I_C6H12)
    # ----------------

    # ------ CCl4 ------
    trueR_CCl4 = gen_intensity_mat (ref_CCl4, 1)  # col 1 has ref. area
    expt_CCl4 = gen_intensity_mat (data_CCl4, 0)
    I_CCl4 = np.divide(expt_CCl4, trueR_CCl4)
    I_CCl4 = clean_mat(I_CCl4)
    # ----------------

    # generate the RHS : sensitivity factor
    sC6H6 = gen_s_quartic(ref_C6H6, param )
    sC6H12 = gen_s_quartic(ref_C6H12, param )
    sCCl4 = gen_s_quartic(ref_CCl4, param )

    # residual matrix
    e_C6H6 = I_C6H6 - sC6H6
    e_C6H12 = I_C6H12 - sC6H12
    e_CCl4 = I_CCl4 - sCCl4

    e_C6H6 = np.multiply(wMat_C6H6, e_C6H6)
    e_C6H12 = np.multiply(wMat_C6H12, e_C6H12)
    e_CCl4 = np.multiply(wMat_CCl4, e_CCl4)
    
    e_CCl4 = clean_mat(e_CCl4)
    e_C6H6 = clean_mat(e_C6H6)
    e_C6H12 = clean_mat(e_C6H12)    

    E = np.sum(np.abs(e_C6H6)) + np.sum(np.abs(e_C6H12)) \
       + np.sum(np.abs(e_CCl4))

    #E = np.sum(np.square(e_C6H6)) + np.sum(np.square(e_C6H12)) \
    #   + np.sum(np.square(e_CCl4))   

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
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0} output = {1}".format( init_k1, \
          (residual_linear(param_init))))

    print("\nOptimization run     \n")
    res = opt.minimize(residual_linear, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optk1 = res.x[0]
    print("\nOptimized result : k1={0} \n".format(round(optk1, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis-scenter)     # generate the correction curve

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
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1} output = {2}".format( init_k1, \
         init_k2, (residual_quadratic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quadratic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    print("\nOptimized result : k1={0}, k2={1} \n".format( round(optk1, 6),
                                                          round(optk2, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis-scenter) \
        + ((optk2/scale2)*(xaxis-scenter)**2)  # generate the\
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
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, output = {3}".format( init_k1, \
         init_k2, init_k3, (residual_cubic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_cubic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    print("\nOptimized result : k1={0}, k2={1}, k3={2} \n".format( round(optk1, 6),
                                                                  round(optk2, 6),
                                                                  round(optk3, 6)))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis-scenter)) \
        + ((optk2/scale2)*(xaxis-scenter)**2) + ((optk3/scale3)*(xaxis-scenter)**3)

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
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, k4={3}, output = {4}".format( init_k1, \
         init_k2, init_k3, init_k4, (residual_quartic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quartic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    optk4 = res.x[3]
    print("\nOptimized result : k1={0}, k2={1}, k3={2}, k4={3} \n".format( 
        round(optk1, 6), round(optk2, 6), round(optk3, 6) ,round(optk4, 6) ))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis-scenter))\
        + ((optk2/scale2)*(xaxis-scenter)**2) + ((optk3/scale3)*(xaxis-scenter)**3) \
            + ((optk4/scale4)*(xaxis-scenter)**4)

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


def plot_curves(residual_array="None"):
    '''
    If array containing residuals is not provided
    then the plot of residuals vs number of variables
    will not be made
    '''
    
    '''
    option = 1 : plot
           = 0 : do not plot

    '''
    option=1
    if option == 1:
        # Load the saved correction curves for  plotting
        # outputs from last run will be loaded
        correction_line = np.loadtxt("./correction_linear.txt", skiprows=1)
        correction_quad = np.loadtxt("./correction_quadratic.txt", skiprows=1)
        correction_cubic = np.loadtxt("./correction_cubic.txt", skiprows=1)
        correction_quartic = np.loadtxt("./correction_quartic.txt", skiprows=1)

        #********************************************************************

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

        xmin=np.amin(xaxis-10)
        xmax=np.amax(xaxis+10)


        plt.xlim((xmax, xmin)) #  change this if the xlimit is not correct
        ax0.set_ylim([0, 1.5]) # change this if the ylimit is not enough

        ax0.minorticks_on()
        ax0.tick_params(which='minor', right='on')
        ax0.tick_params(axis='y', labelleft='on', labelright='on')
        plt.text(0.05, 0.0095, txt, fontsize=6, color="dimgrey",
                 transform=plt.gcf().transFigure)


        # Add reference data to the plot
        x=xaxis
        yquadr=1+(-0.92351714/1e4)*x + (-0.25494267/1e7)*x**2
        plt.plot(xaxis,  yquadr, 'k-', linewidth=2.65, label='REF-Quadratic')
        
        plt.legend(loc='upper left', fontsize=16)
        
        # Add markers 
        # markers showing the bands positions whose data is used for fit 
        plt.plot(ref_CCl4[:,0], dummyCCl4, 'mo' )
        plt.plot(ref_C6H12[:,0], dummyC6H12, 'cv' )
        plt.plot(ref_C6H6[:,0], dummyC6H6, 'gD' )

        # *********************
        if type(residual_array) != str:
            if isinstance(residual_array, (list, np.ndarray)):
                # -----------------------------------------------------
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
                print('\tWrong type of parameter : residual_array. \
                      Quitting plotting.')
        else:
            print('\tResidual array not provided. plot of residuals not made!')


        #  For saving the plot
        #plt.savefig('fit_output.png', dpi=120)
#********************************************************************
# -----------------------------------------------------
#  Dummy value for plot (vs frequencies)
#   Shows which band were analyzed in the fitting
val=0.125
dummyCCl4 = np.full(len(ref_CCl4), val)
dummyC6H12 = np.full(len(ref_C6H12), val)
dummyC6H6 = np.full(len(ref_C6H6), val)
# -----------------------------------------------------
