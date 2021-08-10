#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Module describing the weighted non-linear optimization scheme used to
determine the wavelength sensitivity of the spectrometer using a  polynomial
as a model function"""
import os
import numpy as np
import math

import scipy.optimize as opt
import logging
from datetime import datetime
import matplotlib.pyplot as plt

import compute_series_perp
import boltzmann_popln as bp


# ------------------------------------------------------
def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))
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

dataH2 = np.loadtxt("BA_H2_perp") 
dataHD = np.loadtxt("BA_HD_perp")
dataD2 = np.loadtxt("BA_D2_perp")
xaxis = np.loadtxt("Ramanshift_axis")
# ------------------------------------------------------
# PARALLEL POLARIZATION

# set indices for OJ,QJ and SJ for HD and D2
# these are required for computing spectra at a temperature T

OJ_H2 = 3
QJ_H2 = 4

OJ_HD = 3
QJ_HD = 3
SJ_HD = 2

OJ_D2 = 4
QJ_D2 = 6
SJ_D2 = 3

# ----------------------------------------

print('Dimension of input data')
print('\t', dataH2.shape)
print('\t', dataHD.shape)
print('\t', dataD2.shape)
# ------------------------------------------------------

# Fixed temperature for analysis

T_fixed = 298   # Kelvin

# ------------------------------------------------------

scenter = 3316.3  # center of the spectra
# used to scale the xaxis

scaled_xaxis = ( xaxis - scenter ) 
magn = orderOfMagnitude(np.amax(scaled_xaxis))

# for example,
# magn = 2 for 100 

# scaling_factor=(10**magn)
# scaling factor below is thus deduced from the magnitude

scale1 = (10**magn)
scale2 = (10**magn)**2
scale3 = (10**magn)**3
scale4 = (10**magn)**4
scale5= (10**magn)**5

# or use fixed constants as scale

#scale1 = 1e3
#scale2 = 1e6
#scale3 = 1e9
#scale4 = 1e12
#scale5= 1e13
# ----------------------------------------

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


# Set logging ------------------------------------------
fileh = logging.FileHandler('./logfile_Tfixed_perpendicular', 'w+')
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


print('\t**********************************************************')

print('\t ')
print('\t This module is for generating the wavenumber-dependent')
print('\t intensity correction curve termed as C2 from ')
print('\t  experimental Raman intensities. ')

print('\n\t This modeule requires edit on line 21 to 92 to ')
print('\n\t  load data and set parameters for the analysis.')
print('\n\t  Temperature is fixed in this analysis. See line 51.')
print('\t**********************************************************')
print('\n\t\t Checking imported data and set params')

data_error=0

if isinstance(dataH2, np.ndarray):
    print("\t\t ", "dataH2 found, OK")
else:
    print("\t\t ", "dataH2 not found.")

if isinstance(dataHD, np.ndarray):
    print("\t\t ", "dataHD found, OK")
else:
    print("\t\t ", "dataHD not found.")

if isinstance(dataD2, np.ndarray):
    print("\t\t ", "dataD2 found, OK")
else:
    print("\t\t ", "dataD2 not found.")

if isinstance(xaxis, np.ndarray):
    print("\t\t ", "xaxis found, OK")
else:
    print("\t\t ", "xaxis not found.")

print('\n\t\t  Analysis parameters:')

print("\t\t Temperature (will be fixed) :", T_fixed)
print("\t\t scaling factors (for c1 to c3) :", scale1, scale2, scale3)
print("\t\t Norm (defn of residual) : ", norm)


print('\t**********************************************************')
print('\n\t REQUIRED DATA')
print('\t\t\t Ramanshift = vector, the x-axis in relative wavenumbers')
print('\t\t\t band area and error = 2D (2 columns), for H2, HD and D2')
print('\n\t\t\t J_max = scalar, for H2, HD and D2 (to compute reference spectra)')


print('\t**********************************************************')

print('\n\t\t\t  Example:')

print('\t\t\t  run_fit_linear_TF(  1.04586 )')
print('\t\t\t  run_fit_quadratic_TF(  -1, -0.242 ) ')


print('\t**********************************************************')

#############################################################################
#############################################################################

# write analysis data to log
log.info('\n\t Input data')
if data_error==0 :
    log.info('\n\t OK')
else:
    log.info('\n\t Some data is missing. Exiting...')
    sys.exit()
    
log.info('\n\t Input data shape:')
log.info('\t\t H2:\t %s\n', dataH2.shape )
log.info('\t\t HD:\t %s\n', dataHD.shape )
log.info('\t\t D2:\t %s\n', dataD2.shape )

log.info('\t\t Temperature (fixed):\t %s\n', T_fixed )

log.info('\n\t Parameters:')
log.info('\t\t Norm:\t %s', norm)

#############################################################################

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
        

    #return factor * inverse_square(error_mat)
    return  np.abs(error_mat)

#------------------------------------------------

def inverse_square(array):
    """return the inverse square of array, for all elements"""
    return 1/(array**2)

#------------------------------------------------
  
def gen_s_linear(computed_data, param ):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as line"""
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter
            
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
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter
            
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
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter
            
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
            v1 = computed_data[i, 1] - scenter
            v2 = computed_data[j, 1] - scenter

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

#------------------------------------------------ 
#------------------------------------------------ 


#*******************************************************************
#  GENERATE WEIGHT MATRICES 

# following gives example to generate weights

wMat_H2 = gen_weight(dataH2, 0.2)
wMat_HD = gen_weight(dataHD, 0.2)
wMat_D2 = gen_weight(dataD2, 0.2)

#print(wMat_H2 )

wMat_H2 = np.divide(wMat_H2, 300)
wMat_HD = np.divide(wMat_HD, 300)
wMat_D2 = np.divide(wMat_D2, 300)

wMat_H2=scale_opp_diagonal (wMat_H2, 500)
wMat_HD=scale_opp_diagonal (wMat_HD, 500)
wMat_D2=scale_opp_diagonal (wMat_D2, 500)

# for testing purpose the weight is set to unity below for 
#   the elements of the difference matrices for the
#   three gases

wMat_H2=1
wMat_HD=1
wMat_D2=1
#*******************************************************************

#*******************************************************************
# Define the residual function
#*******************************************************************

def residual_linear_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x ) 
    
    param : T, c1
    
    '''

    # temperature is read from the variable `T_fixed` defined earlier

    sosD2 = bp.sumofstate_D2(T_fixed )
    sosHD = bp.sumofstate_HD(T_fixed )
    sosH2 = bp.sumofstate_H2(T_fixed )

    computed_D2 = compute_series_perp.spectra_D2(T_fixed , OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_perp.spectra_HD(T_fixed , OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)
    computed_H2 = compute_series_perp.spectra_H2_c(T_fixed , OJ_H2,
                                                   QJ_H2, sosH2)
    

    # remove row for Q(J=0) --
    i, = np.where(computed_D2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_D2 = np.delete(computed_D2, (row_index), axis=0)
    
    i, = np.where(computed_HD[:,0] == 0.0)
    row_index = np.amin(i)
    computed_HD = np.delete(computed_HD, (row_index), axis=0)
    
    i, = np.where(computed_H2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_H2 = np.delete(computed_H2, (row_index), axis=0)    
    # ------------------------
    
    # ------ D2 ------ 
    trueR_D2=gen_intensity_mat (computed_D2, 2)
    expt_D2=gen_intensity_mat (dataD2, 0)   
    I_D2=np.divide(expt_D2,trueR_D2 )
    I_D2=clean_mat(I_D2)
    # ----------------  
    
    # ------ HD ------ 
    trueR_HD=gen_intensity_mat (computed_HD, 2)
    expt_HD=gen_intensity_mat (dataHD, 0)
    I_HD=np.divide(expt_HD,trueR_HD )
    I_HD=clean_mat(I_HD)
    # ----------------  
    
    # ------ H2 ------ 
    trueR_H2=gen_intensity_mat (computed_H2, 2)
    expt_H2=gen_intensity_mat (dataH2, 0)
    I_H2=np.divide(expt_H2,trueR_H2 )    
    I_H2=clean_mat(I_H2)
    # ---------------- 
    
    # generate the RHS : sensitivity factor
    sD2=gen_s_linear(computed_D2, param )
    sHD=gen_s_linear(computed_HD, param )
    sH2=gen_s_linear(computed_H2, param )
    
    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD
    eH2 = I_H2 - sH2

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)
    eH2 = np.multiply(wMat_H2, eH2)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)
    eH2 = clean_mat(eH2)
    
    #  choosing norm 
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) 
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD))  
        
    return(E)

#*******************************************************************
#*******************************************************************

def residual_quadratic_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 ) 
    
    param : T, c1, c2
    
    '''

    # temperature is read from the variable `T_fixed` defined earlier

    sosD2 = bp.sumofstate_D2(T_fixed )
    sosHD = bp.sumofstate_HD(T_fixed )
    sosH2 = bp.sumofstate_H2(T_fixed )

    computed_D2 = compute_series_perp.spectra_D2(T_fixed , OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_perp.spectra_HD(T_fixed , OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)
    computed_H2 = compute_series_perp.spectra_H2_c(T_fixed , OJ_H2,
                                                   QJ_H2, sosH2)
    
    

    # remove row for Q(J=0) --
    i, = np.where(computed_D2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_D2 = np.delete(computed_D2, (row_index), axis=0)
    
    i, = np.where(computed_HD[:,0] == 0.0)
    row_index = np.amin(i)
    computed_HD = np.delete(computed_HD, (row_index), axis=0)
    
    i, = np.where(computed_H2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_H2 = np.delete(computed_H2, (row_index), axis=0)    
    # ------------------------    
    
    
    # ------ D2 ------ 
    trueR_D2=gen_intensity_mat (computed_D2, 2)
    expt_D2=gen_intensity_mat (dataD2, 0)   
    I_D2=np.divide(expt_D2,trueR_D2 )
    I_D2=clean_mat(I_D2)
    # ----------------  
    
    # ------ HD ------ 
    trueR_HD=gen_intensity_mat (computed_HD, 2)
    expt_HD=gen_intensity_mat (dataHD, 0)
    I_HD=np.divide(expt_HD,trueR_HD )
    I_HD=clean_mat(I_HD)
    # ----------------  
    
    # ------ H2 ------ 
    trueR_H2=gen_intensity_mat (computed_H2, 2)
    expt_H2=gen_intensity_mat (dataH2, 0)
    I_H2=np.divide(expt_H2,trueR_H2 )    
    I_H2=clean_mat(I_H2)
    # ---------------- 
    
    # generate the RHS : sensitivity factor
    sD2=gen_s_quadratic(computed_D2, param)
    sHD=gen_s_quadratic(computed_HD, param)
    sH2=gen_s_quadratic(computed_H2, param)
    
    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD
    eH2 = I_H2 - sH2

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)
    eH2 = np.multiply(wMat_H2, eH2)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)
    eH2 = clean_mat(eH2)
    
    
    #  choosing norm 
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) 
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD))   
        
    return(E)

#*******************************************************************
#*******************************************************************

def residual_cubic_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 ) 
    
    param : T, c1, c2, c3
    
    '''
    # temperature is read from the variable `T_fixed` defined earlier    

    sosD2 = bp.sumofstate_D2(T_fixed )
    sosHD = bp.sumofstate_HD(T_fixed )
    sosH2 = bp.sumofstate_H2(T_fixed )

    computed_D2 = compute_series_perp.spectra_D2(T_fixed , OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_perp.spectra_HD(T_fixed , OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)
    computed_H2 = compute_series_perp.spectra_H2_c(T_fixed , OJ_H2,
                                                   QJ_H2, sosH2)
    
    

    # remove row for Q(J=0) --
    i, = np.where(computed_D2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_D2 = np.delete(computed_D2, (row_index), axis=0)
    
    i, = np.where(computed_HD[:,0] == 0.0)
    row_index = np.amin(i)
    computed_HD = np.delete(computed_HD, (row_index), axis=0)
    
    i, = np.where(computed_H2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_H2 = np.delete(computed_H2, (row_index), axis=0)    
    # ------------------------    
    
    # ------ D2 ------ 
    trueR_D2=gen_intensity_mat (computed_D2, 2)
    expt_D2=gen_intensity_mat (dataD2, 0)   
    I_D2=np.divide(expt_D2,trueR_D2 )
    I_D2=clean_mat(I_D2)
    # ----------------  
    
    # ------ HD ------ 
    trueR_HD=gen_intensity_mat (computed_HD, 2)
    expt_HD=gen_intensity_mat (dataHD, 0)
    I_HD=np.divide(expt_HD,trueR_HD )
    I_HD=clean_mat(I_HD)
    # ----------------  
    
    # ------ H2 ------ 
    trueR_H2=gen_intensity_mat (computed_H2, 2)
    expt_H2=gen_intensity_mat (dataH2, 0)
    I_H2=np.divide(expt_H2,trueR_H2 )    
    I_H2=clean_mat(I_H2)
    # ---------------- 
    
    # generate the RHS : sensitivity factor
    sD2=gen_s_cubic(computed_D2, param)
    sHD=gen_s_cubic(computed_HD, param)
    sH2=gen_s_cubic(computed_H2, param)
    
    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD
    eH2 = I_H2 - sH2

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)
    eH2 = np.multiply(wMat_H2, eH2)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)
    eH2 = clean_mat(eH2)

 
    #  choosing norm 
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) 
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD))  
        
    return(E)

#*******************************************************************
#*******************************************************************

def residual_quartic_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 ) 
    
    param : T, c1, c2, c3
    
    '''
    # temperature is read from the variable `T_fixed` defined earlier    

    sosD2 = bp.sumofstate_D2( T_fixed )
    sosHD = bp.sumofstate_HD( T_fixed )
    sosH2 = bp.sumofstate_H2( T_fixed )

    computed_D2 = compute_series_perp.spectra_D2(T_fixed , OJ_D2, QJ_D2,
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_perp.spectra_HD(T_fixed , OJ_HD, QJ_HD,
                                                 SJ_HD, sosHD)
    computed_H2 = compute_series_perp.spectra_H2_c(T_fixed , OJ_H2,
                                                   QJ_H2, sosH2)
    
    

    # remove row for Q(J=0) --
    i, = np.where(computed_D2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_D2 = np.delete(computed_D2, (row_index), axis=0)
    
    i, = np.where(computed_HD[:,0] == 0.0)
    row_index = np.amin(i)
    computed_HD = np.delete(computed_HD, (row_index), axis=0)
    
    i, = np.where(computed_H2[:,0] == 0.0)
    row_index = np.amin(i)
    computed_H2 = np.delete(computed_H2, (row_index), axis=0)    
    # ------------------------    
    
    
    # ------ D2 ------ 
    trueR_D2=gen_intensity_mat (computed_D2, 2)
    expt_D2=gen_intensity_mat (dataD2, 0)   
    I_D2=np.divide(expt_D2,trueR_D2 )
    I_D2=clean_mat(I_D2)
    # ----------------  
    
    # ------ HD ------ 
    trueR_HD=gen_intensity_mat (computed_HD, 2)
    expt_HD=gen_intensity_mat (dataHD, 0)
    I_HD=np.divide(expt_HD,trueR_HD )
    I_HD=clean_mat(I_HD)
    # ----------------  
    
    # ------ H2 ------ 
    trueR_H2=gen_intensity_mat (computed_H2, 2)
    expt_H2=gen_intensity_mat (dataH2, 0)
    I_H2=np.divide(expt_H2,trueR_H2 )    
    I_H2=clean_mat(I_H2)
    # ---------------- 
    
    # generate the RHS : sensitivity factor
    sD2=gen_s_quartic(computed_D2, param)
    sHD=gen_s_quartic(computed_HD, param)
    sH2=gen_s_quartic(computed_H2, param)
    
    # residual matrix
    eD2 = ( np.multiply( wMat_D2, I_D2 )) - sD2 
    eHD = ( np.multiply( wMat_HD, I_HD )) - sHD 
    eH2 = ( np.multiply( wMat_H2, I_H2 )) - sH2 
    
    # residual matrix
    eD2 = I_D2 - sD2
    eHD = I_HD - sHD
    eH2 = I_H2 - sH2

    eD2 = np.multiply(wMat_D2, eD2)
    eHD = np.multiply(wMat_HD, eHD)
    eH2 = np.multiply(wMat_H2, eH2)

    eD2 = clean_mat(eD2)
    eHD = clean_mat(eHD)
    eH2 = clean_mat(eH2)
 
    #  choosing norm 
    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) 
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) 
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD))  
        
    return(E)

#*******************************************************************    

#***************************************************************
#***************************************************************
# Fit functions
#***************************************************************
#***************************************************************

def run_fit_linear_TF ( init_k1 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1  ])
    print("**********************************************************")
    print("\t\t -- Linear fit -- ")
    print("\t\tNorm (defn of residual): ", norm)       
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0} output = {1}".format( init_k1, \
          (residual_linear_TF(param_init))))

    print("\nOptimization run     \n")
    res = opt.minimize(residual_linear_TF, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optk1 = res.x[0]
    print("\nOptimized result : k1={0} \n".format(round(optk1, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis-scenter)     # generate the correction curve

    np.savetxt("correction_linear_TF.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_linear_TF', comments='')

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : c1 = %4.8f\n', init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f\n', optk1 )
    log.info(' *******************************************')
    # --------------------    
    return res.fun
    # --------------------
#***************************************************************
    
def run_fit_quadratic_TF ( init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1, init_k2 : Intial guess

    param_init = np.array([   init_k1 , init_k2  ])
    print("**********************************************************")
    print("\t\t -- Quadratic fit -- ")
    print("\t\tNorm (defn of residual): ", norm)       
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1} output = {2}".format( init_k1, \
         init_k2, (residual_quadratic_TF(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quadratic_TF, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)

    optk1 = res.x[0]
    optk2 = res.x[1]
    print("\nOptimized result : k1={0}, k2={1} \n".format( round(optk1, 6), round(optk2, 6) ))

    correction_curve= 1+(optk1/scale1)*(xaxis-scenter) \
        + ((optk2/scale2)*(xaxis-scenter)**2)  # generate the\
                                                                               #correction curve

    np.savetxt("correction_quadratic_TF.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quadratic_TF', comments='')

    print("**********************************************************")
    return res.fun
    # --------------------
#***************************************************************    
    
    
def run_fit_cubic_TF ( init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
    print("\t\t -- Cubic fit -- ")
    print("\t\tNorm (defn of residual): ", norm)       
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, output = {3}".format( init_k1, \
         init_k2, init_k3, (residual_cubic_TF(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_cubic_TF, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    
    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    print("\nOptimized result : k1={0}, k2={1}, k3={2} \n".format( round(optk1, 6), round(optk2, 6), round(optk3, 6)))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis-scenter)) \
        + ((optk2/scale2)*(xaxis-scenter)**2) + ((optk3/scale3)*(xaxis-scenter)**3) 
    
    np.savetxt("correction_cubic_TF.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_cubic_TF', comments='')

    print("**********************************************************")
    return res.fun
    # --------------------
#***************************************************************     

    
def run_fit_quartic_TF ( init_k1, init_k2, init_k3, init_k4 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3 , init_k4  ])
    print("**********************************************************")
    print("\t\t -- Quartic fit -- ")
    print("\t\tNorm (defn of residual): ", norm)       
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0}, k2={1}, k3={2}, k4={3}, output = {4}".format( init_k1, \
         init_k2, init_k3, init_k4, (residual_quartic_TF(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quartic_TF, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    
    optk1 = res.x[0]
    optk2 = res.x[1]
    optk3 = res.x[2]
    optk4 = res.x[3]
    print("\nOptimized result : k1={0}, k2={1}, k3={2}, k4={3} \n".format( round(optk1, 6), round(optk2, 6), round(optk3, 6) ,round(optk4, 6) ))

    # generate the correction curve
    correction_curve = (1+(optk1/scale1)*(xaxis-scenter))\
        + ((optk2/scale2)*(xaxis-scenter)**2) + ((optk3/scale3)*(xaxis-scenter)**3) \
            + ((optk4/scale4)*(xaxis-scenter)**4) 
    
    np.savetxt("correction_quartic_TF.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quartic_TF', comments='')

    print("**********************************************************")
    return res.fun
    # --------------------

#***************************************************************   


wMat_D2 = 1
wMat_HD = 1
wMat_H2 = 1

#***************************************************************   

run=1
plot_option=1

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
    resd_5 = 0

    resd_1 = run_fit_linear_TF( -1.04586)
    resd_2 = run_fit_quadratic_TF( -0.955, -0.052)
    resd_3 = run_fit_cubic_TF( -0.954, -1.28, +0.001)
    resd_4 = run_fit_quartic_TF( -0.5566, -0.158, 0.05, +0.0012)
  
    out = np.array([resd_1, resd_2, resd_3, resd_4])
    return out

#***************************************************************   
#***************************************************************  

def plot_curves(residual_array="None"):
    '''
    If array containing residuals is not provided
    then the plot of residuals vs number of variables
    will not be made

    '''
    # Load the saved correction curves for  plotting
    # outputs from last run will be loaded
    correction_line = np.loadtxt("./correction_linear_TF.txt", skiprows=1)
    correction_quad = np.loadtxt("./correction_quadratic_TF.txt", skiprows=1)
    correction_cubic = np.loadtxt("./correction_cubic_TF.txt", skiprows=1)
    correction_quartic = np.loadtxt("./correction_quartic_TF.txt", skiprows=1)

    # ---------------------------------------------------------------------

    # Plotting the data

    txt = ("*Generated from 'wavelength_sensitivity.py' on the\
          \nGitHub Repository: IntensityCalbr ")

    # FIGURE 0 INITIALIZED

    plt.figure(num=0, figsize=(8.2, 5.7), dpi=325)
    #figure(num=0, figsize=(8, 6), dpi=300)
    ax0 = plt.axes()
    plt.title('Fitting result', fontsize=22)

    plt.plot(xaxis, correction_line, 'r', linewidth=3, label='line_fit')
    plt.plot(xaxis, correction_quad, 'g', linewidth=3.2, label='quad_fit')
    plt.plot(xaxis, correction_cubic, 'b--', linewidth=3.65, label='cubic_fit')
    plt.plot(xaxis, correction_quartic, 'k--', linewidth=2.65, label='quartic_fit')


    plt.xlabel('Wavenumber / $cm^{-1}$', fontsize=20)
    plt.ylabel('Relative sensitivity', fontsize=20)
    plt.grid(True, which='both')  # ax.grid(True, which='both')

    # change following as needed
    ax0.tick_params(axis='both', labelsize=20)

    xmin = 2500
    xmax = np.amax(xaxis + 10)

    plt.xlim((xmax, xmin)) #  setting xaxis range
    ax0.set_ylim([0, 1.6]) #  change ylimit

    ax0.minorticks_on()
    ax0.tick_params(which='minor', right='on')
    ax0.tick_params(axis='y', labelleft='on', labelright='on')
    plt.text(0.05, 0.0095, txt, fontsize=6, color="dimgrey",
             transform=plt.gcf().transFigure)
    plt.legend(loc='upper left', fontsize=16)

    # markers showing the bands positions whose data is used for fit
    #plt.plot(computed_D2[:,1], dummyD2, 'mo' )
    #plt.plot(computed_HD[:,1], dummyHD, 'cv' )
    #plt.plot(computed_H2[:,1], dummyH2, 'gD' )

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

#********************************************************************
#resd_array=run_all_fit()


#plot_curves( resd_array )