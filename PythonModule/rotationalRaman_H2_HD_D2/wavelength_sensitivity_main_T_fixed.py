#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Module describing the weighted non-linear optimization scheme used to
determine the wavelength sensitivity of the spectrometer using a  polynomial
as a model function"""

import numpy as np
import math
import compute_spectra
import scipy.optimize as opt
import logging
from datetime import datetime

import cProfile

import time     
import utils                                           
# ----------------------------------------
def timeme(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))

        print(endTime - startTime,'ms')
        return result

    return wrapper

# ----------------------------------------

# Set logging ------------------------------------------
fileh = logging.FileHandler('logfile', 'w+')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
fileh.setFormatter(formatter)

log = logging.getLogger()  # root logger
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(fileh)      # set the new handler
# ------------------------------------------------------

# Logging starts here
log.debug( datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
logger= logging.getLogger( __file__ )
log.info(logger)
logging.getLogger().setLevel(logging.INFO)
log.warning('\n',)
log.error("------------ Run log ------------\n")
# ------------------------------------------------------


# LOAD EXPERIMENTAL BAND AREA DATA 

dataH2 = np.loadtxt("./BA_H2_1.txt")
dataHD = np.loadtxt("./BA_HD_1.txt")
dataD2 = np.loadtxt("./BA_D2_1.txt")


dataO2 = np.loadtxt("./DataO2_o1s1.txt")
dataO2_p = np.loadtxt("./DataO2_pR.txt")

xaxis  = np.loadtxt("./Wavenumber_axis_pa.txt")


print(dataH2.shape)
print(dataHD.shape)
print(dataD2.shape)


# Constants ------------------------------
# these are used for scaling the coefs
scale1 = 1e4
scale2 = 1e7
scale3 = 1e9
scale4 = 1e12



# these are used for scaling the weights for O2 is needed
# edit as needed
scale_O2_S1O1 = 2.0
scale_O2_pureRotn= 0.1
# ----------------------------------------

# ----------------------------------------


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

#@utils.MeasureTime
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
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
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
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
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
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
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
def gen_s_quartic(computed_data, param, scale1):
    """Generate sensitivity matrix for wavelength dependent sensitivity
    modeled as quartic polynomial"""    
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1=computed_data[i,1]
            v2=computed_data[j,1]

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

computed_D2=compute_spectra.spectra_D2( 298, 7, 7)
computed_HD=compute_spectra.spectra_HD( 298, 5, 5)
computed_H2=compute_spectra.spectra_H2( 298, 5, 5)
    

trueR_D2=gen_intensity_mat (computed_D2, 2)
expt_D2=gen_intensity_mat (dataD2, 0)   
  
trueR_HD=gen_intensity_mat (computed_HD, 2)
expt_HD=gen_intensity_mat (dataHD, 0)

trueR_H2=gen_intensity_mat (computed_H2, 2)
expt_H2=gen_intensity_mat (dataH2, 0)

I_D2=np.divide(expt_D2,trueR_D2 )
I_HD=np.divide(expt_HD,trueR_HD )
I_H2=np.divide(expt_H2,trueR_H2 )    
print(I_D2)
    
I_D2=clean_mat(I_D2)
I_HD=clean_mat(I_HD)
I_H2=clean_mat(I_H2)
print(I_D2)


errH2_output=gen_weight(dataH2, 0.1)
errHD_output=gen_weight(dataHD, 0.2)
errD2_output=gen_weight(dataD2, 0.2)

errH2_output=scale_opp_diagonal (errH2_output, 400)
errHD_output=scale_opp_diagonal (errHD_output, 400)
errD2_output=scale_opp_diagonal (errD2_output, 400)

sD2=gen_s_linear(computed_D2, param_linear)
sHD=gen_s_linear(computed_HD, param_linear)
sH2=gen_s_linear(computed_H2, param_linear)

eD2 = ( np.multiply(errD2_output, I_D2 ) - sD2 ) 
eHD = ( np.multiply(errHD_output, I_HD ) - sHD ) 
eH2 = ( np.multiply(errH2_output, I_H2 ) - sH2 ) 

eD2=clean_mat(eD2)
eHD=clean_mat(eHD)
eH2=clean_mat(eH2)

#E=np.sum(np.square(eD2)) + np.sum(np.square(eHD)) + np.sum(np.square(eH2))
E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) + np.sum(np.abs(eH2))
print(E )


#*******************************************************************
#  GENERATE WEIGHT MATRICES 

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
    TK=298.6
    #param_init = np.array([ init_k1  ])
    
    computed_D2=compute_spectra.spectra_D2( TK, 7, 7)
    computed_HD=compute_spectra.spectra_HD( TK, 5, 5)
    computed_H2=compute_spectra.spectra_H2( TK, 5, 5)
    
    
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
    eD2 = ( np.multiply( wMat_D2, I_D2 )) - sD2 
    eHD = ( np.multiply( wMat_HD, I_HD )) - sHD 
    eH2 = ( np.multiply( wMat_H2, I_H2 )) - sH2 
    
    eD2=clean_mat(eD2)
    eHD=clean_mat(eHD)
    eH2=clean_mat(eH2)
    
    
    # oxygen----------------------------
	# - O2 high frequency -
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + param[0]/scale1 * dataO2[:, 3] )/ (1.0 +\
             param[0]/scale1 * dataO2[:, 4] )
    resd_O2 = ( dataO2[:, 5] * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------


    # - O2 pure rotation -
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + param[0]/scale1 * dataO2_p[:, 3] )/ (1.0 +\
             param[0]/scale1 * dataO2_p[:, 4] )
    resd_O2p = (dataO2_p[:, 5] * scale_O2_pureRotn ) * ((ratio_O2p - RHS_O2p)**2)
	# ------
    
    
    E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
        np.sum(np.abs(eH2)) +  np.sum(resd_O2)  + np.sum(resd_O2p)
        
    return(E)

#*******************************************************************
#*******************************************************************

def residual_quadratic_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 ) 
    
    param : T, c1, c2
    
    '''
    TK=298.6
    #param 

    computed_D2=compute_spectra.spectra_D2( TK, 7, 7)
    computed_HD=compute_spectra.spectra_HD( TK, 5, 5)
    computed_H2=compute_spectra.spectra_H2( TK, 5, 5)
    
    
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
    eD2 = ( np.multiply( wMat_D2, I_D2 )) - sD2 
    eHD = ( np.multiply( wMat_HD, I_HD )) - sHD 
    eH2 = ( np.multiply( wMat_H2, I_H2 )) - sH2 
    
    eD2=clean_mat(eD2)
    eHD=clean_mat(eHD)
    eH2=clean_mat(eH2)
    
    # oxygen----------------------------
    c1=param[0]
    c2=param[1]


	# - O2 high frequency -
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + c1/scale1 * dataO2[:, 3] + c2/scale2 * (dataO2[:, 3]**2))/ (1.0 +\
             c1/scale1 * dataO2[:, 4] + c2/scale2 * (dataO2[:, 4]**2))

    resd_O2 = (dataO2[:, 5] * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------

    # - O2 pure rotation -
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + c1/scale1 * dataO2_p[:, 3] + c2/scale2 * (dataO2_p[:, 3]**2))/ (1.0 +\
             c1/scale1 * dataO2_p[:, 4] + c2/scale2 * (dataO2_p[:, 4]**2))


    resd_O2p = (dataO2_p[:, 5]* scale_O2_pureRotn ) * ((ratio_O2p - RHS_O2p)**2)
	# ------
    
    E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
        np.sum(np.abs(eH2)) +  np.sum(resd_O2)  + np.sum(resd_O2p)
        
    return(E)

#*******************************************************************
#*******************************************************************

def residual_cubic_TF(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 ) 
    
    param : T, c1, c2, c3
    
    '''
    TK = 298.6

    computed_D2=compute_spectra.spectra_D2( TK, 7, 7)
    computed_HD=compute_spectra.spectra_HD( TK, 5, 5)
    computed_H2=compute_spectra.spectra_H2( TK, 5, 5)
    
    
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
    eD2 = ( np.multiply( wMat_D2, I_D2 )) - sD2 
    eHD = ( np.multiply( wMat_HD, I_HD )) - sHD 
    eH2 = ( np.multiply( wMat_H2, I_H2 )) - sH2 
    
    eD2=clean_mat(eD2)
    eHD=clean_mat(eHD)
    eH2=clean_mat(eH2)

    # oxygen----------------------------
    c1=param[0]
    c2=param[1]
    c3=param[2]


	# - O2 high frequency -
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + c1/scale1 * dataO2[:, 3] + c2/scale2 * (dataO2[:, 3]**2)+\
              c3/scale3 * (dataO2[:, 3]**3))/ ( 1.0 + c1/scale1 * dataO2[:, 4] +\
                          c2/scale2 *(dataO2[:, 4]**2)+ c3/scale3 *( dataO2[:, 4]**3))

    resd_O2 =( dataO2[:, 5]  * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------

    # - O2 pure rotation -
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + c1/scale1 * dataO2_p[:, 3] + c2/scale2 * (dataO2_p[:, 3]**2)+\
               c3/scale3 * (dataO2_p[:, 3]**3))/ ( 1.0 + c1/scale1 * dataO2_p[:, 4] +\
                           c2/scale2 * (dataO2_p[:, 4]**2)+ c3/scale3 * ( dataO2_p[:, 4]**3))

    resd_O2p = (dataO2_p[:, 5] * scale_O2_pureRotn  ) * ((ratio_O2p - RHS_O2p)**2)
	# ------    
    
    E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
        np.sum(np.abs(eH2)) +  np.sum(resd_O2)  + np.sum(resd_O2p)
        
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
    #print("Testing the residual function with data")
    print("Initial coef :  k1={0} output = {1}".format( init_k1, \
          (residual_linear_TF(param_init))))

    print("\nOptimization run     \n")
    res = opt.minimize(residual_linear_TF, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optk1 = res.x[0]
    print("\nOptimized result : k1={0} \n".format(round(optk1, 6) ))

    correction_curve_line= 1+(optk1/scale1)*xaxis     # generate the correction curve

    np.savetxt("correction_linearv3.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_linearv3', comments='')

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : c1 = %4.8f\n', init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : c1 = %4.8f\n', optk1 )
    log.info(' *******************************************')
    # --------------------    

#***************************************************************
    
def run_fit_quadratic_TF ( init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1, init_k2 : Intial guess

    param_init = np.array([   init_k1 , init_k2  ])
    print("**********************************************************")
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

    correction_curve_line= 1+(optk1/scale1)*xaxis + ((optk2/scale2)*xaxis**2)  # generate the\
                                                                               #correction curve

    np.savetxt("correction_quadraticv3.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_quadraticv3', comments='')

    print("**********************************************************")

#***************************************************************    
    
    
def run_fit_cubic_TF ( init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
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
    correction_curve_line = (1+(optk1/scale1)*xaxis) + ((optk2/scale2)*xaxis**2) + ((optk3/scale3)*xaxis**3) 
    
    np.savetxt("correction_cubicv3.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_cubicv3', comments='')

    print("**********************************************************")

#***************************************************************     

run_fit_linear_TF(  1.04586 )

run_fit_quadratic_TF(  -0.931, -0.242 )
run_fit_cubic_TF(  -0.931, -0.242 , -0.000001 )
        
#run_fit_linear_T_fixed (1.045)

print(param_linear)
#cProfile.run('residual_linear(param_linear)')