#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Module describing the weighted non-linear optimization scheme used to
determine the wavelength sensitivity of the spectrometer using a  polynomial
as a model function"""

import numpy as np
import math
import compute_spectra
import cProfile
# ----------------------------------------

# LOAD EXPERIMENTAL BAND AREA DATA 

dataH2 = np.loadtxt("./BA_H2_1.txt")
dataHD = np.loadtxt("./BA_HD_1.txt")
dataD2 = np.loadtxt("./BA_D2_1.txt")


print(dataH2.shape)
print(dataHD.shape)
print(dataD2.shape)


# Constants ------------------------------
# these are used for scaling the coefs
scale1 = 1e4
scale2 = 1e7
scale3 = 1e9
scale4 = 1e12
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
    return  (error_mat)

#------------------------------------------------

def inverse_square(array):
    out=np.zeros(( array.shape[0], array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i,j]=1/(array[i,j]**2)
    return out        
    
#------------------------------------------------
  
def gen_s_linear(computed_data, param ):
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
            # param[1] = c1
            # param[0] = temperature
            
            mat [i,j]=(1+ (param[1]/scale1)*v1 )/ \
                (1+ (param[1]/scale1)*v2) 

    return mat

#------------------------------------------------    
def gen_s_quadratic(computed_data, param ):
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
            # param[1] = c1
            # param[2] = c2
            # param[0] = temperature
            
            mat [i,j]=(1+(param[1]/scale1)*v1 + (param[2]/scale2)*v1**2 )/ \
                (1+(param[1]/scale1)*v2 + (param[2]/scale2)*v2**2 )

    return mat

#------------------------------------------------ 
def gen_s_cubic(computed_data, param ):
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1=computed_data[i,1]
            v2=computed_data[j,1]
            
            # param[1] = c1 
            # param[2] = c2 
            # param[3] = c3 
            # param[0] = temperature
            
            mat [i,j]=(1+(param[1]/scale1)*v1 + (param[2]/scale2)*v1**2 +\
                       (param[3]/scale3)*v1**3 )/ \
                (1+(param[1]/scale1)*v2 + (param[2]/scale2)*v2**2 \
                 + (param[3]/scale3)*v2**3 ) 
                    
    return mat

#------------------------------------------------ 
def gen_s_quartic(computed_data, param, scale1):
    mat=np.zeros((computed_data.shape[0],computed_data.shape[0]))
    #print(mat.shape)
    
    for i in range(computed_data.shape[0]):
        for j in range(computed_data.shape[0]):
            v1=computed_data[i,1]
            v2=computed_data[j,1]

            # param[1] = c1 
            # param[2] = c2 
            # param[3] = c3 
            # param[4] = c4 
            # param[0] = temperature
            
            mat [i,j]=(1+(param[1]/scale1)*v1 + (param[2]/scale2)*v1**2 +\
                       (param[3]/scale3)*v1**3 + (param[4]/scale4)*v1**4  )/ \
                (1+(param[1]/scale1)*v2 + (param[2]/scale2)*v2**2 \
                 + (param[3]/scale3)*v2**3 + (param[4]/scale4)*v2**4 ) 

    return mat

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
    
I_D2=clean_mat(I_D2)
I_HD=clean_mat(I_HD)
I_H2=clean_mat(I_H2)


print(I_H2)

print("number of non zero in H2 :")
print (np.count_nonzero(I_H2), "\n")
print("number of non zero in HD :")
print (np.count_nonzero(I_HD), "\n")

print("number of non zero in D2 :")
print (np.count_nonzero(I_D2), "\n")


errH2_output=gen_weight(dataH2, 0.1)
errHD_output=gen_weight(dataHD, 0.2)
errD2_output=gen_weight(dataD2, 0.2)

param_linear=np.zeros((2))
param_linear[0]=298
param_linear[1]=-1.044940

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
#      GENERATE WEIGHT MATRICES 

wMat_H2 = gen_weight(dataH2, 0.1)
wMat_HD = gen_weight(dataHD, 0.2)
wMat_D2 = gen_weight(dataD2, 0.2)


#*******************************************************************

#*******************************************************************
# Define the residual function
#*******************************************************************

def residual_linear(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x ) 
    
    param : T, c1
    
    '''
    TK = param[0]
    #c1 = param[1]
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
    
    sD2=gen_s_linear(computed_D2, param)
    sHD=gen_s_linear(computed_HD, param)
    sH2=gen_s_linear(computed_H2, param)
    
    # residual matrix
    eD2 = ( np.multiply( wMat_D2, I_D2 )) - sD2 
    eHD = ( np.multiply( wMat_HD, I_HD )) - sHD 
    eH2 = ( np.multiply( wMat_H2, I_H2 )) - sH2 
    
    eD2=clean_mat(eD2)
    eHD=clean_mat(eHD)
    eH2=clean_mat(eH2)
    
    E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
        np.sum(np.abs(eH2))
        
    return(E)

#*******************************************************************

    
    
    
print(param_linear)
#cProfile.run('residual_linear(param_linear)')