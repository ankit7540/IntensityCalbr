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


################ EDIT FOLLOWING BLOCK  ##################

# LOAD EXPERIMENTAL BAND AREA DATA

# see readme for data formatting for these expt. data
# Do not change the variable name on the LHS 

dataH2 = np.loadtxt("./BA_H2_1.txt")
dataHD = np.loadtxt("./BA_HD_1.txt")
dataD2 = np.loadtxt("./BA_D2_1.txt")
xaxis  = np.loadtxt("./Wavenumber_axis_pa.txt")

# Jlevels information for the three gases
#  This is required to correspond to the input expt band area provided above
#  see readme and examples for more details
H2_aSJmax = 5
H2_SJmax = 5

HD_aSJmax = 5
HD_SJmax = 5

D2_aSJmax = 7
D2_SJmax = 7
# ----------------------------------------

# data format for O2 differs from that of H2 and isotopologues 
# see readme for more details
# Do not change the variable name on the LHS 
dataO2 = np.loadtxt("./DataO2_o1s1.txt")
dataO2_p = np.loadtxt("./DataO2_pR.txt")

# Constants ------------------------------
# these are used for scaling the coefs
# do not change the variable name on the LHS 
scale1 = 1e4
scale2 = 1e7
scale3 = 1e9
scale4 = 1e12
# ----------------------------------------

# norm type 
# Do not change the variable name on the LHS 
# available norm types : Frobenius, Frobenius_sq, absolute
# lower case :           frobenius, frobenius_sq, absolute
# or abbreviations:      F  , FS , A

norm =  'Frobenius'

# if norm is not set then the default is sum of absolute values 
# See readme for more details


# these are used for scaling the weights for O2 as needed
# Do not change the variable name on the LHS 

scale_O2_S1O1 = 0.5
scale_O2_pureRotn= 0.5

# weight = 1.0 means that the net uncertainty depends on the 
#          error of the band

#  weight = 0.0 means that the bands are not included 
#           in the fit altogether

# ----------------------------------------

# enable this to check the shape of the imported data

#print(dataH2.shape)
#print(dataHD.shape)
#print(dataD2.shape)



################ DO NOT EDIT BELOW THIS LINE  ##################
################################################################

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
log.info('================================================' )  
log.error("------------ Run log ------------\n")
# ------------------------------------------------------


print('\t**********************************************************')

print('\t ')
print('\t This module is for generating the wavenumber-dependent')
print('\t intensity correction curve termed as C2 from ')
print('\t  experimental Raman intensities. ')

print('\n\t This modeule requires edit on line 17 to 54 to ')
print('\n\t  load and set parameters for the analysis.')
print('\t ')
print('\t**********************************************************')
print('\n\t\t Checking imported data and set params')

data_error=0

if isinstance(dataH2, np.ndarray):
    print("\t\t ", "dataH2 found, OK")
else:
    print("\t\t ", "dataH2 not found.")
    data_error=1
    
if isinstance(dataHD, np.ndarray):
    print("\t\t ", "dataHD found, OK")
else:
    print("\t\t ", "dataHD not found.")
    data_error=1
    
if isinstance(dataD2, np.ndarray):
    print("\t\t ", "dataD2 found, OK")
else:
    print("\t\t ", "dataD2 not found.")
    data_error=1
    
if isinstance(xaxis, np.ndarray):
    print("\t\t ", "xaxis found, OK")
else:
    print("\t\t ", "xaxis not found.")
    data_error=1
    
if isinstance(dataO2, np.ndarray):
    print("\t\t ", "dataO2 found, OK")
else:
    print("\t\t ", "dataO2 not found.")
    data_error=1
    
if isinstance(dataO2_p, np.ndarray):
    print("\t\t ", "dataO2_p found, OK")
else:
    print("\t\t ", "dataO2_p not found.")
    data_error=1

print('\n\t\t  Analysis parameters:')

print("\t\t scaling factors (for c1 to c3)", scale1, scale2, scale3)
print("\t\t Norm (defn of residual): ", norm)

print("\t\t Scaling factor for O2 (ro-vibrn, O1 and S1): ", scale_O2_S1O1)
print("\t\t Scaling factor for O2 (pure rotn): ", scale_O2_pureRotn)


print('\t**********************************************************')
print('\n\t REQUIRED DATA')
print('\t\t\t Ramanshift = vector, the x-axis in relative wavenumbers')
print('\t\t\t band area and error = 2D (2 columns), for H2, HD and D2')
print('\n\t\t\t J_max = scalar, for H2, HD and D2 (to compute reference spectra)')

print('\t\t\t band area and error = 2D (6 columns), for O2, pure rotation ')
print('\t\t\t                            and for rotation-vibration bands ')



print('\t**********************************************************')

print('\n\t\t\t  Example:')

print('\t\t\t  run_fit_linear ( 299, 0.0 )')

print('\t\t\t  run_fit_quadratic ( 299, 0.05 ,0.02 )')


print('\t**********************************************************')

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
log.info('\t\t O2_O1S1:\t %s\n', dataO2.shape )
log.info('\t\t O2_pureRotn:\t %s\n', dataO2_p.shape )

log.info('\n\t Parameters:')
log.info('\t\t Norm:\t %s', norm)
log.info('\t\t Scaling factor (O2 O1-S1):\t %s\n', scale_O2_S1O1 )
log.info('\t\t Scaling factor (O2 pure rotation):\t %s\n', scale_O2_pureRotn )

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

                      1st column is the corresponding error
    """
    error_mat=np.zeros((expt_data.shape[0],expt_data.shape[0]))

    for i in range(expt_data.shape[0]):
        for j in range(expt_data.shape[0]):
            error_mat [i,j]=(expt_data[i,0]/expt_data[j,0])*\
                math.sqrt( (expt_data[i,1]/expt_data[i,0])**2 + \
                     (expt_data[j,1]/expt_data[j,0])**2   )


    return clean_mat(factor * inverse_square(error_mat))

#------------------------------------------------

def inverse_square(array):
    '''Returns the array where the elements are
    square of the inverse'''
    out=np.zeros(( array.shape[0], array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i,j]=1/(array[i,j]**2)
    return out

#------------------------------------------------

def gen_s_linear(computed_data, param ):
    '''Generates the S-matrix assuming linear function 
    for the wavelength dependent sensitivity'''
    
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
    '''Generates the S-matrix assuming quadratic function 
    for the wavelength dependent sensitivity'''
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
    '''Generates the S-matrix assuming cubic function 
    for the wavelength dependent sensitivity'''
        
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
    '''Generates the S-matrix assuming quartic function 
    for the wavelength dependent sensitivity'''    
    
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
#------------------------------------------------

#  GENERATE WEIGHT MATRICES

wMat_H2 = gen_weight(dataH2, 1e-6)
wMat_HD = gen_weight(dataHD, 1e-6)
wMat_D2 = gen_weight(dataD2, 1e-6)

#  or below we can set all weights as unity
#   that is no specific band ratio is preferred
wMat_H2 = 1
wMat_HD = 1
wMat_D2 = 1

log.info('\n\t Weights:')
if isinstance(wMat_H2, np.ndarray):
    log.info('\t\t Weight for H2 is array, size :\t %s', wMat_H2.shape)
else:
    log.info('\t\t Weight for H2  :\t %s', wMat_H2 )
    
if isinstance(wMat_HD, np.ndarray):
    log.info('\t\t Weight for HD is array, size :\t %s', wMat_HD.shape)
else:
    log.info('\t\t Weight for HD  :\t %s', wMat_HD )
    
if isinstance(wMat_D2, np.ndarray):
    log.info('\t\t Weight for D2 is array size :\t %s', wMat_D2.shape)
else:
    log.info('\t\t Weight for D2  :\t %s', wMat_D2 )    

log.info('================================================' )    

# ----------------------------------------

#*******************************************************************
# The residual functions are defined below
#*******************************************************************

def residual_linear(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x )

    param : T, c1

    '''
    #TK = param[0]
    TK = param[0]
    #c1 = param[1]
    computed_H2=compute_spectra.spectra_H2( TK, H2_aSJmax, H2_SJmax)
    computed_HD=compute_spectra.spectra_HD( TK, HD_aSJmax, HD_SJmax)
    computed_D2=compute_spectra.spectra_D2( TK, D2_aSJmax, D2_SJmax)


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


    # oxygen----------------------------
    # since information on all the polarizability invariants of the 
    # O2 are not avaiable, Raman intensities from common rotational 
    # states are utilized below
    
	# - O2 high frequency : 1400 to 1700 cm-1
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + param[0]/scale1 * dataO2[:, 3] )/ (1.0 +\
             param[0]/scale1 * dataO2[:, 4] )
    resd_O2 = ( dataO2[:, 5] * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------


    # - O2 pure rotation : -150 to +150 cm-1
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + param[0]/scale1 * dataO2_p[:, 3] )/ (1.0 +\
             param[0]/scale1 * dataO2_p[:, 4] )
    resd_O2p = (dataO2_p[:, 5] * scale_O2_pureRotn ) * ((ratio_O2p - RHS_O2p)**2)
	# ------


    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
            np.sum(np.abs(eH2)) +    np.sum(np.abs(resd_O2))  + np.sum(resd_O2p)    
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) +\
            np.sqrt(np.sum(np.square(eH2))) +   np.sqrt(np.sum(np.abs(resd_O2)))  +\
        np.sqrt(np.sum(resd_O2p))
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD)) +\
            np.sum(np.square(eH2)) +  np.sum(np.square(resd_O2))  + np.sum(np.square(resd_O2p))     

    return(E)

#*******************************************************************
#*******************************************************************

def residual_quadratic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 )

    param : T, c1, c2

    '''
    TK = param[0]
    #c1 = param[1]
    computed_H2=compute_spectra.spectra_H2( TK, H2_aSJmax, H2_SJmax)
    computed_HD=compute_spectra.spectra_HD( TK, HD_aSJmax, HD_SJmax)
    computed_D2=compute_spectra.spectra_D2( TK, D2_aSJmax, D2_SJmax)


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


    # oxygen----------------------------
    # since information on all the polarizability invariants of the 
    # O2 are not avaiable, Raman intensities from common rotational 
    # states are utilized below
    
	# - O2 high frequency : 1400 to 1700 cm-1
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + c1/scale1 * dataO2[:, 3] + c2/scale2 * (dataO2[:, 3]**2))/ (1.0 +\
             c1/scale1 * dataO2[:, 4] + c2/scale2 * (dataO2[:, 4]**2))

    resd_O2 = (dataO2[:, 5] * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------

    # - O2 pure rotation : -150 to +150 cm-1
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + c1/scale1 * dataO2_p[:, 3] + c2/scale2 * (dataO2_p[:, 3]**2))/ (1.0 +\
             c1/scale1 * dataO2_p[:, 4] + c2/scale2 * (dataO2_p[:, 4]**2))


    resd_O2p = (dataO2_p[:, 5]* scale_O2_pureRotn ) * ((ratio_O2p - RHS_O2p)**2)
	# ------

    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
            np.sum(np.abs(eH2)) +    np.sum(np.abs(resd_O2))  + np.sum(resd_O2p)    
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) +\
            np.sqrt(np.sum(np.square(eH2))) +   np.sqrt(np.sum(np.abs(resd_O2)))  +\
        np.sqrt(np.sum(resd_O2p))
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD)) +\
            np.sum(np.square(eH2)) +  np.sum(np.square(resd_O2))  + np.sum(np.square(resd_O2p))       

    return(E)

#*******************************************************************
#*******************************************************************

def residual_cubic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : T, c1, c2, c3

    '''
    TK = param[0]
    #c1 = param[1]
    computed_H2=compute_spectra.spectra_H2( TK, H2_aSJmax, H2_SJmax)
    computed_HD=compute_spectra.spectra_HD( TK, HD_aSJmax, HD_SJmax)
    computed_D2=compute_spectra.spectra_D2( TK, D2_aSJmax, D2_SJmax)


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


    # oxygen----------------------------
    # since information on all the polarizability invariants of the 
    # O2 are not avaiable, Raman intensities from common rotational 
    # states are utilized below
    
	# - O2 high frequency : 1400 to 1700 cm-1
    ratio_O2 = dataO2[:, 1]/dataO2[:, 2]
    RHS_O2 = (1.0 + c1/scale1 * dataO2[:, 3] + c2/scale2 * (dataO2[:, 3]**2)+\
              c3/scale3 * (dataO2[:, 3]**3))/ ( 1.0 + c1/scale1 * dataO2[:, 4] +\
                          c2/scale2 *(dataO2[:, 4]**2)+ c3/scale3 *( dataO2[:, 4]**3))

    resd_O2 =( dataO2[:, 5]  * scale_O2_S1O1 ) * ((ratio_O2 - RHS_O2)**2)
	# ------

    # - O2 pure rotation : -150 to +150 cm-1
    ratio_O2p = dataO2_p[:, 1]/dataO2_p[:, 2]
    RHS_O2p = (1.0 + c1/scale1 * dataO2_p[:, 3] + c2/scale2 * (dataO2_p[:, 3]**2)+\
               c3/scale3 * (dataO2_p[:, 3]**3))/ ( 1.0 + c1/scale1 * dataO2_p[:, 4] +\
                           c2/scale2 * (dataO2_p[:, 4]**2)+ c3/scale3 * ( dataO2_p[:, 4]**3))

    resd_O2p = (dataO2_p[:, 5] * scale_O2_pureRotn  ) * ((ratio_O2p - RHS_O2p)**2)
	# ------

    if norm=='' or norm.lower()=='absolute' or norm =='a' or norm =='A':
        E=np.sum(np.abs(eD2)) + np.sum(np.abs(eHD)) +\
            np.sum(np.abs(eH2)) +    np.sum(np.abs(resd_O2))  + np.sum(resd_O2p)    
        
    elif norm.lower()=='frobenius' or norm =='F'  :
        E=np.sqrt(np.sum(np.square(eD2))) + np.sqrt(np.sum(np.square(eHD))) +\
            np.sqrt(np.sum(np.square(eH2))) +   np.sqrt(np.sum(np.abs(resd_O2)))  +\
        np.sqrt(np.sum(resd_O2p))
        
    elif norm.lower()=='frobenius_square' or norm =='FS' :
        E=np.sum(np.square(eD2)) + np.sum(np.square(eHD)) +\
            np.sum(np.square(eH2)) +  np.sum(np.square(resd_O2))  + np.sum(np.square(resd_O2p))      


    return(E)

#***************************************************************
#***************************************************************
# Fit functions
#***************************************************************
#***************************************************************

def run_fit_linear ( init_T, init_k1 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1  ])
    print("**********************************************************")
    print("\t\t -- Linear fit -- ")
    print("\t\tNorm (defn of residual): ", norm)
    print("Initial coef :  T={0}, k1={1} output = {2}".format(init_T, init_k1, \
          (residual_linear(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_linear, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    print("\nOptimized result : T={0}, k1={1} \n".format(round(optT, 6) ,  round(optk1, 6) ))

    correction_curve_line= 1+(optk1/scale1)*xaxis     # generate the correction curve

    np.savetxt("correction_linear.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_linear', comments='')

    print("**********************************************************")
    print("\n C2 correction curve (as linear polynomial) saved as correction_linear.txt\n")
    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : T = %4.8f, c1 = %4.8f\n', init_T, init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f, c1 = %4.8f\n', optT, optk1 )
    log.info(' *******************************************')
    # --------------------

#***************************************************************
#***************************************************************
#***************************************************************

def run_fit_quadratic ( init_T, init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1 , init_k2  ])
    print("**********************************************************")
    print("\t\t -- Quadratic fit -- ")
    print("\t\tNorm (defn of residual): ", norm)
    print("Initial coef :  T={0}, k1={1}, k2={2} output = {3}".format(init_T, init_k1, \
         init_k2, (residual_quadratic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_quadratic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    optk2 = res.x[2]
    print("\nOptimized result : T={0}, k1={1}, k2={2} \n".format(round(optT, 6) ,  round(optk1, 6), round(optk2, 6) ))

    correction_curve_line= 1+(optk1/scale1)*xaxis  +(optk2/scale2)*xaxis**2    # generate the correction curve

    np.savetxt("correction_quadratic.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_quadratic', comments='')
    print("\n C2 correction curve (as quadratic polynomial) saved as quadratic_cubic.txt\n")
    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Quadratic  *******')
    log.info('\n\t Initial : T = %4.8f, c1 = %4.8f, c2 = %4.8f\n', init_T, init_k1, init_k2 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f, c1 = %4.8f, c2 = %4.8f\n', optT, optk1, optk2 )
    log.info(' *******************************************')
    # --------------------    

#***************************************************************


def run_fit_cubic ( init_T, init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
    print("\t\t -- Cubic fit -- ")
    print("\t\tNorm (defn of residual): ", norm)
    print("Initial coef :  T={0}, k1={1}, k2={2}, k3={3}, output = {3}".format(init_T, init_k1, \
         init_k2, init_k3, (residual_cubic(param_init))))


    print("\nOptimization run     \n")
    res = opt.minimize(residual_cubic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    optk2 = res.x[2]
    optk3 = res.x[3]
    print("\nOptimized result : T={0}, k1={1}, k2={2} \n".format(round(optT, 6) ,  round(optk1, 6), round(optk2, 6) ))

    correction_curve_line= 1+(optk1/scale1)*xaxis  +(optk2/scale2)*xaxis**2  +\
        +(optk3/scale3)*xaxis**3 # generate the correction curve

    np.savetxt("correction_cubic.txt", correction_curve_line, fmt='%2.8f',\
               header='corrn_curve_cubic', comments='')
    print("\n C2 correction curve (as cubic polynomial) saved as correction_cubic.txt\n")
    print("**********************************************************")
    # save log -----------
    log.info('\n *******  Optimization run : Cubic  *******')
    log.info('\n\t Initial : T = %4.8f, c1 = %4.8f, c2 = %4.8f, c3 = %4.8f\n', init_T, init_k1, init_k2, init_k3 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f, c1 = %4.8f, c2 = %4.8f, c3 = %4.8f\n', optT, optk1, optk2, optk3 )
    log.info(' *******************************************')
    # -------------------- 
#***************************************************************
