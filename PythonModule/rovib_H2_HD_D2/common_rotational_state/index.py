# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:22:27 2020

@author: ankit
"""
import numpy as np
from common import compute_series_para


OJ_HD = 3
QJ_HD = 3
SJ_HD = 2

OJ_D2 = 4
QJ_D2 = 6
SJ_D2 = 3


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
# ------------------------------------------------


def clean_mat(square_array):
    """Set the upper triangular portion of square matrix to zero
        including the diagonal
        input = any square array     """
    np.fill_diagonal(square_array, 0)
    return np.tril(square_array, k=0)

# ------------------------------------------------    

def T_independent_index():

    TK = 298  #  ------------------------------------------
    sosD2 = compute_series_para.sumofstate_D2(TK)
    sosHD = compute_series_para.sumofstate_HD(TK)

    computed_D2 = compute_series_para.spectra_D2( TK, OJ_D2, QJ_D2, 
                                                 SJ_D2, sosD2)
    computed_HD = compute_series_para.spectra_HD( TK, OJ_HD, QJ_HD, 
                                                 SJ_HD, sosHD)

    calc_298_D2 = gen_intensity_mat(computed_D2, 2)
    calc_298_HD = gen_intensity_mat(computed_HD, 2)

    TK = 1000  #  ------------------------------------------
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
#diffD2 =  T_independent_index()    
index = T_independent_index()
indexD2=index[0]
indexHD=index[1]

test_mat = np.arange(196).reshape(14,14)
test_mat = clean_mat(test_mat)
test_mat[indexD2] = 0