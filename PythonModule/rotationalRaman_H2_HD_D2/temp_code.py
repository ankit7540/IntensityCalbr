# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:47:12 2020

@author: ankit
"""


def residual_linear(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x )

    param : T, c1

    '''

    TK = param[0]

    computed_D2_o1s1 = compute_series_para.spectra_D2_o1s1(TK, OJ_D2,
                                                           SJ_D2, sosD2)
    computed_D2_q1 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)
    
    computed_HD_o1s1 = compute_series_para.spectra_HD_o1s1(TK, OJ_HD,
                                                           SJ_HD, sosHD)
    computed_HD_q1 = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)
    
    computed_H2_o1 = compute_series_para.H2_O1(TK, OJ_H2, sosH2)
    computed_H2_q1 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # --------------------------------------------------
    #   generate the matrix of ratios 
    
    trueR_D2_o1s1 = gen_intensity_mat(computed_D2_o1s1, 2)
    expt_D2_o1s1 = gen_intensity_mat(dataD2_o1s1, 0)
    
    trueR_D2_q1 = gen_intensity_mat(computed_D2_q1, 2)
    expt_D2_q1 = gen_intensity_mat(dataD2_q1, 0)
    
    # --------------------------------------------------
    trueR_HD_o1s1 = gen_intensity_mat(computed_HD_o1s1, 2)
    expt_HD_o1s1 = gen_intensity_mat(dataHD_o1s1, 0)
    
    trueR_HD_q1 = gen_intensity_mat(computed_HD_q1, 2)
    expt_HD_q1 = gen_intensity_mat(dataHD_q1, 0)
    
    # --------------------------------------------------
    trueR_H2_o1 = gen_intensity_mat(computed_H2_o1, 2)
    expt_H2_o1 = gen_intensity_mat(dataH2_o1, 0)
    
    trueR_H2_q1 = gen_intensity_mat(computed_H2_q1, 2)
    expt_H2_q1 = gen_intensity_mat(dataH2_q1, 0)

    # --------------------------------------------------
    
    # generate sensitivity matrix using true data
    
    sD2_q1 = gen_s_linear(computed_D2_q1, param_linear)
    sHD_q1 = gen_s_linear(computed_HD_q1, param_linear)
    sH2_q1 = gen_s_linear(computed_H2_q1, param_linear)
    
    
    sD2_o1s1 = gen_s_linear(computed_D2_o1s1, param_linear)
    sHD_o1s1 = gen_s_linear(computed_HD_o1s1, param_linear)
    sH2_o1 = gen_s_linear(computed_H2_o1, param_linear)
    
    # --------------------------------------------------
    eD2_q1 = (np.multiply(errD2_q1_output, I_D2_q1) - sD2_q1)
    eHD_q1 = (np.multiply(errHD_q1_output, I_HD_q1) - sHD_q1)
    eH2_q1 = (np.multiply(errH2_q1_output, I_H2_q1) - sH2_q1)
    
    eD2_o1s1 = (np.multiply(errD2_o1s1_output, I_D2_o1s1) - sD2_o1s1)
    eHD_o1s1 = (np.multiply(errHD_o1s1_output, I_HD_o1s1) - sHD_o1s1)
    eH2_o1 = (np.multiply(errH2_o1_output, I_H2_o1) - sH2_o1)

    eD2_o1s1 = clean_mat(eD2_o1s1)
    eD2_q1 = clean_mat(eD2_q1)
    
    eHD_o1s1 = clean_mat(eHD_o1s1)
    eHD_q1 = clean_mat(eHD_q1)
    
    eH2_q1 = clean_mat(eH2_q1)    
    eH2_o1 = clean_mat(eH2_o1)

    E = np.sum(np.abs(eD2_q1)) + np.sum(np.abs(eHD_q1)) \
        + np.sum(np.abs(eH2_q1)) + np.sum(np.abs(eD2_o1s1)) \
        + np.sum(np.abs(eHD_o1s1)) + + np.sum(np.abs(eH2_o1))

    return(E)

# *******************************************************************
# *******************************************************************


def residual_quadratic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 )

    param : T, c1, c2

    '''
    TK = param[0]

    computed_D2_o1s1 = compute_series_para.spectra_D2_o1s1(TK, OJ_D2,
                                                           SJ_D2, sosD2)
    computed_D2_q1 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    computed_HD_o1s1 = compute_series_para.spectra_HD_o1s1(TK, OJ_HD,
                                                           SJ_HD, sosHD)
    computed_HD_q1 = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)

    computed_H2_o1 = compute_series_para.H2_O1(TK, OJ_H2, sosH2)
    computed_H2_q1 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # --------------------------------------------------
    #   generate the matrix of ratios

    trueR_D2_o1s1 = gen_intensity_mat(computed_D2_o1s1, 2)
    expt_D2_o1s1 = gen_intensity_mat(dataD2_o1s1, 0)
    trueR_D2_q1 = gen_intensity_mat(computed_D2_q1, 2)
    expt_D2_q1 = gen_intensity_mat(dataD2_q1, 0)
    # --------------------------------------------------
    trueR_HD_o1s1 = gen_intensity_mat(computed_HD_o1s1, 2)
    expt_HD_o1s1 = gen_intensity_mat(dataHD_o1s1, 0)
    trueR_HD_q1 = gen_intensity_mat(computed_HD_q1, 2)
    expt_HD_q1 = gen_intensity_mat(dataHD_q1, 0)
    # --------------------------------------------------
    trueR_H2_o1 = gen_intensity_mat(computed_H2_o1, 2)
    expt_H2_o1 = gen_intensity_mat(dataH2_o1, 0)
    trueR_H2_q1 = gen_intensity_mat(computed_H2_q1, 2)
    expt_H2_q1 = gen_intensity_mat(dataH2_q1, 0)
    # --------------------------------------------------

    # generate sensitivity matrix using true data
    sD2_q1 = gen_s_linear(computed_D2_q1, param_quadratic)
    sHD_q1 = gen_s_linear(computed_HD_q1, param_quadratic)
    sH2_q1 = gen_s_linear(computed_H2_q1, param_quadratic)

    sD2_o1s1 = gen_s_linear(computed_D2_o1s1, param)
    sHD_o1s1 = gen_s_linear(computed_HD_o1s1, param)
    sH2_o1 = gen_s_linear(computed_H2_o1, param)
    # --------------------------------------------------
    eD2_q1 = (np.multiply(errD2_q1_output, I_D2_q1) - sD2_q1)
    eHD_q1 = (np.multiply(errHD_q1_output, I_HD_q1) - sHD_q1)
    eH2_q1 = (np.multiply(errH2_q1_output, I_H2_q1) - sH2_q1)

    eD2_o1s1 = (np.multiply(errD2_o1s1_output, I_D2_o1s1) - sD2_o1s1)
    eHD_o1s1 = (np.multiply(errHD_o1s1_output, I_HD_o1s1) - sHD_o1s1)
    eH2_o1 = (np.multiply(errH2_o1_output, I_H2_o1) - sH2_o1)

    eD2_o1s1 = clean_mat(eD2_o1s1)
    eD2_q1 = clean_mat(eD2_q1)

    eHD_o1s1 = clean_mat(eHD_o1s1)
    eHD_q1 = clean_mat(eHD_q1)

    eH2_q1 = clean_mat(eH2_q1)
    eH2_o1 = clean_mat(eH2_o1)

    E = np.sum(np.abs(eD2_q1)) + np.sum(np.abs(eHD_q1)) \
        + np.sum(np.abs(eH2_q1)) + np.sum(np.abs(eD2_o1s1)) \
        + np.sum(np.abs(eHD_o1s1)) + + np.sum(np.abs(eH2_o1))

    return(E)


# *******************************************************************
# *******************************************************************


def residual_cubic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 )

    param : T, c1, c2, c3

    '''
    TK = param[0]

    computed_D2_o1s1 = compute_series_para.spectra_D2_o1s1(TK, OJ_D2,
                                                           SJ_D2, sosD2)
    computed_D2_q1 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    computed_HD_o1s1 = compute_series_para.spectra_HD_o1s1(TK, OJ_HD,
                                                           SJ_HD, sosHD)
    computed_HD_q1 = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)

    computed_H2_o1 = compute_series_para.H2_O1(TK, OJ_H2, sosH2)
    computed_H2_q1 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # --------------------------------------------------
    #   generate the matrix of ratios

    trueR_D2_o1s1 = gen_intensity_mat(computed_D2_o1s1, 2)
    expt_D2_o1s1 = gen_intensity_mat(dataD2_o1s1, 0)
    trueR_D2_q1 = gen_intensity_mat(computed_D2_q1, 2)
    expt_D2_q1 = gen_intensity_mat(dataD2_q1, 0)
    # --------------------------------------------------
    trueR_HD_o1s1 = gen_intensity_mat(computed_HD_o1s1, 2)
    expt_HD_o1s1 = gen_intensity_mat(dataHD_o1s1, 0)
    trueR_HD_q1 = gen_intensity_mat(computed_HD_q1, 2)
    expt_HD_q1 = gen_intensity_mat(dataHD_q1, 0)
    # --------------------------------------------------
    trueR_H2_o1 = gen_intensity_mat(computed_H2_o1, 2)
    expt_H2_o1 = gen_intensity_mat(dataH2_o1, 0)
    trueR_H2_q1 = gen_intensity_mat(computed_H2_q1, 2)
    expt_H2_q1 = gen_intensity_mat(dataH2_q1, 0)
    # --------------------------------------------------

    #   take ratio of expt to calculated

    I_D2_q1 = np.divide(expt_D2_q1, trueR_D2_q1)
    I_D2_o1s1 = np.divide(expt_D2_o1s1, trueR_D2_o1s1)

    I_HD_q1 = np.divide(expt_HD_q1, trueR_HD_q1)
    I_HD_o1s1 = np.divide(expt_HD_o1s1, trueR_HD_o1s1)

    I_H2_q1 = np.divide(expt_H2_q1, trueR_H2_q1)
    I_H2_o1 = np.divide(expt_H2_o1, trueR_H2_o1)

    #   remove redundant elements
    I_D2_q1 = clean_mat(I_D2_q1)
    I_HD_q1 = clean_mat(I_HD_q1)
    I_H2_q1 = clean_mat(I_H2_q1)

    I_D2_o1s1 = clean_mat(I_D2_o1s1)
    I_HD_o1s1 = clean_mat(I_HD_o1s1)
    I_H2_o1 = clean_mat(I_H2_o1)
    # --------------------------------------------------

    # generate sensitivity matrix using true data
    sD2_q1 = gen_s_cubic(computed_D2_q1, param)
    sHD_q1 = gen_s_cubic(computed_HD_q1, param)
    sH2_q1 = gen_s_cubic(computed_H2_q1, param)

    sD2_o1s1 = gen_s_cubic(computed_D2_o1s1, param)
    sHD_o1s1 = gen_s_cubic(computed_HD_o1s1, param)
    sH2_o1 = gen_s_cubic(computed_H2_o1, param)
    # --------------------------------------------------
    eD2_q1 = (np.multiply(errD2_q1_output, I_D2_q1) - sD2_q1)
    eHD_q1 = (np.multiply(errHD_q1_output, I_HD_q1) - sHD_q1)
    eH2_q1 = (np.multiply(errH2_q1_output, I_H2_q1) - sH2_q1)

    eD2_o1s1 = (np.multiply(errD2_o1s1_output, I_D2_o1s1) - sD2_o1s1)
    eHD_o1s1 = (np.multiply(errHD_o1s1_output, I_HD_o1s1) - sHD_o1s1)
    eH2_o1 = (np.multiply(errH2_o1_output, I_H2_o1) - sH2_o1)
    # --------------------------------------------------

    eD2_o1s1 = clean_mat(eD2_o1s1)
    eD2_q1 = clean_mat(eD2_q1)

    eHD_o1s1 = clean_mat(eHD_o1s1)
    eHD_q1 = clean_mat(eHD_q1)

    eH2_q1 = clean_mat(eH2_q1)
    eH2_o1 = clean_mat(eH2_o1)

    # --------------------------------------------------

    E = np.sum(np.abs(eD2_q1)) + np.sum(np.abs(eHD_q1)) \
        + np.sum(np.abs(eH2_q1)) + np.sum(np.abs(eD2_o1s1)) \
        + np.sum(np.abs(eHD_o1s1)) + + np.sum(np.abs(eH2_o1))

    return(E)

# *******************************************************************
# *******************************************************************


def residual_quartic(param):
    '''Function which computes the residual (as sum of squares) comparing the
    ratio of expt to theoretical intensity ratio to the sensitivity  profile
    modelled as  a line, ( 1+ c1*x + c2*x**2 + c3*x**3 + c4*x**4 )

    param : T, c1, c2, c3, c4

    '''
    TK = param[0]

    computed_D2_o1s1 = compute_series_para.spectra_D2_o1s1(TK, OJ_D2,
                                                           SJ_D2, sosD2)
    computed_D2_q1 = compute_series_para.D2_Q1(TK, QJ_D2, sosD2)

    computed_HD_o1s1 = compute_series_para.spectra_HD_o1s1(TK, OJ_HD,
                                                           SJ_HD, sosHD)
    computed_HD_q1 = compute_series_para.HD_Q1(TK, QJ_HD, sosHD)

    computed_H2_o1 = compute_series_para.H2_O1(TK, OJ_H2, sosH2)
    computed_H2_q1 = compute_series_para.H2_Q1(TK, QJ_H2, sosH2)

    # --------------------------------------------------
    #   generate the matrix of ratios

    trueR_D2_o1s1 = gen_intensity_mat(computed_D2_o1s1, 2)
    expt_D2_o1s1 = gen_intensity_mat(dataD2_o1s1, 0)
    trueR_D2_q1 = gen_intensity_mat(computed_D2_q1, 2)
    expt_D2_q1 = gen_intensity_mat(dataD2_q1, 0)
    # --------------------------------------------------
    trueR_HD_o1s1 = gen_intensity_mat(computed_HD_o1s1, 2)
    expt_HD_o1s1 = gen_intensity_mat(dataHD_o1s1, 0)
    trueR_HD_q1 = gen_intensity_mat(computed_HD_q1, 2)
    expt_HD_q1 = gen_intensity_mat(dataHD_q1, 0)
    # --------------------------------------------------
    trueR_H2_o1 = gen_intensity_mat(computed_H2_o1, 2)
    expt_H2_o1 = gen_intensity_mat(dataH2_o1, 0)
    trueR_H2_q1 = gen_intensity_mat(computed_H2_q1, 2)
    expt_H2_q1 = gen_intensity_mat(dataH2_q1, 0)
    # --------------------------------------------------

    #   take ratio of expt to calculated

    I_D2_q1 = np.divide(expt_D2_q1, trueR_D2_q1)
    I_D2_o1s1 = np.divide(expt_D2_o1s1, trueR_D2_o1s1)

    I_HD_q1 = np.divide(expt_HD_q1, trueR_HD_q1)
    I_HD_o1s1 = np.divide(expt_HD_o1s1, trueR_HD_o1s1)

    I_H2_q1 = np.divide(expt_H2_q1, trueR_H2_q1)
    I_H2_o1 = np.divide(expt_H2_o1, trueR_H2_o1)

    #   remove redundant elements
    I_D2_q1 = clean_mat(I_D2_q1)
    I_HD_q1 = clean_mat(I_HD_q1)
    I_H2_q1 = clean_mat(I_H2_q1)

    I_D2_o1s1 = clean_mat(I_D2_o1s1)
    I_HD_o1s1 = clean_mat(I_HD_o1s1)
    I_H2_o1 = clean_mat(I_H2_o1)
    # --------------------------------------------------

    # generate sensitivity matrix using true data
    sD2_q1 = gen_s_quartic(computed_D2_q1, param)
    sHD_q1 = gen_s_quartic(computed_HD_q1, param)
    sH2_q1 = gen_s_quartic(computed_H2_q1, param)

    sD2_o1s1 = gen_s_quartic(computed_D2_o1s1, param)
    sHD_o1s1 = gen_s_quartic(computed_HD_o1s1, param)
    sH2_o1 = gen_s_quartic(computed_H2_o1, param)
    # --------------------------------------------------
    eD2_q1 = (np.multiply(errD2_q1_output, I_D2_q1) - sD2_q1)
    eHD_q1 = (np.multiply(errHD_q1_output, I_HD_q1) - sHD_q1)
    eH2_q1 = (np.multiply(errH2_q1_output, I_H2_q1) - sH2_q1)

    eD2_o1s1 = (np.multiply(errD2_o1s1_output, I_D2_o1s1) - sD2_o1s1)
    eHD_o1s1 = (np.multiply(errHD_o1s1_output, I_HD_o1s1) - sHD_o1s1)
    eH2_o1 = (np.multiply(errH2_o1_output, I_H2_o1) - sH2_o1)
    # --------------------------------------------------

    eD2_o1s1 = clean_mat(eD2_o1s1)
    eD2_q1 = clean_mat(eD2_q1)

    eHD_o1s1 = clean_mat(eHD_o1s1)
    eHD_q1 = clean_mat(eHD_q1)

    eH2_q1 = clean_mat(eH2_q1)
    eH2_o1 = clean_mat(eH2_o1)

    # --------------------------------------------------

    E = np.sum(np.abs(eD2_q1)) + np.sum(np.abs(eHD_q1)) \
        + np.sum(np.abs(eH2_q1)) + np.sum(np.abs(eD2_o1s1)) \
        + np.sum(np.abs(eHD_o1s1)) + + np.sum(np.abs(eH2_o1))

    return(E)

# *******************************************************************
# *******************************************************************
# Fit functions
# *******************************************************************
# *******************************************************************


def run_fit_linear(init_T, init_k1):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0}, k1={1} output = {2}".format(init_T, init_k1, \
          (residual_linear(param_init))))


    print("\nOptimization run: Linear     \n")
    res = opt.minimize(residual_linear, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    print("\nOptimized result : T={0}, k1={1} \n".format(round(optT, 6) ,
                                                         round(optk1, 6) ))

    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  # generate the correction curve

    np.savetxt("correction_linear.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_linear', comments='')

    print("**********************************************************")

    # save log -----------
    log.info('\n *******  Optimization run : Linear  *******')
    log.info('\n\t Initial : T = %4.8f, c1 = %4.8f\n', init_T, init_k1 )
    log.info('\n\t %s\n', res )
    log.info('\n Optimized result : T = %4.8f, c1 = %4.8f\n', optT, optk1 )
    log.info(' *******************************************')
    return res.fun
    # --------------------

# *******************************************************************
# *******************************************************************

def run_fit_quadratic ( init_T, init_k1, init_k2 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1 , init_k2  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0}, k1={1}, k2={2} output = {3}".format(init_T, init_k1, \
         init_k2, (residual_quadratic(param_init))))


    print("\nOptimization run: Quadratic     \n")
    res = opt.minimize(residual_quadratic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':1500})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    optk2 = res.x[2]
    print("\nOptimized result : T={0}, k1={1}, k2={2} \n".format(round(optT, 6)\
     ,  round(optk1, 6), round(optk2, 6) ))

     # generate the correction curve
    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  +(optk2/scale2)\
                                                       * (xaxis-scenter)**2

    np.savetxt("correction_quadratic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quadratic', comments='')

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

def run_fit_cubic ( init_T, init_k1, init_k2, init_k3 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([ init_T, init_k1 , init_k2 , init_k3  ])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0}, k1={1}, k2={2}, k3={3}, output = {4}".\
          format(init_T, init_k1, init_k2, init_k3, (residual_cubic(param_init))))


    print("\nOptimization run : Cubic     \n")
    res = opt.minimize(residual_cubic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':2500})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    optk2 = res.x[2]
    optk3 = res.x[3]
    print("\nOptimized result : T={0}, k1={1}, k2={2}, k3={3} \n".\
          format(round(optT, 6) ,  round(optk1, 6), round(optk2, 6),\
                 round(optk3, 6)))

    correction_curve = 1+(optk1/scale1)*(xaxis-scenter)  + (optk2/scale2)*(xaxis-scenter)**2  +\
        +(optk3/scale3)*(xaxis-scenter)**3 # generate the correction curve

    np.savetxt("correction_cubic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_cubic', comments='')

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

def run_fit_quartic ( init_T, init_k1, init_k2, init_k3, init_k4 ):
    '''Function performing the actual fit using the residual_linear function
    defined earlier '''

    # init_k1 : Intial guess

    param_init = np.array([init_T, init_k1 , init_k2 , init_k3, init_k4])
    print("**********************************************************")
    #print("Testing the residual function with data")
    print("Initial coef :  T={0}, k1={1}, k2={2}, k3={3}, k4={4} output = {5}".\
          format(init_T, init_k1, init_k2, init_k3, init_k4, (residual_cubic(param_init))))


    print("\nOptimization run : Quartic     \n")
    res = opt.minimize(residual_quartic, param_init, method='Nelder-Mead', \
                              options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter':1500})

    print(res)
    optT = res.x[0]
    optk1 = res.x[1]
    optk2 = res.x[2]
    optk3 = res.x[3]
    optk4 = res.x[4]
    print("\nOptimized result : T={0}, k1={1}, k2={2}, k3={3}, k4={4} \n".\
          format(round(optT, 6) ,  round(optk1, 6), round(optk2, 6),\
                 round(optk3, 6), round(optk4, 6)))

    # generate the correction curve
    correction_curve= 1+(optk1/scale1)*(xaxis-scenter)  +(optk2/scale2)*(xaxis-scenter)**2  +\
        +(optk3/scale3)*(xaxis-scenter)**3 +(optk4/scale4)*(xaxis-scenter)**4

    np.savetxt("correction_quartic.txt", correction_curve, fmt='%2.8f',\
               header='corrn_curve_quartic', comments='')

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
