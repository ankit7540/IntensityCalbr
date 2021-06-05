#*******************************************************************************

@utils.MeasureTime
def H2_Q1_vec(T, JMax, sos):
    '''compute the intensity for H2 Q-branch upto given JMax and sum of state '''

    specH2 = np.zeros(shape=(JMax+1, 4))
    v0     = np.zeros(shape=(JMax+1,1))
    v1     = np.zeros(shape=(JMax+1,1))
    energy     = np.zeros(shape=(JMax+1,1))
    gammaVec     = np.zeros(shape=(JMax+1))
    alphaVec     = np.zeros(shape=(JMax+1))
    popn = np.zeros(shape=(JMax+1))
    bj = np.zeros(shape=(JMax+1))
    
    
    i=np.arange(0,JMax+1,1)

    
    v0         = np.take(eJH2v0,i)
    v1         = np.take(eJH2v1,i)
    
    position   = (v1-v0)
    
    energy     = (-1*v0*H*C)
    #print(i, position, energy)
    popn       = (2*i+1)*np.exp(energy/(K*T))
    bj         = (i*(i+1))/((2*i-1)*(2*i+3))
    
    alpha = ME_alpha_H2_532_Q1[:,4]
    gamma = ME_gamma_H2_532_Q1[:,4]
    
    alphaVec = assign_v (alpha, JMax+1)
    gammaVec = assign_v (gamma, JMax+1)
    
    factor = (popn/sos)*omega_sc*((omega_sc-position/1e4)**3)*\
                (bj*(gammaVec**2)+ alphaVec**2)
    
    specH2[:,1]= position
    specH2[:,0]= i
    specH2[:,2] = factor  # unnormalized intensity, arbitrary unit
    specH2[:,3] = (omega-position)


    return specH2
#*******************************************************************************