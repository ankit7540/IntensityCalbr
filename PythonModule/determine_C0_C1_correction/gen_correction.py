
# This file has function(s) for determination of the C0 and C1 corrections




# fit function for the white light

def photons_per_unit_wavenum_abs(x,a,T) :
    return (a*599584916*w**2)/(exp(0.1438776877e-1*x/T)-1)
