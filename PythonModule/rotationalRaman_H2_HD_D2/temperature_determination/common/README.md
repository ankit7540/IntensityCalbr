# Contents

Python submodules which take in data from the folder 'energy_levels_and_ME' and
output computed spectra are included here.

Included scripts include,

 - `compute_pureRotation ` Calculating pure rotation spectra for a given T,
   normalized to unity
    - `H2`
    - `HD`
    - `D2`


 - `compute_highWavenum` Calculating the bands in the high wavenumber region
 comprising the O1, Q1 and S1 bands for a given T, normalized to unity.
    - `H2_s1`

 - `sumofstates` Calculating the sum of states required for Boltzmann population
 for a given T for H2, HD and D2.
