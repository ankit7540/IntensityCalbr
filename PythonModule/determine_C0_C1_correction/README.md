# Generating C<sub>0</sub>/C<sub>1</sub> correction

The module `gen_correction.py` allows one to generate the correction (C<sub>0</sub>/C<sub>1</sub>)

## Methodology
Observed intensities from selected bands are analyzed as pairs among all such bands, to form a matrix. A similar matrix of intensity ratios are compared to the true ratios, and the coefficients for the wavelength/wavenumber dependent sensitivity curve, modelled as a polynomial function, is obtained via non-linear minimization technique.

The general scheme is given as follows.
